"""Shared CARec experiment code for the native and compression notebooks."""

import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from copy import copy, deepcopy
from datetime import datetime, timezone
import json
from time import perf_counter

from IPython.display import display
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from recbole.config import Config
from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.sampler import Sampler
from recbole.evaluator import Collector, Evaluator
from recbole.utils import init_seed
from recbole.utils.enum_type import FeatureSource, FeatureType

ROOT = Path(__file__).resolve().parents[1]
CAREC_URL = "https://github.com/ChenMetanoia/CARec.git"
CAREC_COMMIT = "bbfc7f385e796840f6eb5c967dfc8714284958ce"
CAREC_ROOT = ROOT / "external" / "CARec"


def load_carec():
    if not (CAREC_ROOT / ".git").exists():
        raise RuntimeError("CARec checkout is missing. Run bash scripts/setup.sh first.")
    revision = subprocess.check_output(
        ["git", "-C", str(CAREC_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != CAREC_COMMIT:
        raise RuntimeError(f"Expected CARec {CAREC_COMMIT}, found {revision}.")
    changes = subprocess.check_output(
        ["git", "-C", str(CAREC_ROOT), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if changes.strip():
        raise RuntimeError("CARec has local source changes. Record them before benchmarking.")

    # Upstream imports generic top-level names such as model and utils.
    for name in ("model", "utils", "recbole_gnn"):
        module = sys.modules.get(name)
        if module is not None:
            paths = list(getattr(module, "__path__", []))
            if not paths or any(not Path(p).resolve().is_relative_to(CAREC_ROOT) for p in paths):
                raise RuntimeError(f"Import name {name!r} is already in use. Restart the kernel.")

    if str(CAREC_ROOT) not in sys.path:
        sys.path.insert(0, str(CAREC_ROOT))
    model = importlib.import_module("model.collabcontex")
    util = importlib.import_module("utils.util")
    return SimpleNamespace(
        CollabContex=model.CollabContex,
        LGCNEncoder=model.LGCNEncoder,
        MLP=model.MLP,
        ColdEvaluator=util.ColdEvaluator,
        data_preparation=util.data_preparation,
        root=CAREC_ROOT,
        commit=revision,
    )


RESULT_FILES = {"CARec": "carec.json", "Content-only": "content-only.json"}


class Experiment:
    """Load one fixed split and reuse its settings across embedding conditions."""

    def __init__(
        self, project_root, dataset_id="electronics_carec_035_seed2023", *,
        device="mps", seed=2023, phase1_epochs=90, phase2_epochs=30,
        patience=15, weight_decay=0.0, training_batch_size=2048, learning_rate=0.001,
    ):
        self.project_root = Path(project_root)
        self.dataset_id = dataset_id
        self.device = torch.device(device)
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable.")
        self.seed = seed
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.training_batch_size = training_batch_size
        self.learning_rate = learning_rate
        self.prepared_dir = self.project_root / "data/processed" / dataset_id
        self.recbole_root = self.project_root / "artifacts/recbole"
        self.recbole_dir = self.recbole_root / dataset_id
        self.embedding_dir = self.project_root / "artifacts/embeddings" / dataset_id
        self.runs_dir = self.project_root / "artifacts/carec" / dataset_id
        self.results_dir = self.project_root / "results/carec" / dataset_id
        self.carec = load_carec()
        self._load_data()

    def _load_data(self):
        self.recbole_dir.mkdir(parents=True, exist_ok=True)
        interaction_columns = ["user_id:token", "item_id:token", "timestamp:float"]
        for split_name in ("train", "valid", "test", "cold"):
            split_path = self.recbole_dir / f"{self.dataset_id}.{split_name}.inter"
            if not split_path.exists():
                interactions = pd.read_csv(
                    self.prepared_dir / f"{split_name}.tsv.gz",
                    sep="\t",
                    usecols=interaction_columns,
                    dtype={"user_id:token": str, "item_id:token": str},
                    float_precision="round_trip",
                )
                # RecBole requires typed headers. Leave source_row out of this export.
                interactions[interaction_columns].to_csv(split_path, sep="\t", index=False)

        catalogue_path = self.recbole_dir / f"{self.dataset_id}.item"
        if not catalogue_path.exists():
            catalogue_ids = pd.read_csv(self.prepared_dir / "item_ids.tsv.gz", sep="\t", dtype={"item_id": str})
            pd.DataFrame({"item_id:token": catalogue_ids["item_id"]}).to_csv(
                catalogue_path, sep="\t", index=False,
            )

        dataset_settings = {
            "data_path": str(self.recbole_root),
            "benchmark_filename": ["train", "valid", "test", "cold"],
            "field_separator": "\t",
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "TIME_FIELD": "timestamp",
            "load_col": {
                "inter": ["user_id", "item_id", "timestamp"],
                "item": ["item_id"],
            },
            "normalize_all": False,
            "use_gpu": False,
        }

        self.base_config = Config(model=self.carec.CollabContex, dataset=self.dataset_id, config_dict=dataset_settings)
        # Set before construction: CARec stores graph and item tensors outside registered buffers.
        self.base_config["device"] = self.device
        self.recbole_dataset = Dataset(self.base_config)
        self.recbole_product_ids = self.recbole_dataset.field2id_token["item_id"]
        self.train_dataset, self.valid_dataset, self.test_dataset, self.cold_dataset = self.recbole_dataset.build()

        self.base_config["metrics"] = ["Recall", "NDCG"]
        self.base_config["topk"] = [10, 50]
        self.base_config["eval_batch_size"] = 64 * self.train_dataset.item_num

        # Upstream repeatable sampling leaves history empty. Supply cumulative history.
        history_sampler = Sampler(
            ["train", "valid", "test", "cold"],
            [self.train_dataset, self.valid_dataset, self.test_dataset, self.cold_dataset],
        )
        self.evaluation_loaders = {}
        for split_name, split in [("valid", self.valid_dataset), ("warm", self.test_dataset), ("cold", self.cold_dataset)]:
            phase = "test" if split_name == "warm" else split_name
            self.evaluation_loaders[split_name] = FullSortEvalDataLoader(
                self.base_config, split, history_sampler.set_phase(phase), shuffle=False,
            )

        cold_product_ids = self.cold_dataset.inter_feat[self.base_config["ITEM_ID_FIELD"]].unique()
        train_product_ids = self.train_dataset.inter_feat[self.base_config["ITEM_ID_FIELD"]].unique()
        assert not set(cold_product_ids.tolist()) & set(train_product_ids.tolist())
        print({name: len(loader.uid_list) for name, loader in self.evaluation_loaders.items()})

    def prepare_model(self, embedding_name):
        embedding_path = self.embedding_dir / embedding_name
        with np.load(embedding_path, allow_pickle=False) as saved:
            item_ids = saved["item_ids"]
            vectors = saved["embeddings"]
            dimension = vectors.shape[1]
        rows = {product_id: row for row, product_id in enumerate(item_ids)}
        assert len(rows) == len(item_ids), "Duplicate cache IDs"
        assert len(vectors) == len(item_ids), "Cache row count differs"
        assert self.recbole_product_ids[0] == "[PAD]"
        assert set(self.recbole_product_ids[1:]) == set(item_ids), "Product catalogues differ"
        aligned = np.zeros((self.train_dataset.item_num, vectors.shape[1]), dtype=np.float32)
        for item_id, product_id in enumerate(self.recbole_product_ids[1:], start=1):
            aligned[item_id] = vectors[rows[product_id]]

        config = deepcopy(self.base_config)
        config["seed"] = self.seed
        config["encoder"] = "LightGCN"
        config["num_layers"] = 2
        config["embedding_size"] = dimension
        config["mlp"] = {
            "input_dim": dimension, "hidden_dims": [dimension, dimension],
            "output_dim": dimension, "num_layers": 2, "dropout": 0.2,
        }
        # Share immutable interactions; give this variant its own item features and metadata.
        model_dataset = copy(self.train_dataset)
        model_dataset.config = config
        for field in ("field2type", "field2source", "field2seqlen"):
            setattr(model_dataset, field, getattr(self.train_dataset, field).copy())
        model_dataset.set_field_property("item_emb", FeatureType.FLOAT_SEQ, FeatureSource.ITEM, dimension)
        model_dataset.item_feat = Interaction({
            **self.train_dataset.item_feat.interaction, "item_emb": torch.from_numpy(aligned),
        })
        init_seed(self.seed, config["reproducibility"])
        model = self.carec.CollabContex(config, model_dataset).to(self.device)
        encoder = model.encoder
        node_count = encoder.n_users + encoder.n_items
        adjacency = torch.sparse_coo_tensor(
            encoder.edge_index.flip(0).cpu(), encoder.edge_weight.cpu(), (node_count, node_count),
        ).coalesce().to_sparse_csr()

        def sparse_forward(node_vectors, edge_index, edge_weight):
            return torch.sparse.mm(adjacency, node_vectors.cpu()).to(node_vectors.device)

        def cpu_uniformity(vectors, t=2):
            return self.carec.CollabContex.uniformity(vectors.cpu(), t=t).to(vectors.device)

        encoder.gcn_conv.forward = sparse_forward
        model.uniformity = cpu_uniformity
        return model, config, dimension

    @torch.no_grad()
    def evaluate_carec(self, model, loader, config, cold_only=False):
        collector = Collector(config)
        item_count = loader.dataset.item_num
        evaluator = Evaluator(config)
        previous_training = model.training
        previous_mlp_bypass = model.encoder.not_apply_mlp
        model.eval()
        if cold_only:
            model.encoder.not_apply_mlp = True
        # full_sort_predict caches graph outputs. Never reuse them across modes or epochs.
        model.restore_user_e = model.restore_item_e = None
        candidate_mask = torch.ones(item_count, dtype=torch.bool, device=config["device"])
        if cold_only:
            candidate_mask[:] = False
            cold_ids = loader.dataset.inter_feat[config["ITEM_ID_FIELD"]].unique()
            candidate_mask[cold_ids.to(config["device"])] = True
        candidate_mask[0] = False

        try:
            for interaction, history, positive_users, positive_items in tqdm(loader, desc="Cold" if cold_only else "Full ranking"):
                scores = model.full_sort_predict(interaction.to(config["device"]))
                scores = scores.view(-1, item_count)
                if not torch.isfinite(scores).all():
                    raise ValueError("Non-finite recommendation scores before masking.")
                # Apply the item mask to columns, for every user in the batch.
                scores[:, ~candidate_mask] = -torch.inf
                if history is not None:
                    scores[history[0].to(scores.device), history[1].to(scores.device)] = -torch.inf
                collector.eval_batch_collect(scores, interaction, positive_users, positive_items)
            return {name: float(value) for name, value in evaluator.evaluate(collector.get_data_struct()).items()}
        finally:
            model.encoder.not_apply_mlp = previous_mlp_bypass
            model.restore_user_e = model.restore_item_e = None
            model.train(previous_training)

    def save_experiment_result(self, embedding_name, method, result):
        variant = Path(embedding_name).stem
        result_path = self.results_dir / variant / RESULT_FILES[method]
        result_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "dataset_id": self.dataset_id,
            "embedding_file": Path(embedding_name).name,
            "method": method,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(result_path)

    def load_experiment_results(self, method):
        results = {}
        for result_path in sorted(self.results_dir.glob(f"*/{RESULT_FILES[method]}")):
            record = json.loads(result_path.read_text())
            variant = result_path.parent.name
            if (record["schema_version"] != 1 or record["dataset_id"] != self.dataset_id
                    or record["method"] != method
                    or Path(record["embedding_file"]).stem != variant):
                raise ValueError(f"Result identity differs: {result_path}")
            results[variant] = record["result"]
        return results

    @torch.no_grad()
    def evaluate_content_only(self, embedding_name):
        started = perf_counter()
        config = deepcopy(self.base_config)
        config["device"] = torch.device("cpu")
        with np.load(self.embedding_dir / embedding_name, allow_pickle=False) as saved:
            item_ids = saved["item_ids"]
            vectors = saved["embeddings"]
        if vectors.ndim != 2 or len(vectors) != len(item_ids):
            raise ValueError("Cache must contain one vector per item ID.")
        rows = {product_id: row for row, product_id in enumerate(item_ids)}
        if len(rows) != len(item_ids):
            raise ValueError("Duplicate cache IDs.")
        if self.recbole_product_ids[0] != "[PAD]" or set(self.recbole_product_ids[1:]) != set(item_ids):
            raise ValueError("Cache and RecBole product catalogues differ.")
        aligned = np.zeros((self.train_dataset.item_num, vectors.shape[1]), dtype=np.float32)
        for item_id, product_id in enumerate(self.recbole_product_ids[1:], start=1):
            aligned[item_id] = vectors[rows[product_id]]
        if not np.isfinite(aligned).all():
            raise ValueError("Non-finite item embeddings.")
        item_vectors = torch.from_numpy(aligned)
        if (item_vectors[1:].norm(dim=1) == 0).any():
            raise ValueError("An item has a zero embedding.")

        # Preserve repeated training interactions as repeated contributions to the mean.
        history = self.train_dataset.inter_matrix(form="csr").astype(np.float32)
        counts = np.asarray(history.sum(axis=1)).ravel()
        mean_history = history.multiply((1.0 / np.maximum(counts, 1))[:, None]).tocsr()
        profiles = torch.from_numpy(np.asarray(mean_history @ aligned, dtype=np.float32))
        profile_norms = profiles.norm(dim=1)
        user_vectors = torch.nn.functional.normalize(profiles, dim=1)
        item_vectors = torch.nn.functional.normalize(item_vectors, dim=1)
        profile_seconds = perf_counter() - started
        del vectors, aligned, history, mean_history, profiles

        test = {}
        zero_profile_users = {}
        scoring_started = perf_counter()
        for split_name in ("warm", "cold"):
            loader = self.evaluation_loaders[split_name]
            collector = Collector(config)
            evaluator = Evaluator(config)
            candidate_mask = torch.ones(self.train_dataset.item_num, dtype=torch.bool)
            if split_name == "cold":
                candidate_mask[:] = False
                cold_ids = loader.dataset.inter_feat[config["ITEM_ID_FIELD"]].unique().cpu()
                candidate_mask[cold_ids] = True
            candidate_mask[0] = False
            zero_profile_users[split_name] = 0
            for interaction, history, positive_users, positive_items in tqdm(loader, desc=f"Content-only {split_name}"):
                user_ids = interaction[config["USER_ID_FIELD"]].cpu()
                zero_profile_users[split_name] += int((profile_norms[user_ids] == 0).sum())
                scores = user_vectors[user_ids] @ item_vectors.T
                if not torch.isfinite(scores).all():
                    raise ValueError("Non-finite content-only scores before masking.")
                scores[:, ~candidate_mask] = -torch.inf
                if history is not None:
                    scores[history[0].cpu(), history[1].cpu()] = -torch.inf
                collector.eval_batch_collect(scores, interaction, positive_users, positive_items)
            test[split_name] = {
                name: float(value)
                for name, value in evaluator.evaluate(collector.get_data_struct()).items()
            }
        result = {
            "dimension": item_vectors.shape[1], "test": test,
            "zero_profile_users": zero_profile_users,
            "profile_seconds": profile_seconds,
            "scoring_seconds": perf_counter() - scoring_started,
            "total_seconds": perf_counter() - started,
        }
        self.save_experiment_result(embedding_name, "Content-only", result)
        display(pd.DataFrame(test).T)
        print(f"Zero-profile users: {zero_profile_users}")
        return result

    def train_phase(self, model, config, phase, max_epochs, run_dir):
        if max_epochs < 1 or self.patience < 1:
            raise ValueError("Epoch limit and patience must be positive.")
        if phase == 1:
            model.set_item_tutoring_phase()
            uniformity_field = config["USER_ID_FIELD"]
        elif phase == 2:
            model.set_user_tutoring_phase()
            uniformity_field = config["ITEM_ID_FIELD"]
        else:
            raise ValueError("CARec has phases 1 and 2.")

        # Frozen parameters have no optimizer state in this phase.
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        generator = torch.Generator().manual_seed(self.seed + phase)
        checkpoint_path = run_dir / f"phase{phase}-best.pt"
        best_score, stale_epochs = -float("inf"), 0
        best_epoch, best_metrics = None, None
        history = []
        started = perf_counter()

        for epoch in range(1, max_epochs + 1):
            model.train()
            indices = torch.randperm(len(self.train_dataset.inter_feat), generator=generator)
            total_loss, interaction_count = 0.0, 0
            for batch_ids in tqdm(indices.split(self.training_batch_size), desc=f"Phase {phase}, epoch {epoch}"):
                batch = self.train_dataset.inter_feat[batch_ids].to(config["device"])
                if batch[uniformity_field].unique().numel() < 2:
                    raise ValueError("Uniformity needs at least two distinct active-side IDs.")
                optimizer.zero_grad(set_to_none=True)
                loss = model.calculate_loss(batch)
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is not finite.")
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
                interaction_count += len(batch)

            valid_metrics = self.evaluate_carec(model, self.evaluation_loaders["valid"], config)
            score = float(valid_metrics["ndcg@10"])
            if not np.isfinite(score):
                raise ValueError("Validation NDCG@10 is not finite.")
            improved = score > best_score
            stale_epochs = 0 if improved else stale_epochs + 1
            history.append({
                "epoch": epoch, "loss": total_loss / interaction_count,
                "valid": valid_metrics, "elapsed_seconds": perf_counter() - started,
            })
            if improved:
                best_score = score
                # Write a temporary file so interruption cannot truncate the previous best.
                temporary = checkpoint_path.with_suffix(".pt.tmp")
                best_epoch, best_metrics = epoch, valid_metrics
                torch.save(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    temporary,
                )
                temporary.replace(checkpoint_path)
            print(f"Phase {phase}, epoch {epoch}: loss={history[-1]['loss']:.4f}, valid NDCG@10={score:.4f}, patience={stale_epochs}/{self.patience}")
            if stale_epochs >= self.patience:
                break

        # Reload the best weights before the next phase or final evaluation.
        best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best)
        model.restore_user_e = model.restore_item_e = None
        model.eval()
        return {"best_epoch": best_epoch, "valid": best_metrics, "epochs_run": len(history)}

    def run_experiment(self, embedding_name, *, phase2_only=False):
        variant = Path(embedding_name).stem
        run_dir = self.runs_dir / variant
        checkpoints = sorted(path for path in run_dir.glob("*.pt") if path.is_file())
        if phase2_only and not (run_dir / "phase1-best.pt").is_file():
            raise FileNotFoundError(f"Phase 2 requires {run_dir / 'phase1-best.pt'}")
        started = perf_counter()
        model, config, dimension = self.prepare_model(embedding_name)
        if phase2_only and not (run_dir / "phase2-best.pt").is_file():
            checkpoint_path = run_dir / "phase1-best.pt"
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            model.restore_user_e = model.restore_item_e = None
            print(f"Phase 2 only: {embedding_name}, loaded {checkpoint_path.name}")
            training = {"phase1": {"best_epoch": None}}
            training["phase2"] = self.train_phase(model, config, 2, self.phase2_epochs, run_dir)
        elif checkpoints:
            checkpoint_path = next(
                (run_dir / name for name in ("phase2-best.pt", "phase1-best.pt")
                 if (run_dir / name).is_file()),
                None,
            )
            if checkpoint_path is None:
                raise ValueError(f"Checkpoint exists in {run_dir}, but no recognized phase checkpoint was found. Training skipped.")
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            if checkpoint_path.name == "phase2-best.pt":
                model.set_user_tutoring_phase()
            else:
                model.set_item_tutoring_phase()
            model.restore_user_e = model.restore_item_e = None
            model.eval()
            print(f"Evaluation only: {embedding_name}, loaded {checkpoint_path.name}")
            training = {"phase1": {"best_epoch": None}, "phase2": {"best_epoch": None}}
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"{embedding_name}: {dimension} dimensions, model {self.device}, graph CPU")
            print(f"Model directory: {run_dir}")
            training = {}
            training["phase1"] = self.train_phase(model, config, 1, self.phase1_epochs, run_dir)
            training["phase2"] = self.train_phase(model, config, 2, self.phase2_epochs, run_dir)
        test = {
            "warm": self.evaluate_carec(model, self.evaluation_loaders["warm"], config),
            "cold": self.evaluate_carec(model, self.evaluation_loaders["cold"], config, cold_only=True),
        }
        result = {
            "dimension": dimension, "training": training, "test": test,
            "total_seconds": perf_counter() - started, "run_dir": str(run_dir),
        }
        self.save_experiment_result(embedding_name, "CARec", result)
        print("CARec")
        display(pd.DataFrame(test).T)
        return result
