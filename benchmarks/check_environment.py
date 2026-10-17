"""Check dependencies, bounded raw-data reads, and the upstream graph layer."""

import argparse
import importlib.metadata
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from benchmarks.carec import ROOT, load_carec


def inspect_data(data_dir: Path):
    fields = {
        "Electronics.json": {"reviewerID", "asin", "overall", "unixReviewTime"},
        "meta_Electronics.json": {"asin", "title", "brand", "category"},
    }
    result = {}
    for name, required in fields.items():
        path = data_dir / name
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in itertools.islice(stream, 10)]
        if not rows:
            raise ValueError(f"Empty data file: {path}")
        missing = required - rows[0].keys()
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        result[name] = {"bytes": path.stat().st_size, "sample_rows": len(rows)}
    return result


def check_graph(device):
    import torch
    from recbole_gnn.model.layers import LightGCNConv

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)
    edges = torch.tensor([[0, 1], [1, 0]], device=device)
    weights = torch.ones(2, device=device)
    result = LightGCNConv(dim=2)(x, edges, weights)
    torch.testing.assert_close(result, x.flip(0))
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    return "forward and backward passed"


def check_model(upstream):
    import numpy as np
    from scipy.sparse import coo_matrix
    import torch
    from recbole.data.interaction import Interaction

    class TinyDataset:
        user_num, item_num = 3, 4
        uid_field, iid_field = "user_id", "item_id"

        def __init__(self):
            self.inter_feat = Interaction({
                "user_id": torch.tensor([1, 1, 2, 2]),
                "item_id": torch.tensor([1, 2, 2, 3]),
            })
            self.item_feat = SimpleNamespace(item_emb=torch.randn(self.item_num, 8))
            self.item_feat.item_emb[0] = 0

        def num(self, field):
            return {self.uid_field: self.user_num, self.iid_field: self.item_num}[field]

        def inter_matrix(self, form):
            return coo_matrix((
                np.ones(4),
                (self.inter_feat[self.uid_field].numpy(), self.inter_feat[self.iid_field].numpy()),
            ), shape=(self.user_num, self.item_num))

    torch.manual_seed(2023)
    data = TinyDataset()
    model = upstream.CollabContex({
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "NEG_PREFIX": "neg_",
        "device": torch.device("cpu"), "encoder": "LightGCN", "num_layers": 2,
        "embedding_size": 8,
        "mlp": {"input_dim": 8, "hidden_dims": [8, 8], "output_dim": 8,
                "num_layers": 2, "dropout": 0.2},
    }, data)
    losses = {}
    for phase in ("item", "user"):
        getattr(model, f"set_{phase}_tutoring_phase")()
        model.zero_grad(set_to_none=True)
        loss = model.calculate_loss(data.inter_feat)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite {phase} phase loss")
        loss.backward()
        losses[phase] = float(loss.detach())
    model.eval()
    with torch.no_grad():
        scores = model.full_sort_predict(Interaction({"user_id": torch.tensor([1, 2])}))
    assert scores.shape == (2 * data.item_num,)
    assert torch.isfinite(scores).all()
    return {"device": "cpu", "phase_losses": losses, "flat_score_shape": list(scores.shape)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()

    import torch
    from recbole.config import Config
    from recbole.trainer import Trainer
    from sentence_transformers import SentenceTransformer

    upstream = load_carec()
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch", "numpy", "recbole", "torch-geometric", "transformers", "sentence-transformers"
        )
    }
    report = {
        "versions": versions,
        "carec_commit": upstream.commit,
        "imports": [upstream.CollabContex.__name__, upstream.ColdEvaluator.__name__,
                    Config.__name__, Trainer.__name__, SentenceTransformer.__name__],
        "graph_cpu": check_graph("cpu"),
        "carec_tiny_model": check_model(upstream),
        "mps_available": torch.backends.mps.is_available(),
    }
    if report["mps_available"]:
        report["graph_mps"] = check_graph("mps")
    if not args.skip_data:
        report["data"] = inspect_data(args.data_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
