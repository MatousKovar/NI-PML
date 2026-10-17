"""Import the pinned upstream checkout without editing its source."""

import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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
