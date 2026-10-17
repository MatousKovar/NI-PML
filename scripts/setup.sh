#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PIP_CACHE_DIR="$PWD/.cache/pip"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export HF_HOME="$PWD/.cache/huggingface"

if [[ ! -x .venv/bin/python ]]; then
    python_command="${PYTHON_BIN:-python3.11}"
    if ! command -v "$python_command" >/dev/null 2>&1; then
        echo 'Install Python 3.11 first, or set PYTHON_BIN to its executable path.' >&2
        exit 1
    fi
    "$python_command" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
    "$python_command" -m venv .venv
fi

.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), "The .venv must use Python 3.11"'
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    .venv/bin/python -m ensurepip
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
carec_url="$(.venv/bin/python -c 'from benchmarks.carec import CAREC_URL; print(CAREC_URL)')"
carec_commit="$(.venv/bin/python -c 'from benchmarks.carec import CAREC_COMMIT; print(CAREC_COMMIT)')"
if [[ ! -d external/CARec ]]; then
    git clone --no-checkout "$carec_url" external/CARec
    git -C external/CARec checkout --detach "$carec_commit"
fi

.venv/bin/python -m ipykernel install --prefix .venv --name ni-pml --display-name 'NI-PML benchmarks'
.venv/bin/python -m benchmarks.check_environment
