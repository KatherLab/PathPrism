#!/usr/bin/env bash
set -euo pipefail

# PathPrism one-shot installer
# - Creates conda env (preferred) or Python venv named "pathprism"
# - Installs PyTorch (CUDA if available, else CPU)
# - Installs Python dependencies from requirements.txt
# - Prints usage hints

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="pathprism"
PY_VERSION="3.9"

has_cmd() { command -v "$1" >/dev/null 2>&1; }

banner() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

choose_pytorch_index() {
  # Detect NVIDIA GPU presence for CUDA build; fallback to CPU
  if has_cmd nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    # Default to CUDA 11.8 wheels (broadly compatible)
    echo "https://download.pytorch.org/whl/cu118"
  else
    echo "https://download.pytorch.org/whl/cpu"
  fi
}

install_with_pip() {
  local python_bin="$1"
  local torch_index
  torch_index="$(choose_pytorch_index)"

  banner "Installing PyTorch"
  "$python_bin" -m pip install --upgrade pip setuptools wheel
  # Install core PyTorch; torchvision is not needed for MacroNet but may be useful
  "$python_bin" -m pip install --index-url "$torch_index" torch torchvision --extra-index-url https://pypi.org/simple

  banner "Installing Python dependencies"
  "$python_bin" -m pip install -r "$PROJECT_ROOT/requirements.txt"
}

create_conda_env() {
  local conda_bin="$1"
  banner "Creating conda env: $ENV_NAME (Python $PY_VERSION)"
  "$conda_bin" create -y -n "$ENV_NAME" python="$PY_VERSION"
  source "$("$conda_bin" info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  install_with_pip "$(which python)"
  banner "Environment '$ENV_NAME' ready (conda)"
}

create_venv_env() {
  banner "Creating venv: $ENV_NAME (Python $PY_VERSION if available)"
  # Prefer system python matching PY_VERSION if present
  if has_cmd "python$PY_VERSION"; then
    "python$PY_VERSION" -m venv "$PROJECT_ROOT/.venv-$ENV_NAME"
  else
    python3 -m venv "$PROJECT_ROOT/.venv-$ENV_NAME"
  fi
  source "$PROJECT_ROOT/.venv-$ENV_NAME/bin/activate"
  install_with_pip "$(which python)"
  banner "Environment '$ENV_NAME' ready (venv)"
}

main() {
  banner "PathPrism installer"
  echo "Project root: $PROJECT_ROOT"

  if has_cmd conda; then
    create_conda_env "$(command -v conda)"
    echo "To activate: conda activate $ENV_NAME"
  else
    create_venv_env
    echo "To activate: source $PROJECT_ROOT/.venv-$ENV_NAME/bin/activate"
  fi

  cat <<"EOF"

Next steps:
- Activate the environment (see above)
- Verify torch: python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
- Explore README for pipeline usage.

Tips:
- If CUDA version differs, you can reinstall specific builds, e.g.:
  pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu121 torch torchvision

EOF
}

main "$@"
