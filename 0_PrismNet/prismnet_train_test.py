# -*- coding: utf-8 -*-
"""
PrismNet linear probe runner.

This script performs the following steps:
1) Ensure reproducibility and dtype/device policies
2) Load precomputed features and labels
3) Stratified train/validation split
4) PCA on the training set only (no leakage), project train/val/test
5) Search the optimal `max_iter` with automatic early stopping on validation
6) Retrain on train with the best `max_iter`, evaluate on test
7) Save PCA parameters and trained linear probe for inference

"""

import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from linear_probe import (
    train_linear_probe, test_linear_probe
)


def print_metrics(eval_metrics):
    """Pretty-print numeric metrics from an evaluation dictionary."""
    for k, v in eval_metrics.items():
        if "report" in k:
            continue
        
        print(f"Test {k}: {v:.3f}")

# -----------------------------
# 0) Reproducibility
# -----------------------------
SEED = 10

def set_global_seed(seed: int = SEED) -> None:
    """Set seeds and deterministic flags for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # deterministic matmul (CUDA >= 10.2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# 1) Ensure CPU/dtypes
# -----------------------------
def _to_cpu_f32(x: torch.Tensor) -> torch.Tensor:
    """Detach and move a tensor to CPU with dtype float32."""
    return x.detach().to("cpu", dtype=torch.float32)

def _to_cpu_long(x: torch.Tensor) -> torch.Tensor:
    """Detach and move a tensor to CPU with dtype long (class labels)."""
    return x.detach().to("cpu", dtype=torch.long)

def load_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load train/test features and labels from disk, move to CPU, set dtypes."""
    train_feats = torch.load('/mnt/bulk-saturn/junhao/pathfinder/PathPrism/0_PrismNet/CRC100k_UNI/train_feats.pt')
    train_labels = torch.load('/mnt/bulk-saturn/junhao/pathfinder/PathPrism/0_PrismNet/CRC100k_UNI/train_labels.pt')
    test_feats = torch.load('/mnt/bulk-saturn/junhao/pathfinder/PathPrism/0_PrismNet/CRC100k_UNI/test_feats.pt')
    test_labels = torch.load('/mnt/bulk-saturn/junhao/pathfinder/PathPrism/0_PrismNet/CRC100k_UNI/test_labels.pt')

    X_all = _to_cpu_f32(train_feats)
    y_all = _to_cpu_long(train_labels)
    X_test = _to_cpu_f32(test_feats)
    y_test = _to_cpu_long(test_labels)
    return X_all, y_all, X_test, y_test

# -----------------------------
# 2) Stratified split (indices) to avoid implicit conversions
# -----------------------------
def stratified_split_indices(y_all: torch.Tensor, test_size: float = 0.2, seed: int = SEED):
    """Return train/val index tensors using a stratified split."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (tr_idx, val_idx), = sss.split(np.zeros(len(y_all)), y_all.numpy())
    tr_idx_t = torch.from_numpy(tr_idx).long()
    val_idx_t = torch.from_numpy(val_idx).long()
    return tr_idx_t, val_idx_t

# -----------------------------
# 3) PCA on TRAIN ONLY (no leakage)
#    - uses torch.pca_lowrank when beneficial, else SVD
# -----------------------------
def fit_pca_train_only(X: torch.Tensor, k: int = 64):
    """Fit PCA on training features only and return (mean, projection matrix).

    The projection is orthonormal (no whitening). The effective rank is
    bounded by min(n-1, d).
    """
    # center
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu

    n, d = Xc.shape
    # k cannot exceed rank <= min(n-1, d)
    k_eff = max(1, min(k, d, max(1, n - 1)))

    # choose method
    use_lowrank = min(n, d) > (k_eff + 5)  # heuristic: tall/wide matrices benefit
    if use_lowrank:
        # q: oversampling for randomized PCA
        q = min(k_eff + 10, d, n)
        # torch.pca_lowrank returns U, S, V; we need top-k columns of V
        U, S, V = torch.pca_lowrank(Xc, q=q, center=False)
        W = V[:, :k_eff]  # shape d x k
    else:
        # exact SVD (economy)
        # note: full_matrices=False yields Vt with shape (min(n,d), d)
        _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
        W = Vt[:k_eff].T  # d x k

    return mu, W

def transform_pca(X: torch.Tensor, mu: torch.Tensor, W: torch.Tensor):
    """Apply a trained PCA transform (no whitening)."""
    return (X - mu) @ W  # no whitening; orthonormal basis projection

def pca_fit_transform(X_tr_raw: torch.Tensor, X_val_raw: torch.Tensor, X_test: torch.Tensor, k: int = 32):
    """Fit PCA on train and project train/val/test."""
    mu, W = fit_pca_train_only(X_tr_raw, k=k)
    X_tr = transform_pca(X_tr_raw,  mu, W)
    X_val = transform_pca(X_val_raw, mu, W)
    X_te  = transform_pca(X_test,    mu, W)
    return mu, W, X_tr, X_val, X_te

best_mi, best_score = None, -np.inf

def _pick_score(metrics: dict) -> float:
    """Select a scalar score from a metrics dict (prefer AUROC/AUC)."""
    # Prefer AUROC/AUC; fallback to ACC
    for k in ("val_auroc", "val_auc", "val_acc", "val_accuracy",
              "auroc", "auc", "acc", "accuracy"):
        if k in metrics:
            return float(metrics[k])
    # fallback: first numeric value
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating)):
            return float(v)
    raise RuntimeError("No usable metric in metrics dict.")

def search_best_max_iter(X_tr, y_tr, X_val, y_val, patience: int = 2, max_mi: int = 500) -> int:
    """Iterate max_iter from 1..max_mi and early-stop based on val metric."""
    global best_mi, best_score
    best_mi, best_score = None, -np.inf
    no_improve = 0
    for mi in range(1, max_mi + 1):
        clf = train_linear_probe(
            X_tr, y_tr,
            valid_feats=None, valid_labels=None,
            max_iter=mi,
            combine_trainval=False,
            use_sklearn=False,
            verbose=False,
        )
        val_metrics, _ = test_linear_probe(
            clf, X_val, y_val,
            prefix="val_", verbose=False
        )
        score = _pick_score(val_metrics)
        print(f"[val] max_iter={mi:<4} -> score={score:.6f}")
        if score > best_score:
            best_score, best_mi = score, mi
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"[early-stop] No improvement in {patience} steps. Stopping at max_iter={mi}.")
            break
    assert best_mi is not None, "Failed to select a best max_iter."
    print(f"[search] best max_iter = {best_mi} (val score={best_score:.6f})")
    return best_mi

# -----------------------------
# 5) Retrain on TRAIN ONLY with best iter, evaluate on TEST (no merge)
# -----------------------------
def train_and_evaluate(X_tr, y_tr, X_te, y_test, max_iter: int):
    """Train final classifier on train and evaluate on test."""
    best_clf = train_linear_probe(
        X_tr, y_tr,
        valid_feats=None, valid_labels=None,
        max_iter=max_iter,
        combine_trainval=False,
        use_sklearn=False,
        verbose=True,
    )
    test_metrics, dump = test_linear_probe(
        best_clf, X_te, y_test,
        prefix="lin_", verbose=True
    )
    print_metrics(test_metrics)
    return best_clf, test_metrics, dump

# -----------------------------
# 6) Save model and PCA for inference
# -----------------------------
def save_checkpoint(mu, W, clf, y_all, save_dir: str) -> str:
    """Save PCA parameters and linear probe state_dict for inference."""
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        'pca_mu': mu.cpu(),
        'pca_W': W.cpu(),
        'state_dict': clf.logreg.state_dict(),
        'pca_k': W.shape[1],
        'num_classes': int(torch.unique(y_all).numel()),
    }
    save_path = os.path.join(save_dir, 'prismnet_linprobe.pt')
    torch.save(ckpt, save_path)
    print(f"Saved model checkpoint to: {save_path}")
    return save_path


def main():
    set_global_seed(SEED)

    # Load features/labels
    X_all, y_all, X_test, y_test = load_data()

    # Stratified split indices and slice tensors
    tr_idx_t, val_idx_t = stratified_split_indices(y_all, test_size=0.2, seed=SEED)
    X_tr_raw = X_all.index_select(0, tr_idx_t)
    y_tr     = y_all.index_select(0, tr_idx_t)
    X_val_raw = X_all.index_select(0, val_idx_t)
    y_val     = y_all.index_select(0, val_idx_t)

    # PCA on train only then project
    K = 32  
    mu, W, X_tr, X_val, X_te = pca_fit_transform(X_tr_raw, X_val_raw, X_test, k=K)

    # Early-stop search for best max_iter
    patience = 2
    MAX_MI = 500
    best_mi_local = search_best_max_iter(X_tr, y_tr, X_val, y_val, patience=patience, max_mi=MAX_MI)

    # Retrain with best iter and evaluate
    best_clf, test_metrics, dump = train_and_evaluate(X_tr, y_tr, X_te, y_test, max_iter=best_mi_local)

    # Save checkpoint
    save_dir = '/mnt/bulk-saturn/junhao/pathfinder/PathPrism/0_PrismNet'
    save_checkpoint(mu, W, best_clf, y_all, save_dir)


if __name__ == "__main__":
    main()




