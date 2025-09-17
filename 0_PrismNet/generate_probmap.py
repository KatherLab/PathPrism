"""
Generate probability maps (heatmaps) for whole-slide image (WSI) tiles using
precomputed patch features and a PCA + Logistic Regression linear probe.

Overview
--------
This script walks through a directory of HDF5 feature files, loads per-patch
features and coordinates, applies a PCA projection followed by a trained
logistic regression classifier to obtain class probabilities per patch, and
then rasterizes the probabilities into a centered heatmap (numpy array) that is
saved to disk for downstream visualization or analysis.

"""

import torch
import numpy as np
import h5py
import os
from logistic_regression import LogisticRegression


def all_path(dirname, file_type):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if file_type in apath:
                result.append(apath)
    return result


def generate_probmap(feature_path, model, num_classes, save_dir, patch_size = 224):
    with h5py.File(feature_path, 'r') as file:
        print(list(file.keys()))
        features = torch.tensor(file['feats']).type(torch.FloatTensor)
        coords = file['coords'][:]
        # augmented = file['augmented'][:]
        print(type(features))

    # Predict probabilities and class labels
    probs, predictions = model.predict(features)

    # Calculate heatmap size
    heatmap_coords = coords / patch_size
    height = max(heatmap_coords[:, 0])
    width = max(heatmap_coords[:, 1])

    # Create background channel and fill 1, WSI center-aligned
    matrix_size = int(max(height, width) + 16)
    heatmap_matrix = np.zeros((matrix_size, matrix_size, num_classes))

    # Center-align the heatmap and fill in the probabilities
    offset_x = int((matrix_size - height) // 2)
    offset_y = int((matrix_size - width) // 2)

    for i, coord in enumerate(heatmap_coords):
        x, y = int(coord[0]) + offset_x, int(coord[1]) + offset_y  # Apply offsets
        heatmap_matrix[x, y, :] = probs[i].cpu()

    # Print or save the heatmap matrix for further processing or visualization
    print("Heatmap matrix created with shape:", heatmap_matrix.shape)

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = save_dir + '/' + feature_path.split('/')[-1][:-3]
    np.save(save_path, heatmap_matrix)


class PCALogRegWrapper:
    """Minimal wrapper to apply PCA projection before LR inference."""
    def __init__(self, mu: torch.Tensor, W: torch.Tensor, clf: LogisticRegression):
        self.mu = mu
        self.W = W
        self.clf = clf

    @staticmethod
    def _transform_pca(X: torch.Tensor, mu: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return (X - mu) @ W

    def predict(self, feats: torch.Tensor):
        X_proj = self._transform_pca(feats.float(), self.mu, self.W)
        return self.clf.predict(X_proj)

    def predict_proba(self, feats: torch.Tensor):
        X_proj = self._transform_pca(feats.float(), self.mu, self.W)
        return self.clf.predict_proba(X_proj)


def main():
    # Source of H5 features and destination for probability maps
    feature_source = '/mnt/bulk-saturn/junhao/pathfinder/PathPrism/0_PrismNet/WSI_UNI'
    save_dir = '/mnt/bulk-saturn/junhao/pathfinder/PathPrism/0_PrismNet/WSI_probmap'

    # Load PCA + Logistic Regression from checkpoint
    ckpt_path = '/mnt/bulk-saturn/junhao/pathfinder/PathPrism/0_PrismNet/prismnet_linprobe.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    mu = ckpt['pca_mu']
    W = ckpt['pca_W']
    num_classes = int(ckpt['num_classes'])
    feat_dim = W.shape[1]

    clf = LogisticRegression(C=1.0, max_iter=1, verbose=False, random_state=0)
    clf.load_weights(weight_path=None, feat_dim=feat_dim, num_classes=num_classes)
    clf.logreg.load_state_dict(ckpt['state_dict'])
    model = PCALogRegWrapper(mu=mu, W=W, clf=clf)
    print('PCA + Logistic Regression model loaded!')

    feature_paths = all_path(feature_source, '.h5')
    for feature_path in feature_paths:
        generate_probmap(feature_path, model, num_classes, save_dir, patch_size = 224)


if __name__ == "__main__":
    main()
