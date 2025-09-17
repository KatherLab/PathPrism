import os
import pickle
from itertools import chain
import cv2
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from tqdm import tqdm
from scipy.stats import entropy

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

import os
import numpy as np
import pandas as pd
import pickle

def cal_raw_ratio(seg_filepath):
    prob_matrix = np.load(seg_filepath)
    
    class_idx = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 
                 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
    idx_class = {v: k for k, v in class_idx.items()}
    
    multi_classes = np.ones((prob_matrix.shape[0], prob_matrix.shape[1])) * class_idx['BACK']
    
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            if np.sum(prob_matrix[i, j, :]) == 0:
                multi_classes[i][j] = class_idx['BACK']
            else:
                multi_classes[i][j] = np.argmax(prob_matrix[i, j, :])

    class_pixel_counts = {}
    for class_id, class_name in idx_class.items():
        class_pixels = np.sum(multi_classes == class_id)
        class_pixel_counts[class_name] = class_pixels

    total_pixels = sum(count for name, count in class_pixel_counts.items() if name != 'BACK')

    ratios = {}
    for class_name, count in class_pixel_counts.items():
        if class_name != 'BACK':
            ratios[class_name] = count / total_pixels

    return ratios


def batch_calculate(paths_list, save_csv_path):
    all_data = []

    for seg_path in paths_list:
        try:
            ratios = cal_raw_ratio(seg_path)
            ratios['Patient_Path'] = seg_path
            ratios['PATIENT'] = os.path.splitext(os.path.basename(seg_path))[0]
            all_data.append(ratios)
        except Exception as e:
            print(f"Error processing {seg_path}: {e}")
    
    df = pd.DataFrame(all_data)
    columns_order = ['PATIENT', 'Patient_Path', 'ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    df = df[columns_order]
    df.to_csv(save_csv_path, index=False)
    print(f"Saved batch ratios to {save_csv_path}")
    return df


def process_dataset_npy(npy_dir, save_dir):
    """Directly process all .npy files in a directory to compute tissue fractions."""
    # Use directory name as dataset name
    dataset_name = os.path.basename(npy_dir.rstrip('/'))
    
    # Get all .npy files in the directory
    if not os.path.exists(npy_dir):
        raise FileNotFoundError(f"Directory not found: {npy_dir}")
    
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in directory {npy_dir}")
    
    # Build full file paths
    paths = [os.path.join(npy_dir, f) for f in npy_files]
    
    print(f"Found {len(paths)} .npy files in {npy_dir}")

    # Compute tissue fractions
    ratio_csv_path = os.path.join(save_dir, f"{dataset_name}_tissue_fraction.csv")
    ratio_df = batch_calculate(paths, ratio_csv_path)

    print(f"Successfully processed dataset {dataset_name}, total {len(ratio_df)} samples")
    return ratio_df


# Class index mapping
class_idx = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4,
             'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
idx_class = {v: k for k, v in class_idx.items()}

# Convert probability map to label map
def probability_to_multiclass(prob_matrix):
    H, W = prob_matrix.shape[:2]
    multi_classes = np.ones((H, W)) * class_idx['BACK']
    for i in range(H):
        for j in range(W):
            if np.sum(prob_matrix[i, j, :]) == 0:
                multi_classes[i, j] = class_idx['BACK']
            else:
                multi_classes[i, j] = np.argmax(prob_matrix[i, j, :])
    return multi_classes.astype(int)

# Patch entropy calculation
def calculate_patch_entropy(patch):
    values, counts = np.unique(patch, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs)

# For each image/class compute mean entropy at scale=4
def calculate_entropy_scale4_per_class(seg_map, save_dir=None, img_name='sample'):
    H, W = seg_map.shape
    scale = 4
    entropy_means = {}

    for class_id, class_name in idx_class.items():
        if class_name == 'BACK':
            continue
        class_mask = (seg_map == class_id).astype(int)
        if H < scale or W < scale:
            entropy_means[class_name] = np.nan
            continue

        i_starts = range(0, H - scale + 1, scale)
        j_starts = range(0, W - scale + 1, scale)
        entropy_map = np.zeros((len(i_starts), len(j_starts)))

        for i_idx, i in enumerate(i_starts):
            for j_idx, j in enumerate(j_starts):
                patch = class_mask[i:i+scale, j:j+scale]
                if patch.sum() == 0:
                    continue
                entropy_map[i_idx, j_idx] = calculate_patch_entropy(patch)

        nonzero = entropy_map[entropy_map > 0]
        mean_entropy = np.mean(nonzero) if len(nonzero) > 0 else np.nan
        entropy_means[class_name] = mean_entropy

        if save_dir is not None:
            class_dir = os.path.join(save_dir, class_name, 'scale_4')
            os.makedirs(class_dir, exist_ok=True)
            np.save(os.path.join(class_dir, f'{img_name}_entropy.npy'), entropy_map)
            plt.figure(figsize=(6, 6))
            plt.imshow(entropy_map, cmap='viridis', interpolation='nearest')
            plt.title(f'{class_name} - Entropy Map (Scale 4)')
            plt.colorbar(label='Entropy')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(class_dir, f'{img_name}_entropy.png'))
            plt.close()

    return entropy_means

# Batch process main function for each dataset
def process_entropy_dataset_npy(npy_dir, save_dir_entropy_base):
    """Directly process all .npy files to compute entropy metrics."""
    # Use directory name as dataset name
    dataset_name = os.path.basename(npy_dir.rstrip('/'))
    
    # Get all .npy files in the directory
    if not os.path.exists(npy_dir):
        raise FileNotFoundError(f"Directory not found: {npy_dir}")
    
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in directory {npy_dir}")
    
    # Build full file paths
    paths = [os.path.join(npy_dir, f) for f in npy_files]
    
    print(f"Found {len(paths)} .npy files in {npy_dir}")

    save_dir_entropy_maps = os.path.join(save_dir_entropy_base, dataset_name)
    save_summary_csv = os.path.join(save_dir_entropy_base, f'{dataset_name}.csv')

    # Step 1: compute entropy for each patch
    results = []
    for seg_path in tqdm(paths, desc=f"Processing entropy for {dataset_name}"):
        try:
            prob_matrix = np.load(seg_path)
            seg_map = probability_to_multiclass(prob_matrix)
            img_name = os.path.splitext(os.path.basename(seg_path))[0]
            entropy_means = calculate_entropy_scale4_per_class(
                seg_map, save_dir=save_dir_entropy_maps, img_name=img_name
            )
            row = {'Image_Name': img_name, 'PATIENT': img_name}
            for class_name, mean_entropy in entropy_means.items():
                row[f'{class_name}_Entropy_Scale_4'] = mean_entropy
            results.append(row)
        except Exception as e:
            print(f"Error in {seg_path}: {e}")

    # Step 2: save entropy summary
    entropy_df = pd.DataFrame(results)
    entropy_df.to_csv(save_summary_csv, index=False)
    print(f"Entropy summary saved to {save_summary_csv}")

    print(f"Successfully processed entropy for {dataset_name}, total {len(entropy_df)} samples")
    return entropy_df


if __name__ == "__main__":
    # Specify .npy directory
    npy_dir = "/mnt/bulk-saturn/junhao/pathfinder/CRC/features/JUNHAO_CRC/DACHS/prob_map"
    
    # Output path for tissue fraction computation
    save_dir_fraction = "/mnt/bulk-saturn/junhao/pathfinder/CRC/code/graph_biomarker/DACHS_graph/results_tissue_fraction"
    os.makedirs(save_dir_fraction, exist_ok=True)
    
    # Output path for entropy computation
    save_dir_entropy = "/mnt/bulk-saturn/junhao/pathfinder/CRC/code/graph_biomarker/DACHS_graph/results_entropy"
    os.makedirs(save_dir_entropy, exist_ok=True)
    
    print("Start processing tissue fraction...")
    fraction_df = process_dataset_npy(npy_dir, save_dir_fraction)
    
    print("\nStart processing entropy computation...")
    entropy_df = process_entropy_dataset_npy(npy_dir, save_dir_entropy)
    
    print("\nAll done!")
    print(f"Tissue fraction result: {fraction_df.shape}")
    print(f"Entropy result: {entropy_df.shape}")