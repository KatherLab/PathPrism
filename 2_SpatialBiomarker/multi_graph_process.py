import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from skimage.measure import label, regionprops
from skimage.morphology import erosion, disk
from skimage.util import view_as_blocks
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import time
import json
from numba import njit, prange

# === Utility Functions ===
def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# === Basic Parameters ===
class_idx = {
    'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4,
    'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8
}
color_map = {
    0: "#4F81BD", 1: "#FFFFFF", 2: "#9C6ADE", 3: "#A6A6A6", 4: "#F7A1C4",
    5: "#88C090", 6: "#76CBEF", 7: "#FDB66F", 8: "#E15759"
}

# === Split Large Region ===
def split_large_region(region, global_mask, block_size=64):
    """Split a large region into smaller blocks."""
    minr, minc, maxr, maxc = region.bbox
    region_mask = global_mask[minr:maxr, minc:maxc]
    label_mask = (label(global_mask) == region.label)[minr:maxr, minc:maxc]

    h, w = region_mask.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    padded_mask = np.pad(region_mask, ((0, pad_h), (0, pad_w)), mode='constant')
    padded_label = np.pad(label_mask, ((0, pad_h), (0, pad_w)), mode='constant')

    blocks_mask = view_as_blocks(padded_mask, block_shape=(block_size, block_size))
    blocks_label = view_as_blocks(padded_label, block_shape=(block_size, block_size))

    subregions = []
    for i in range(blocks_mask.shape[0]):
        for j in range(blocks_mask.shape[1]):
            block_label = blocks_label[i, j]
            if np.sum(block_label) / block_label.size > 0.5:
                cy = minr + i * block_size + block_size // 2
                cx = minc + j * block_size + block_size // 2
                subregions.append(((cx, cy), (minr + i * block_size, minc + j * block_size,
                                              minr + (i + 1) * block_size, minc + (j + 1) * block_size)))
    return subregions

# === Entropy ===
def entropy(values):
    """Compute information entropy."""
    if len(values) == 0:
        return 0.0
    probs = np.array(list(Counter(values).values())) / len(values)
    return -np.sum(probs * np.log2(probs + 1e-10))

# === Build Region Graph ===
def build_region_contact_graph(label_map, class_id1, class_id2, dist_thresh=30, erosion_radius=2, area_thresh=3000, block_size=64):
    """Build a region contact graph between two classes."""
    # Get class names
    class_names = {v: k for k, v in class_idx.items()}
    class_name1 = class_names[class_id1]
    class_name2 = class_names[class_id2]
    
    mask1 = (label_map == class_id1).astype(np.uint8)
    mask2 = (label_map == class_id2).astype(np.uint8)

    # Check if both classes have enough pixels
    if np.sum(mask1) < 5 or np.sum(mask2) < 5:
        return nx.Graph(), 0, 0

    # Apply erosion to reduce noise
    if erosion_radius > 0:
        selem = disk(erosion_radius)
        mask1 = erosion(mask1, selem)
        mask2 = erosion(mask2, selem)

    # Label connected components
    label1 = label(mask1)
    label2 = label(mask2)
    props1 = regionprops(label1)
    props2 = regionprops(label2)

    # Process regions for the first class
    nodes1 = []
    for i, region in enumerate(props1):
        if region.area > area_thresh:
            # Split large region
            subs = split_large_region(region, mask1, block_size)
            for k, (centroid, bbox) in enumerate(subs):
                nodes1.append((f"{class_name1}_{i}_{k}", centroid, bbox))
        else:
            cy, cx = region.centroid
            nodes1.append((f"{class_name1}_{i}", (cx, cy), region.bbox))

    # Process regions for the second class
    nodes2 = []
    for j, region in enumerate(props2):
        if region.area > area_thresh:
            # Split large region
            subs = split_large_region(region, mask2, block_size)
            for k, (centroid, bbox) in enumerate(subs):
                nodes2.append((f"{class_name2}_{j}_{k}", centroid, bbox))
        else:
            cy, cx = region.centroid
            nodes2.append((f"{class_name2}_{j}", (cx, cy), region.bbox))

    # Extract coordinates for KDTree
    coords1 = [node[1][::-1] for node in nodes1]  # Note: reverse to (y, x) format
    coords2 = [node[1][::-1] for node in nodes2]

    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for nid, pos, bbox in nodes1:
        G.add_node(nid, pos=pos, bbox=bbox, type=class_name1)
    for nid, pos, bbox in nodes2:
        G.add_node(nid, pos=pos, bbox=bbox, type=class_name2)

    # Build KDTree for fast neighbor search
    if len(coords2) > 0:
        tree2 = KDTree(coords2)
        # Add inter-class edges
        for i, (nid, pos, _) in enumerate(nodes1):
            indices = tree2.query_ball_point(pos[::-1], r=dist_thresh)  # Note the reverse here
            for idx in indices:
                sid, spos, _ = nodes2[idx]
                dist = np.linalg.norm(np.array(pos) - np.array(spos))
                G.add_edge(nid, sid, distance=dist, edge_type='inter')

    # Add intra-class edges for class 1
    if len(coords1) > 1:
        tree1 = KDTree(coords1)
        for i, (nid1, pos1, _) in enumerate(nodes1):
            indices = tree1.query_ball_point(pos1[::-1], r=dist_thresh)  # Note the reverse here
            for j in indices:
                if i == j: continue
                nid2, pos2, _ = nodes1[j]
                if not G.has_edge(nid1, nid2):
                    dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    G.add_edge(nid1, nid2, distance=dist, edge_type=f'intra_{class_name1}')

    # Add intra-class edges for class 2
    if len(coords2) > 1:
        tree2 = KDTree(coords2) if 'tree2' not in locals() else tree2
        for i, (nid1, pos1, _) in enumerate(nodes2):
            indices = tree2.query_ball_point(pos1[::-1], r=dist_thresh)  # Note the reverse here
            for j in indices:
                if i == j: continue
                nid2, pos2, _ = nodes2[j]
                if not G.has_edge(nid1, nid2):
                    dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    G.add_edge(nid1, nid2, distance=dist, edge_type=f'intra_{class_name2}')

    return G, len(nodes1), len(nodes2)

# === Feature Extraction ===
def extract_graph_features(G, class_id1, class_id2, n_nodes1, n_nodes2):
    """Extract features from the region graph."""
    features = {}
    
    # Get class names
    class_names = {v: k for k, v in class_idx.items()}
    class_name1 = class_names[class_id1]
    class_name2 = class_names[class_id2]
    
    # Return default features if graph is empty
    if G.number_of_nodes() == 0:
        prefix = f'region_{class_name1}_{class_name2}'
        default_features = {
            f'{prefix}_n_nodes1': 0,
            f'{prefix}_n_nodes2': 0,
            f'{prefix}_n_inter_edges': 0,
            f'{prefix}_inter_edge_density': 0,
            f'{prefix}_mean_inter_dist': 0,
            f'{prefix}_std_inter_dist': 0,
            f'{prefix}_isolated_ratio_{class_name1}': 1.0,
            f'{prefix}_isolated_ratio_{class_name2}': 1.0,
            f'{prefix}_avg_degree_{class_name1}': 0,
            f'{prefix}_avg_degree_{class_name2}': 0,
            f'{prefix}_connected_components': 0,
            f'{prefix}_modularity': 0,
            f'{prefix}_assortativity': 0,
            f'{prefix}_entropy_degree_{class_name1}': 0,
            f'{prefix}_entropy_degree_{class_name2}': 0,
            f'{prefix}_avg_{class_name2}_per_{class_name1}': 0,
            f'{prefix}_avg_{class_name1}_per_{class_name2}': 0,
            f'{prefix}_cv_degree_{class_name1}': 0,
            f'{prefix}_cv_degree_{class_name2}': 0,
            f'{prefix}_betweenness_mean_{class_name1}': 0,
            f'{prefix}_betweenness_mean_{class_name2}': 0
        }
        return default_features
    
    # Get nodes by type
    nodes1 = [n for n, d in G.nodes(data=True) if d['type'] == class_name1]
    nodes2 = [n for n, d in G.nodes(data=True) if d['type'] == class_name2]
    
    # Get inter-class edges
    edges_inter = [(u, v, d) for u, v, d in G.edges(data=True) if d['edge_type'] == 'inter']
    edge_dists = [d['distance'] for _, _, d in edges_inter]
    
    # Compute node degree statistics
    def degree_stats(node_list, neighbor_type):
        if not node_list:
            return 0.0, 0.0, 0.0
        
        degrees = []
        for n in node_list:
            count = 0
            for nbr in G.neighbors(n):
                if G.nodes[nbr]['type'] == neighbor_type:
                    count += 1
            degrees.append(count)
        
        if not degrees:
            return 0.0, 0.0, 0.0
        
        mean_deg = np.mean(degrees)
        std_deg = np.std(degrees)
        cv_deg = std_deg / (mean_deg + 1e-6)
        return mean_deg, std_deg, cv_deg
    
    # Use class names in prefix instead of IDs
    prefix = f'region_{class_name1}_{class_name2}'
    
    # Basic features
    features[f'{prefix}_n_nodes1'] = n_nodes1
    features[f'{prefix}_n_nodes2'] = n_nodes2
    features[f'{prefix}_n_inter_edges'] = len(edge_dists)
    
    # Edge distance features
    if edge_dists:
        features[f'{prefix}_mean_inter_dist'] = float(np.mean(edge_dists))
        features[f'{prefix}_std_inter_dist'] = float(np.std(edge_dists))
    else:
        features[f'{prefix}_mean_inter_dist'] = 0.0
        features[f'{prefix}_std_inter_dist'] = 0.0
    
    # Degree features
    avg_2_per_1, _, cv_1 = degree_stats(nodes1, class_name2)
    avg_1_per_2, _, cv_2 = degree_stats(nodes2, class_name1)
    
    features[f'{prefix}_avg_{class_name2}_per_{class_name1}'] = avg_2_per_1
    features[f'{prefix}_avg_{class_name1}_per_{class_name2}'] = avg_1_per_2
    features[f'{prefix}_cv_degree_{class_name1}'] = cv_1
    features[f'{prefix}_cv_degree_{class_name2}'] = cv_2
    
    # Isolated node ratios
    isolated1_count = sum(1 for n in nodes1 if all(G.nodes[nbr]['type'] != class_name2 for nbr in G.neighbors(n)))
    features[f'{prefix}_isolated_ratio_{class_name1}'] = isolated1_count / (len(nodes1) + 1e-6)
    
    isolated2_count = sum(1 for n in nodes2 if all(G.nodes[nbr]['type'] != class_name1 for nbr in G.neighbors(n)))
    features[f'{prefix}_isolated_ratio_{class_name2}'] = isolated2_count / (len(nodes2) + 1e-6)
    
    # Edge density
    features[f'{prefix}_inter_edge_density'] = len(edge_dists) / (n_nodes1 * n_nodes2 + 1e-6)
    
    # Connected components
    features[f'{prefix}_connected_components'] = nx.number_connected_components(G)
    
    # Average degree
    features[f'{prefix}_avg_degree_{class_name1}'] = np.mean([G.degree(n) for n in nodes1]) if nodes1 else 0.0
    features[f'{prefix}_avg_degree_{class_name2}'] = np.mean([G.degree(n) for n in nodes2]) if nodes2 else 0.0
    
    # Degree entropy
    features[f'{prefix}_entropy_degree_{class_name1}'] = entropy([G.degree(n) for n in nodes1]) if nodes1 else 0.0
    features[f'{prefix}_entropy_degree_{class_name2}'] = entropy([G.degree(n) for n in nodes2]) if nodes2 else 0.0
    
    # Community structure
    try:
        communities = list(nx.algorithms.community.louvain_communities(G, seed=0))
        features[f'{prefix}_modularity'] = nx.algorithms.community.quality.modularity(G, communities)
    except Exception:
        features[f'{prefix}_modularity'] = 0.0
    
    # Assortativity
    try:
        features[f'{prefix}_assortativity'] = nx.attribute_assortativity_coefficient(G, 'type')
    except Exception:
        features[f'{prefix}_assortativity'] = 0.0
    
    # Betweenness centrality
    try:
        betweenness = nx.betweenness_centrality(G)
        features[f'{prefix}_betweenness_mean_{class_name1}'] = np.mean([betweenness[n] for n in nodes1]) if nodes1 else 0.0
        features[f'{prefix}_betweenness_mean_{class_name2}'] = np.mean([betweenness[n] for n in nodes2]) if nodes2 else 0.0
    except Exception:
        features[f'{prefix}_betweenness_mean_{class_name1}'] = 0.0
        features[f'{prefix}_betweenness_mean_{class_name2}'] = 0.0
    
    return features

# === Visualization ===
def save_region_graph_visualization(label_map, G, class_id1, class_id2, save_path):
    """Save a visualization of the region graph."""
    # Get class names
    class_names = {v: k for k, v in class_idx.items()}
    class_name1 = class_names[class_id1]
    class_name2 = class_names[class_id2]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(label_map, cmap='tab10')
    ax = plt.gca()
    
    # Draw nodes
    for n, data in G.nodes(data=True):
        x, y = data['pos']
        if data['type'] == class_name1:
            ax.plot(x, y, 'o', markersize=5, color=color_map[class_id1])
        else:
            ax.plot(x, y, 'o', markersize=5, color=color_map[class_id2])
    
    # Draw edges (up to 1000 for performance)
    edge_sample = list(G.edges())[:1000]
    for u, v in edge_sample:
        x1, y1 = G.nodes[u]['pos']
        x2, y2 = G.nodes[v]['pos']
        edge_type = G[u][v]['edge_type']
        if edge_type == 'inter':
            ax.plot([x1, x2], [y1, y2], color='green', linewidth=0.5)
        elif edge_type == f'intra_{class_name1}':
            ax.plot([x1, x2], [y1, y2], color='blue', linewidth=0.3, alpha=0.5)
        else:
            ax.plot([x1, x2], [y1, y2], color='red', linewidth=0.3, alpha=0.5)
    
    plt.title(f"Region Graph: {class_name1} - {class_name2}\n#Nodes: {G.number_of_nodes()}, #Edges: {G.number_of_edges()}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# === Process Single Sample ===
def process_sample_region_graph(path, class_id1, class_id2, params, cohort_dir, save_images=False):
    """Process region graph features for a single sample."""
    try:
        # Get class names
        class_names = {v: k for k, v in class_idx.items()}
        class_name1 = class_names[class_id1]
        class_name2 = class_names[class_id2]
        
        patient_name = os.path.splitext(os.path.basename(path))[0]
        
        # Load probability map
        prob_map_path = path.replace('.svs', '.npy').replace('.ndpi', '.npy')
        if not os.path.exists(prob_map_path):
            # Try to find in the same folder by basename
            base_name = os.path.basename(path)
            base_name = os.path.splitext(base_name)[0]
            prob_map_path = os.path.join(os.path.dirname(path), f"{base_name}.npy")
            if not os.path.exists(prob_map_path):
                raise FileNotFoundError(f"Probability map file not found: {prob_map_path}")
        
        # Load probability map
        prob_map = np.load(prob_map_path)
        
        # Create output dir for this tissue pair
        pair_dir = os.path.join(cohort_dir, f"{class_name1}_{class_name2}")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Skip if already processed
        result_file = os.path.join(pair_dir, f"{patient_name}.json")
        if os.path.exists(result_file):
            print(f"Exists: {result_file}, skip")
            with open(result_file, 'r') as f:
                return json.load(f)
        
        # Compute label map
        label_map = np.argmax(prob_map, axis=-1)
        
        # Adjust ADI vs BACK classification
        if class_id1 == class_idx['ADI'] or class_id2 == class_idx['ADI']:
            adi_prob = prob_map[:, :, class_idx['ADI']]
            adi_threshold = 0.5
            adi_mask = (label_map == class_idx['ADI']) & (adi_prob < adi_threshold)
            label_map[adi_mask] = class_idx['BACK']
        
        # Check pixel counts for both classes
        class1_pixels = np.sum(label_map == class_id1)
        class2_pixels = np.sum(label_map == class_id2)
        
        if class1_pixels < 5 or class2_pixels < 5:
            print(f"Insufficient pixels for {class_name1} or {class_name2} in sample {patient_name}")
            features = {
                'PATIENT': patient_name,
                'Path': path,
                'class1': class_name1,
                'class2': class_name2,
                'class1_pixels': int(class1_pixels),
                'class2_pixels': int(class2_pixels),
                'status': 'insufficient_pixels'
            }
            # Ensure JSON-serializable types
            features = convert_numpy_types(features)
            with open(result_file, 'w') as f:
                json.dump(features, f)
            return features
        
        # Base attributes
        all_features = {
            'PATIENT': patient_name,
            'Path': path,
            'class1': class_name1,
            'class2': class_name2,
            'class1_pixels': int(class1_pixels),
            'class2_pixels': int(class2_pixels),
            'status': 'processed'
        }
        
        # Create visualization directory
        if save_images:
            vis_dir = os.path.join(pair_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
        
        # Create graph directory
        graph_dir = os.path.join(pair_dir, "graphs")
        os.makedirs(graph_dir, exist_ok=True)
        
        # Build region graph
        dist_thresh = params.get('dist_thresh', 30)
        erosion_radius = params.get('erosion_radius', 2)
        area_thresh = params.get('area_thresh', 3000)
        block_size = params.get('block_size', 64)
        
        G, n_nodes1, n_nodes2 = build_region_contact_graph(
            label_map, class_id1, class_id2, 
            dist_thresh=dist_thresh,
            erosion_radius=erosion_radius,
            area_thresh=area_thresh,
            block_size=block_size
        )
        
        # If not enough nodes, skip
        if G.number_of_nodes() < 5:
            all_features['status'] = 'insufficient_nodes'
            all_features = convert_numpy_types(all_features)
            with open(result_file, 'w') as f:
                json.dump(all_features, f)
            return all_features
        
        # Extract features
        features = extract_graph_features(G, class_id1, class_id2, n_nodes1, n_nodes2)
        all_features.update(features)
        
        # Save visualization (optional)
        if save_images:
            save_path = os.path.join(vis_dir, f"{patient_name}.png")
            save_region_graph_visualization(label_map, G, class_id1, class_id2, save_path)
        
        # Save graph object
        if G.number_of_nodes() > 10:
            graph_data = {
                'graph': G,
                'class_id1': class_id1,
                'class_id2': class_id2,
                'class_name1': class_name1,
                'class_name2': class_name2,
                'params': params
            }
            graph_path = os.path.join(graph_dir, f"{patient_name}.pkl")
            with open(graph_path, 'wb') as f:
                pickle.dump(graph_data, f)
        
        # Ensure JSON-serializable types
        all_features = convert_numpy_types(all_features)
        # Save results
        with open(result_file, 'w') as f:
            json.dump(all_features, f)
        
        return all_features
    
    except Exception as e:
        error_msg = f"Error processing {path}: {str(e)}"
        print(error_msg)
        # Get class names
        class_names = {v: k for k, v in class_idx.items()}
        class_name1 = class_names[class_id1]
        class_name2 = class_names[class_id2]
        
        error_features = {
            'PATIENT': os.path.splitext(os.path.basename(path))[0],
            'Path': path,
            'class1': class_name1,
            'class2': class_name2,
            'Error': str(e),
            'status': 'error'
        }
        
        # Save error info
        error_dir = os.path.join(cohort_dir, f"{class_name1}_{class_name2}")
        os.makedirs(error_dir, exist_ok=True)
        error_file = os.path.join(error_dir, f"{os.path.splitext(os.path.basename(path))[0]}_error.json")
        with open(error_file, 'w') as f:
            json.dump(error_features, f)
        
        return error_features

# === Batch Processing ===
def batch_process_region_graph_features(pkl_path, save_dir, params=None, 
                                        priority_pairs=None, max_workers=16, 
                                        save_images=False, chunk_size=50):
    """Batch process region graph features."""
    if params is None:
        params = {
            'dist_thresh': 30,
            'erosion_radius': 2,
            'area_thresh': 3000,
            'block_size': 64
        }
    
    dataset_name = os.path.splitext(os.path.basename(pkl_path))[0].replace('pred_test_', '')
    cohort_dir = os.path.join(save_dir, f"region_graph_{dataset_name}")
    os.makedirs(cohort_dir, exist_ok=True)
    
    # Save processing config
    config = {
        'dataset': dataset_name,
        'params': params,
        'priority_pairs': priority_pairs,
        'max_workers': max_workers,
        'save_images': save_images,
        'chunk_size': chunk_size,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    # Ensure JSON-serializable types
    config = convert_numpy_types(config)
    with open(os.path.join(cohort_dir, 'process_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Load dataset
    with open(pkl_path, 'rb') as file:
        pred_test = pickle.load(file)

    risk_score = pred_test[0]
    survival_time = pred_test[1] / 365
    event = pred_test[2]
    paths = [p for group in pred_test[5] for p in group]

    prognosis_df = pd.DataFrame({
        'Patient_Path': paths,
        'risk_score': risk_score,
        'survival_time': survival_time,
        'event': event
    })
    prognosis_df['PATIENT'] = prognosis_df['Patient_Path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    
    # Create progress file
    progress_file = os.path.join(cohort_dir, f"progress_{dataset_name}.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {
            'total_patients': len(paths),
            'processed_pairs': [],
            'current_pair_index': 0,
            'failed_tasks': []
        }
    
    # Generate all possible tissue pairs (except BACK)
    all_possible_pairs = []
    if priority_pairs:
        # Use specified priority pairs
        for name1, name2 in priority_pairs:
            if name1 in class_idx and name2 in class_idx:
                id1, id2 = class_idx[name1], class_idx[name2]
                # Ensure id1 <= id2 to keep a consistent order
                if id1 > id2:
                    id1, id2 = id2, id1
                    name1, name2 = name2, name1
                # Skip identical pairs
                if id1 != id2:
                    all_possible_pairs.append((id1, id2, name1, name2))
    else:
        # Generate all possible tissue pairs (except BACK)
        class_items = [(k, v) for k, v in class_idx.items() if k != 'BACK']
        for i in range(len(class_items)):
            for j in range(i+1, len(class_items)):  # start from i+1 to avoid duplicates
                name1, id1 = class_items[i]
                name2, id2 = class_items[j]
                all_possible_pairs.append((id1, id2, name1, name2))
    
    # Continue from last processed pair index
    current_pair_index = progress['current_pair_index']
    processed_pairs = set(progress['processed_pairs'])
    
    # Process each tissue pair
    for pair_index, (class_id1, class_id2, class_name1, class_name2) in enumerate(all_possible_pairs[current_pair_index:], current_pair_index):
        pair_key = f"{class_name1}_{class_name2}"
        
        # Skip if this pair already processed
        if pair_key in processed_pairs:
            print(f"Pair {pair_key} already processed, skip")
            continue
        
        print(f"\n===== Processing pair {pair_key} ({pair_index+1}/{len(all_possible_pairs)}) =====")
        
        # Update current pair index
        progress['current_pair_index'] = pair_index
        # Ensure JSON-serializable types
        progress = convert_numpy_types(progress)
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Create pair directory
        pair_dir = os.path.join(cohort_dir, pair_key)
        os.makedirs(pair_dir, exist_ok=True)
        
        # Get processed samples
        processed_patients = set()
        if os.path.exists(pair_dir):
            processed_files = [f for f in os.listdir(pair_dir) if f.endswith('.json') and not f.endswith('_error.json')]
            processed_patients = {os.path.splitext(f)[0] for f in processed_files}
        
        # Filter out processed samples
        paths_to_process = [p for p in paths if os.path.splitext(os.path.basename(p))[0] not in processed_patients]
        print(f"Pair {pair_key}: total: {len(paths)}, processed: {len(processed_patients)}, pending: {len(paths_to_process)}")
        
        if len(paths_to_process) == 0:
            print(f"All samples for pair {pair_key} are processed")
            # Mark this pair as processed
            if pair_key not in processed_pairs:
                progress['processed_pairs'].append(pair_key)
                # Ensure JSON-serializable types
                progress = convert_numpy_types(progress)
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)
            continue
        
        # Process in chunks
        for i in range(0, len(paths_to_process), chunk_size):
            chunk_paths = paths_to_process[i:i+chunk_size]
            print(f"Processing pair {pair_key} batch {i//chunk_size + 1}/{(len(paths_to_process)-1)//chunk_size + 1} ({len(chunk_paths)} samples)")
            
            # Parallel processing with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for path in chunk_paths:
                    # Only process current pair
                    future = executor.submit(
                        process_sample_region_graph, 
                        path, class_id1, class_id2, 
                        params, cohort_dir, save_images
                    )
                    futures.append((path, future))
                
                # Collect results
                for path, future in tqdm(futures, desc=f"Processing pair {pair_key}"):
                    try:
                        future.result()  # Results are already written to files
                    except Exception as e:
                        print(f"Failed processing pair {pair_key} for sample {path}: {str(e)}")
                        patient_name = os.path.splitext(os.path.basename(path))[0]
                        progress['failed_tasks'].append({
                            'patient': patient_name,
                            'pair': pair_key,
                            'path': path,
                            'error': str(e)
                        })
                        # Update progress file
                        # Ensure JSON-serializable types
                        progress = convert_numpy_types(progress)
                        with open(progress_file, 'w') as f:
                            json.dump(progress, f, indent=2)
        
        # Mark this pair as processed
        if pair_key not in processed_pairs:
            progress['processed_pairs'].append(pair_key)
            # Ensure JSON-serializable types
            progress = convert_numpy_types(progress)
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
    
    # Merge results after all pairs processed
    print("\n===== All pairs processed, start merging results =====")
    
    # Create a dict keyed by patient id
    patient_features = defaultdict(dict)
    
    # Collect features for each pair
    for _, _, class_name1, class_name2 in all_possible_pairs:
        pair_key = f"{class_name1}_{class_name2}"
        pair_dir = os.path.join(cohort_dir, pair_key)
        if not os.path.exists(pair_dir):
            continue
        
        # Collect all patient features for this pair
        pair_files = [f for f in os.listdir(pair_dir) if f.endswith('.json') and not f.endswith('_error.json')]
        for file in pair_files:
            file_path = os.path.join(pair_dir, file)
            with open(file_path, 'r') as f:
                pair_data = json.load(f)
                patient_id = pair_data['PATIENT']
                
                # Add this pair's features into patient's feature dict
                for key, value in pair_data.items():
                    if key.startswith(f'region_{class_name1}_{class_name2}'):
                        # Use pair-specific prefix for features
                        patient_features[patient_id][key] = value
                    elif key not in ['Path', 'class1', 'class2', 'class1_pixels', 'class2_pixels', 'status']:
                        # Save common attributes (only once)
                        if key not in patient_features[patient_id]:
                            patient_features[patient_id][key] = value
    
    # Convert dict to list for DataFrame
    all_patient_features = []
    for patient_id, features in patient_features.items():
        features['PATIENT'] = patient_id
        all_patient_features.append(features)
    
    # Create final DataFrame
    df_patient = pd.DataFrame(all_patient_features)
    
    # Merge prognosis info
    merged_df = pd.merge(df_patient, prognosis_df, on='PATIENT', how='inner')
    
    # Save final results
    df_patient.to_csv(os.path.join(cohort_dir, f"RegionGraphFeats_patient_{dataset_name}.csv"), index=False)
    merged_df.to_csv(os.path.join(cohort_dir, f"RegionGraphFeats_merged_{dataset_name}.csv"), index=False)
    
    # Update config as completed
    config['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    config['status'] = 'completed'
    # Ensure JSON-serializable types
    config = convert_numpy_types(config)
    with open(os.path.join(cohort_dir, 'process_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Done: {dataset_name}")
    return merged_df

if __name__ == "__main__":
    dataset_paths = [
        "/mnt/bulk-saturn/junhao/pathfinder/CRC/hist/CNN_raw_20251011_external_test/pred_test_TCGA_CRC.pkl",
        "/mnt/bulk-saturn/junhao/pathfinder/CRC/hist/CNN_raw_20251011_external_test/pred_test_DACHS.pkl",
        "/mnt/bulk-saturn/junhao/pathfinder/CRC/hist/CNN_raw_20251011_external_test/pred_test_MCO.pkl",
        "/mnt/bulk-saturn/junhao/pathfinder/CRC/hist/CNN_raw_20251011_external_test/pred_test_CR07.pkl"
    ]

    save_dir = "/mnt/bulk-saturn/junhao/pathfinder/CRC/code/graph_biomarker/results_region_graph"
    
    # Define priority tissue pairs
    priority_pairs = [
        ('TUM', 'STR'),  # tumor & stroma
        ('TUM', 'LYM'),  # tumor & lymphocyte
        ('TUM', 'MUS'),  # tumor & muscle
        ('LYM', 'STR'),  # lymphocyte & stroma
        ('LYM', 'MUS'),  # lymphocyte & muscle
        ('STR', 'MUS'),  # stroma & muscle
        ('TUM', 'ADI'),  # tumor & adipose
        ('TUM', 'NORM'), # tumor & normal epithelium
        ('NORM', 'ADI'),  # normal epithelium & adipose
        ('NORM', 'LYM'),  # normal epithelium & lymphocyte
        ('NORM', 'MUS'),  # normal epithelium & muscle
        ('NORM', 'STR'),  # normal epithelium & stroma
        ('NORM', 'MUC'),  # normal epithelium & mucus
        ('TUM', 'MUC'),  # tumor & mucus
        ('MUC', 'STR'),  # mucus & stroma
        ('MUC', 'LYM'),  # mucus & lymphocyte
        ('MUC', 'MUS'),  # mucus & muscle
        ('MUC', 'ADI'),  # mucus & adipose
        ('STR', 'ADI'),  # stroma & adipose
        ('LYM', 'ADI'),  # lymphocyte & adipose
    ]

    # Region graph parameters
    region_params = {
        'dist_thresh': 20,     # max connection distance between regions
        'erosion_radius': 1.5,   # erosion radius
        'area_thresh': 2000,   # area threshold for large regions
        'block_size': 30       # block size for splitting large regions
    }

    # Sequentially process each dataset
    for pkl_path in dataset_paths:
        batch_process_region_graph_features(
            pkl_path=pkl_path,
            save_dir=save_dir,
            params=region_params,
            priority_pairs=priority_pairs,  # only process specified pairs
            max_workers=32,
            save_images=False,
            chunk_size=50  # process 50 samples per batch
        )