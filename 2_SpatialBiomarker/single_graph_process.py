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

# === Single Tissue Graph Construction ===
def build_single_tissue_graph(label_map, class_id, dist_thresh=30, erosion_radius=2, area_thresh=3000, block_size=64):
    """Build a region graph for a single tissue type."""
    # Get class name
    class_names = {v: k for k, v in class_idx.items()}
    class_name = class_names[class_id]
    
    mask = (label_map == class_id).astype(np.uint8)

    # Check if there are enough pixels for the class
    if np.sum(mask) < 5:
        return nx.Graph(), 0

    # Apply erosion to reduce noise
    if erosion_radius > 0:
        selem = disk(erosion_radius)
        mask = erosion(mask, selem)

    # Label connected components
    label_img = label(mask)
    props = regionprops(label_img)

    # Process regions
    nodes = []
    for i, region in enumerate(props):
        if region.area > area_thresh:
            # Split large region
            subs = split_large_region(region, mask, block_size)
            for k, (centroid, bbox) in enumerate(subs):
                nodes.append((f"{class_name}_{i}_{k}", centroid, bbox, region.area))
        else:
            cy, cx = region.centroid
            nodes.append((f"{class_name}_{i}", (cx, cy), region.bbox, region.area))

    # Extract coordinates for KDTree
    coords = [node[1][::-1] for node in nodes]  # Note: reverse to (y, x) format

    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for nid, pos, bbox, area in nodes:
        G.add_node(nid, pos=pos, bbox=bbox, type=class_name, area=area)

    # Build KDTree for fast neighbor search
    if len(coords) > 1:
        tree = KDTree(coords)
        # Add edges
        for i, (nid, pos, _, _) in enumerate(nodes):
            indices = tree.query_ball_point(pos[::-1], r=dist_thresh)  # Note the reverse here
            for j in indices:
                if i == j: continue
                nid2, pos2, _, _ = nodes[j]
                if not G.has_edge(nid, nid2):
                    dist = np.linalg.norm(np.array(pos) - np.array(pos2))
                    G.add_edge(nid, nid2, distance=dist)

    return G, len(nodes)

# === Single Tissue Feature Extraction ===
def extract_single_tissue_features(G, class_id, n_nodes):
    """Extract features from a graph of a single tissue."""
    features = {}
    
    # Get class name
    class_names = {v: k for k, v in class_idx.items()}
    class_name = class_names[class_id]
    
    # Return default features if graph is empty
    if G.number_of_nodes() == 0:
        prefix = f'single_{class_name}'
        default_features = {
            f'{prefix}_n_nodes': 0,
            f'{prefix}_n_edges': 0,
            f'{prefix}_edge_density': 0,
            f'{prefix}_mean_dist': 0,
            f'{prefix}_std_dist': 0,
            f'{prefix}_avg_degree': 0,
            f'{prefix}_max_degree': 0,
            f'{prefix}_min_degree': 0,
            f'{prefix}_degree_cv': 0,
            f'{prefix}_entropy_degree': 0,
            f'{prefix}_connected_components': 0,
            f'{prefix}_largest_component_ratio': 0,
            f'{prefix}_avg_clustering': 0,
            f'{prefix}_avg_path_length': 0,
            f'{prefix}_diameter': 0,
            f'{prefix}_avg_area': 0,
            f'{prefix}_std_area': 0,
            f'{prefix}_area_cv': 0,
            f'{prefix}_spatial_dispersion': 0,
            f'{prefix}_nearest_neighbor_ratio': 0,
            f'{prefix}_avg_betweenness': 0,
            f'{prefix}_avg_closeness': 0,
            f'{prefix}_modularity': 0,
            f'{prefix}_assortativity': 0
        }
        return default_features
    
    # Prefix uses class name
    prefix = f'single_{class_name}'
    
    # Basic features
    features[f'{prefix}_n_nodes'] = n_nodes
    features[f'{prefix}_n_edges'] = G.number_of_edges()
    
    # Edge density = actual edges / max possible edges
    max_possible_edges = n_nodes * (n_nodes - 1) / 2
    features[f'{prefix}_edge_density'] = G.number_of_edges() / (max_possible_edges + 1e-6)
    
    # Edge distance features
    edge_dists = [d['distance'] for _, _, d in G.edges(data=True)]
    if edge_dists:
        features[f'{prefix}_mean_dist'] = float(np.mean(edge_dists))
        features[f'{prefix}_std_dist'] = float(np.std(edge_dists))
    else:
        features[f'{prefix}_mean_dist'] = 0.0
        features[f'{prefix}_std_dist'] = 0.0
    
    # Degree features
    degrees = [d for _, d in G.degree()]
    if degrees:
        features[f'{prefix}_avg_degree'] = float(np.mean(degrees))
        features[f'{prefix}_max_degree'] = float(np.max(degrees))
        features[f'{prefix}_min_degree'] = float(np.min(degrees))
        features[f'{prefix}_degree_cv'] = float(np.std(degrees) / (np.mean(degrees) + 1e-6))
        features[f'{prefix}_entropy_degree'] = float(entropy(degrees))
    else:
        features[f'{prefix}_avg_degree'] = 0.0
        features[f'{prefix}_max_degree'] = 0.0
        features[f'{prefix}_min_degree'] = 0.0
        features[f'{prefix}_degree_cv'] = 0.0
        features[f'{prefix}_entropy_degree'] = 0.0
    
    # Connectivity features
    features[f'{prefix}_connected_components'] = nx.number_connected_components(G)
    
    # Largest connected component ratio
    largest_cc = max(nx.connected_components(G), key=len)
    features[f'{prefix}_largest_component_ratio'] = len(largest_cc) / (n_nodes + 1e-6)
    
    # Clustering coefficient
    try:
        clustering = nx.clustering(G)
        features[f'{prefix}_avg_clustering'] = float(np.mean(list(clustering.values())))
    except:
        features[f'{prefix}_avg_clustering'] = 0.0
    
    # Average path length and diameter (on the largest connected component only)
    if len(largest_cc) > 1:
        largest_cc_subgraph = G.subgraph(largest_cc)
        try:
            features[f'{prefix}_avg_path_length'] = float(nx.average_shortest_path_length(largest_cc_subgraph))
        except:
            features[f'{prefix}_avg_path_length'] = 0.0
        
        try:
            features[f'{prefix}_diameter'] = float(nx.diameter(largest_cc_subgraph))
        except:
            features[f'{prefix}_diameter'] = 0.0
    else:
        features[f'{prefix}_avg_path_length'] = 0.0
        features[f'{prefix}_diameter'] = 0.0
    
    # Region area features
    areas = [data['area'] for _, data in G.nodes(data=True)]
    if areas:
        features[f'{prefix}_avg_area'] = float(np.mean(areas))
        features[f'{prefix}_std_area'] = float(np.std(areas))
        features[f'{prefix}_area_cv'] = float(np.std(areas) / (np.mean(areas) + 1e-6))
    else:
        features[f'{prefix}_avg_area'] = 0.0
        features[f'{prefix}_std_area'] = 0.0
        features[f'{prefix}_area_cv'] = 0.0
    
    # Spatial dispersion (std of node coordinates)
    positions = np.array([data['pos'] for _, data in G.nodes(data=True)])
    if len(positions) > 0:
        features[f'{prefix}_spatial_dispersion'] = float(np.mean(np.std(positions, axis=0)))
    else:
        features[f'{prefix}_spatial_dispersion'] = 0.0
    
    # Nearest neighbor ratio (actual mean NN distance vs random expectation)
    if len(positions) > 1:
        # Compute actual mean nearest neighbor distance
        tree = KDTree(positions)
        distances, _ = tree.query(positions, k=2)  # k=2 because the first is itself
        actual_mean_nn_dist = np.mean(distances[:, 1])
        
        # Expected NN distance under CSR: 0.5 * sqrt(A/n), A area, n points
        # Use bounding box area as an approximation here
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        area = (max_x - min_x) * (max_y - min_y)
        expected_mean_nn_dist = 0.5 * np.sqrt(area / len(positions))
        
        features[f'{prefix}_nearest_neighbor_ratio'] = float(actual_mean_nn_dist / (expected_mean_nn_dist + 1e-6))
    else:
        features[f'{prefix}_nearest_neighbor_ratio'] = 0.0
    
    # Betweenness and closeness centrality
    if len(largest_cc) > 1:
        largest_cc_subgraph = G.subgraph(largest_cc)
        try:
            betweenness = nx.betweenness_centrality(largest_cc_subgraph)
            features[f'{prefix}_avg_betweenness'] = float(np.mean(list(betweenness.values())))
        except:
            features[f'{prefix}_avg_betweenness'] = 0.0
        
        try:
            closeness = nx.closeness_centrality(largest_cc_subgraph)
            features[f'{prefix}_avg_closeness'] = float(np.mean(list(closeness.values())))
        except:
            features[f'{prefix}_avg_closeness'] = 0.0
    else:
        features[f'{prefix}_avg_betweenness'] = 0.0
        features[f'{prefix}_avg_closeness'] = 0.0
    
    # Community structure
    try:
        communities = list(nx.algorithms.community.louvain_communities(G, seed=0))
        features[f'{prefix}_modularity'] = nx.algorithms.community.quality.modularity(G, communities)
    except:
        features[f'{prefix}_modularity'] = 0.0
    
    # Assortativity (degree assortativity)
    try:
        features[f'{prefix}_assortativity'] = nx.degree_assortativity_coefficient(G)
    except:
        features[f'{prefix}_assortativity'] = 0.0
    
    return features

# === Visualization for Single Tissue Graph ===
def save_single_tissue_visualization(label_map, G, class_id, save_path):
    """Save visualization for single tissue graph."""
    # Get class name
    class_names = {v: k for k, v in class_idx.items()}
    class_name = class_names[class_id]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(label_map, cmap='tab10')
    ax = plt.gca()
    
    # Draw nodes
    for n, data in G.nodes(data=True):
        x, y = data['pos']
        ax.plot(x, y, 'o', markersize=5, color=color_map[class_id])
    
    # Draw edges (up to 1000 for performance)
    edge_sample = list(G.edges())[:1000]
    for u, v in edge_sample:
        x1, y1 = G.nodes[u]['pos']
        x2, y2 = G.nodes[v]['pos']
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=0.5, alpha=0.5)
    
    plt.title(f"Single Tissue Graph: {class_name}\n#Nodes: {G.number_of_nodes()}, #Edges: {G.number_of_edges()}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# === Process Single Sample for One Tissue ===
def process_sample_single_tissue(path, class_id, params, cohort_dir, save_images=False):
    """Process spatial relations of a single tissue for one sample."""
    try:
        # Get class name
        class_names = {v: k for k, v in class_idx.items()}
        class_name = class_names[class_id]
        
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
        
        # Create save dir for this tissue
        tissue_dir = os.path.join(cohort_dir, f"single_{class_name}")
        os.makedirs(tissue_dir, exist_ok=True)
        
        # Skip if already processed
        result_file = os.path.join(tissue_dir, f"{patient_name}.json")
        if os.path.exists(result_file):
            print(f"Exists: {result_file}, skip")
            with open(result_file, 'r') as f:
                return json.load(f)
        
        # Compute label map
        label_map = np.argmax(prob_map, axis=-1)
        
        # Adjust ADI vs BACK classification
        if class_id == class_idx['ADI']:
            adi_prob = prob_map[:, :, class_idx['ADI']]
            adi_threshold = 0.5
            adi_mask = (label_map == class_idx['ADI']) & (adi_prob < adi_threshold)
            label_map[adi_mask] = class_idx['BACK']
        
        # Check pixel count for the class
        class_pixels = np.sum(label_map == class_id)
        
        if class_pixels < 5:
            print(f"Insufficient pixels for class {class_name} in sample {patient_name}")
            features = {
                'PATIENT': patient_name,
                'Path': path,
                'class': class_name,
                'class_pixels': int(class_pixels),
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
            'class': class_name,
            'class_pixels': int(class_pixels),
            'status': 'processed'
        }
        
        # Create visualization directory
        if save_images:
            vis_dir = os.path.join(tissue_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
        
        # Create graph directory
        graph_dir = os.path.join(tissue_dir, "graphs")
        os.makedirs(graph_dir, exist_ok=True)
        
        # Build single tissue graph
        dist_thresh = params.get('dist_thresh', 30)
        erosion_radius = params.get('erosion_radius', 2)
        area_thresh = params.get('area_thresh', 3000)
        block_size = params.get('block_size', 64)
        
        G, n_nodes = build_single_tissue_graph(
            label_map, class_id, 
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
        features = extract_single_tissue_features(G, class_id, n_nodes)
        all_features.update(features)
        
        # Save visualization (optional)
        if save_images:
            save_path = os.path.join(vis_dir, f"{patient_name}.png")
            save_single_tissue_visualization(label_map, G, class_id, save_path)
        
        # Save graph object
        if G.number_of_nodes() > 10:
            graph_data = {
                'graph': G,
                'class_id': class_id,
                'class_name': class_name,
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
        # Get class name
        class_names = {v: k for k, v in class_idx.items()}
        class_name = class_names[class_id]
        
        error_features = {
            'PATIENT': os.path.splitext(os.path.basename(path))[0],
            'Path': path,
            'class': class_name,
            'Error': str(e),
            'status': 'error'
        }
        
        # Save error info
        error_dir = os.path.join(cohort_dir, f"single_{class_name}")
        os.makedirs(error_dir, exist_ok=True)
        error_file = os.path.join(error_dir, f"{os.path.splitext(os.path.basename(path))[0]}_error.json")
        with open(error_file, 'w') as f:
            json.dump(error_features, f)
        
        return error_features

# === Batch Processing for Single Tissue ===
def batch_process_single_tissue_features(pkl_path, save_dir, params=None, 
                                         tissue_list=None, max_workers=16, 
                                         save_images=False, chunk_size=50,
                                         visualize_first=True):
    """Batch process spatial features for single tissues."""
    if params is None:
        params = {
            'dist_thresh': 30,
            'erosion_radius': 2,
            'area_thresh': 3000,
            'block_size': 64
        }
    
    dataset_name = os.path.splitext(os.path.basename(pkl_path))[0].replace('pred_test_', '')
    cohort_dir = os.path.join(save_dir, f"single_tissue_graph_{dataset_name}")
    os.makedirs(cohort_dir, exist_ok=True)
    
    # Save processing config
    config = {
        'dataset': dataset_name,
        'params': params,
        'tissue_list': tissue_list,
        'max_workers': max_workers,
        'save_images': save_images,
        'chunk_size': chunk_size,
        'visualize_first': visualize_first,
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
            'processed_tissues': [],
            'current_tissue_index': 0,
            'failed_tasks': []
        }
    
    # Determine tissues to process
    all_tissues = []
    if tissue_list:
        # Use specified tissues
        for name in tissue_list:
            if name in class_idx:
                all_tissues.append((class_idx[name], name))
    else:
        # Use all tissues except BACK
        all_tissues = [(v, k) for k, v in class_idx.items() if k != 'BACK']
    
    # Continue from last processed tissue index
    current_tissue_index = progress['current_tissue_index']
    processed_tissues = set(progress['processed_tissues'])
    
    # Process each tissue type
    for tissue_index, (class_id, class_name) in enumerate(all_tissues[current_tissue_index:], current_tissue_index):
        # Skip if this tissue already processed
        if class_name in processed_tissues:
            print(f"Tissue {class_name} already processed, skip")
            continue
        
        print(f"\n===== Processing tissue {class_name} ({tissue_index+1}/{len(all_tissues)}) =====")
        
        # Update current tissue index
        progress['current_tissue_index'] = tissue_index
        # Ensure JSON-serializable types
        progress = convert_numpy_types(progress)
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Create tissue directory
        tissue_dir = os.path.join(cohort_dir, f"single_{class_name}")
        os.makedirs(tissue_dir, exist_ok=True)
        
        # Get processed samples
        processed_patients = set()
        if os.path.exists(tissue_dir):
            processed_files = [f for f in os.listdir(tissue_dir) if f.endswith('.json') and not f.endswith('_error.json')]
            processed_patients = {os.path.splitext(f)[0] for f in processed_files}
        
        # Filter out processed samples
        paths_to_process = [p for p in paths if os.path.splitext(os.path.basename(p))[0] not in processed_patients]
        print(f"Tissue {class_name}: total: {len(paths)}, processed: {len(processed_patients)}, pending: {len(paths_to_process)}")
        
        if len(paths_to_process) == 0:
            print(f"All samples for tissue {class_name} are processed")
            # Mark this tissue as processed
            if class_name not in processed_tissues:
                progress['processed_tissues'].append(class_name)
                # Ensure JSON-serializable types
                progress = convert_numpy_types(progress)
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)
            continue
        
        # Optionally visualize the first sample
        if visualize_first and len(paths_to_process) > 0:
            first_path = paths_to_process[0]
            print(f"Visualizing first sample for tissue {class_name}: {os.path.basename(first_path)}")
            process_sample_single_tissue(first_path, class_id, params, cohort_dir, save_images=True)
            # If only one sample, mark done and continue
            if len(paths_to_process) == 1:
                # Mark this tissue as processed
                if class_name not in processed_tissues:
                    progress['processed_tissues'].append(class_name)
                    # Ensure JSON-serializable types
                    progress = convert_numpy_types(progress)
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f, indent=2)
                continue
            # Otherwise continue with the rest
            paths_to_process = paths_to_process[1:]
        
        # Process in chunks
        for i in range(0, len(paths_to_process), chunk_size):
            chunk_paths = paths_to_process[i:i+chunk_size]
            print(f"Processing tissue {class_name} batch {i//chunk_size + 1}/{(len(paths_to_process)-1)//chunk_size + 1} ({len(chunk_paths)} samples)")
            
            # Parallel processing with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for path in chunk_paths:
                    # Only process current tissue
                    future = executor.submit(
                        process_sample_single_tissue, 
                        path, class_id, 
                        params, cohort_dir, save_images=False
                    )
                    futures.append((path, future))
                
                # Collect results
                for path, future in tqdm(futures, desc=f"Processing tissue {class_name}"):
                    try:
                        future.result()  # Results are already written to files
                    except Exception as e:
                        print(f"Failed processing tissue {class_name} for sample {path}: {str(e)}")
                        patient_name = os.path.splitext(os.path.basename(path))[0]
                        progress['failed_tasks'].append({
                            'patient': patient_name,
                            'tissue': class_name,
                            'path': path,
                            'error': str(e)
                        })
                        # Update progress file
                        # Ensure JSON-serializable types
                        progress = convert_numpy_types(progress)
                        with open(progress_file, 'w') as f:
                            json.dump(progress, f, indent=2)
        
        # Mark this tissue as processed
        if class_name not in processed_tissues:
            progress['processed_tissues'].append(class_name)
            # Ensure JSON-serializable types
            progress = convert_numpy_types(progress)
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
    
    # Merge results after all tissues processed
    print("\n===== All tissues processed, start merging results =====")
    
    # Create a dict keyed by patient id
    patient_features = defaultdict(dict)
    
    # Collect features of each tissue
    for class_id, class_name in all_tissues:
        tissue_dir = os.path.join(cohort_dir, f"single_{class_name}")
        if not os.path.exists(tissue_dir):
            continue
        
        # Collect all patient features for this tissue
        tissue_files = [f for f in os.listdir(tissue_dir) if f.endswith('.json') and not f.endswith('_error.json')]
        for file in tissue_files:
            file_path = os.path.join(tissue_dir, file)
            with open(file_path, 'r') as f:
                tissue_data = json.load(f)
                patient_id = tissue_data['PATIENT']
                
                # Add this tissue's features into patient's feature dict
                for key, value in tissue_data.items():
                    if key.startswith(f'single_{class_name}'):
                        # Use tissue-specific prefix
                        patient_features[patient_id][key] = value
                    elif key not in ['Path', 'class', 'class_pixels', 'status']:
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
    df_patient.to_csv(os.path.join(cohort_dir, f"SingleTissueGraphFeats_patient_{dataset_name}.csv"), index=False)
    merged_df.to_csv(os.path.join(cohort_dir, f"SingleTissueGraphFeats_merged_{dataset_name}.csv"), index=False)
    
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

    save_dir = "/mnt/bulk-saturn/junhao/pathfinder/CRC/code/graph_biomarker/results_single_region_graph"
    
    # Tissues to process
    tissue_list = [
        'TUM',  # tumor
        'STR',  # stroma
        'LYM',  # lymphocyte
        'MUS',  # muscle
        'NORM', # normal epithelium
        'MUC',  # mucus
        'ADI',  # adipose
        'DEB'   # debris/necrosis
    ]

    # Region graph parameters
    region_params = {
        'dist_thresh': 20,     # max connection distance between regions
        'erosion_radius': 1.5, # erosion radius
        'area_thresh': 2000,   # area threshold for large regions
        'block_size': 30       # block size for splitting large regions
    }

    # Sequentially process each dataset
    for pkl_path in dataset_paths:
        batch_process_single_tissue_features(
            pkl_path=pkl_path,
            save_dir=save_dir,
            params=region_params,
            tissue_list=tissue_list,  # only process specified tissues
            max_workers=32,
            save_images=False,
            chunk_size=50,  # process 50 samples per batch
            visualize_first=True  # visualize the first sample of each queue
        )