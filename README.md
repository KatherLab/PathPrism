## PathPrism: Spatial Biomarker Discovery via Interpretable Semantic Learning in Histopathology


![PathPrism Overview](assets/figure1_new2.png)

### Quick Install (PathPrism environment)
```bash
bash install_pathprism.sh
```
- If `conda` is available, a conda env named `pathprism` will be created; otherwise a Python venv `.venv-pathprism` will be used.
- Activate the environment:
  - conda: `conda activate pathprism`
  - venv: `source ./.venv-pathprism/bin/activate`
- Verify PyTorch:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```


### Environment
- Python 3.9+
- Recommended: CUDA-enabled GPU for training and large-scale inference
- Key dependencies: numpy, pandas, scikit-image, scipy, networkx, matplotlib, tqdm, lifelines, opencv-python, numba





## 1) WSI Semantic Decomposition
Download the pre-extracted CRC100k UNI features from:
`https://drive.google.com/file/d/1iT9XDXOc-HqopCucUetgrNwdr13yLswm/view?usp=sharing`.

After downloading, extract the archive to `PathPrism/0_PrismNet/CRC100k_UNI`.
Then run `0_PrismNet/prismnet_train_test.py` to train PrismNet. The trained model will be saved as `prismnet_linProbe.pt`.



## 2) Generate WSI Probability Maps
First, use STAMP v1 (`https://github.com/KatherLab/STAMP/tree/v1`) to convert WSIs into UNI feature matrices. An example layout is provided in `0_PrismNet/WSI_UNI`.


Next, use the trained PrismNet to batch-process the UNI feature matrices and obtain class-wise probability maps. An example output is shown in `0_PrismNet/WSI_probmap`.


You can use `0_PrismNet/probmap_check.ipynb` to validate and visualize the outputs, and to derive a segmentation map from the probability maps.



## 3) Link Semantics and Prognosis
MacroNet is designed to learn the relationship between spatial tissue semantics (from segmentation) and patient prognosis.


Use `1_MacroNet/train_cv.py` to perform 5-fold cross-validation on the DACHS cohort.


Use `1_MacroNet/external_test.py` to evaluate generalization on external test sets.


## 4) Build Spatial Biomarkers from Segmentation

### 4.1 Tissue Fractions and Entropy
```bash
python 2_SpatialBiomarker/tissue_fraction_entropy_calculation.py
# In __main__, set:
#   npy_dir = "/path/to/prob_map"
#   save_dir_fraction = "/path/to/results_tissue_fraction"
#   save_dir_entropy  = "/path/to/results_entropy"
```
Outputs:
- `{dataset}_tissue_fraction.csv` (per-WSI tissue ratios)
- `{dataset}.csv` with per-class entropy metrics 

### 4.2 Graph-based Spatial Features
Single-tissue graphs and inter-tissue region-contact graphs can be computed from the label maps (argmax of probability maps). Example interfaces typically include:

```bash
python 2_SpatialBiomarker/single_region_graph_process.py
# Set (example):
#   dataset_paths = ["/path/to/pred_test_*.pkl"]   # packed predictions incl. paths and outcomes
#   save_dir = "/path/to/results_single_region_graph"
#   tissue_list = ['TUM','STR','LYM','MUS','NORM','MUC','ADI','DEB']
#   region_params = {'dist_thresh':20,'erosion_radius':1.5,'area_thresh':2000,'block_size':30}

python 2_SpatialBiomarker/multi_region_graph_process.py
# Set (example):
#   dataset_paths = ["/path/to/pred_test_*.pkl"]
#   save_dir = "/path/to/results_region_graph"
#   priority_pairs = [('TUM','STR'), ('TUM','LYM'), ...]  # optional
#   region_params = {'dist_thresh':20,'erosion_radius':1.5,'area_thresh':2000,'block_size':30}
```

Outputs:
- Per tissue (or pair) subfolders with per-WSI JSON features
- Optional PNG visualizations and graph pickles (conditioned on node/edge count)
- Summary CSVs: `SingleTissueGraphFeats_*` / `RegionGraphFeats_*` (with merged prognosis info)




## Citation & Contributions
If this repository helps your research, please consider citing it. Feedback via Issues/PRs is welcome.


