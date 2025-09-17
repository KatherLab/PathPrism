import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


##################################
#His-seg Heatmap Loader(MacroNet)
##################################
class SegHeatmapDatasetLoader(Dataset):
    def __init__(self, seg_filepaths, transform):
        super(SegHeatmapDatasetLoader, self).__init__()
        self.seg_filepaths = seg_filepaths
        self.transform = transform
        DACHS_info_dir = '/mnt/bulk-saturn/junhao/pathfinder/CRC/info/DACHS_merged.csv'
        self.DACHS_data = pd.read_csv(DACHS_info_dir)
        MCO_info_dir = '/mnt/bulk-saturn/junhao/pathfinder/CRC/info/MCO_merged.csv'
        self.MCO_data = pd.read_csv(MCO_info_dir)
        TCGA_info_dir = '/mnt/bulk-saturn/junhao/pathfinder/CRC/info/TCGA_outcome_new.csv'
        self.TCGA_data = pd.read_csv(TCGA_info_dir)
        CR07_info_dir = '/mnt/bulk-saturn/junhao/pathfinder/CRC/info/CR07_clini_0106.csv'
        self.CR07_data = pd.read_csv(CR07_info_dir)

        # Replace #N/A with NaN new
        self.TCGA_data[['DSS', 'DSS.time']] = self.TCGA_data[['DSS', 'DSS.time']].replace(['#N/A', 'NaN', 'nan', pd.NaT], np.nan)
        # Drop rows where DSS_cr is NaN
        self.TCGA_data.dropna(subset=['DSS'], inplace=True)

    def __len__(self):
        return len(self.seg_filepaths)

    def __getitem__(self, idx):
        seg_filepath = self.seg_filepaths[idx]
        seg = np.load(seg_filepath)

        if self.transform is not None:
            seg = self.transform(image=seg)["image"]

        cohort = seg_filepath.split('/')[-3]  #The training set may from different cohort

        if cohort == 'TCGA_CRC':
            ID = seg_filepath.split('/')[-1][:12]
            pd_index = self.TCGA_data[self.TCGA_data['bcr_patient_barcode'].isin([ID])].index.values[0]
            T = self.TCGA_data['DSS.time'][pd_index]
            O = self.TCGA_data['DSS'][pd_index]

        elif cohort == 'DACHS': 
            ID = seg_filepath.split('/')[-1][:9]
            pd_index = self.DACHS_data[self.DACHS_data['PATIENT'].isin([ID])].index.values[0]
            T = self.DACHS_data['DSS'][pd_index]
            O = self.DACHS_data['DSS_E'][pd_index]
        elif cohort == 'MCO': 
            ID = seg_filepath.split('/')[-1][:7]
            pd_index = self.MCO_data[self.MCO_data['FILENAME'].isin([ID])].index.values[0]
            T = self.MCO_data['DSS'][pd_index] * 30
            O = self.MCO_data['DSS_E'][pd_index]
        elif cohort == 'CR07': 
            ID = seg_filepath.split('/')[-1][:-4]
            pd_index = self.CR07_data[self.CR07_data['FILENAME'].astype(str).isin([ID])].index.values[0]
            T = self.CR07_data['DSS'][pd_index]
            O = self.CR07_data['DSS_E'][pd_index]
        else:
            print('No {} cohort'.format(cohort))

        O = torch.tensor(O).type(torch.FloatTensor)
        T = torch.tensor(T).type(torch.FloatTensor)

        return seg, T, O, seg_filepath

