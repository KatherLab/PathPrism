import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from data_loaders import *
from train_test import test
from networks import resnext50_32x4d
import argparse


#clinical info of different cohorts
DACHS_info_dir = '/mnt/bulk-saturn/junhao/pathfinder/CRC/info/DACHS_merged.csv'
DACHS_data = pd.read_csv(DACHS_info_dir)
MCO_info_dir = '/mnt/bulk-saturn/junhao/pathfinder/CRC/info/MCO_merged.csv'
MCO_data = pd.read_csv(MCO_info_dir)
TCGA_info_dir = '/mnt/bulk-saturn/junhao/pathfinder/CRC/info/TCGA_outcome.csv'
TCGA_data = pd.read_csv(TCGA_info_dir)
CR07_info_dir = '/mnt/bulk-saturn/junhao/pathfinder/CRC/info/CR07_clini_0106.csv'
CR07_data = pd.read_csv(CR07_info_dir)

# Replace #N/A with NaN
TCGA_data[['DSS_cr', 'DSS.time.cr']] = TCGA_data[['DSS_cr', 'DSS.time.cr']].replace(['#N/A', 'NaN', 'nan', pd.NaT], np.nan)
# Drop rows where DSS_cr is NaN
TCGA_data.dropna(subset=['DSS_cr', 'DSS.time.cr'], inplace=True)


def path_clean(paths, TCGA_data = TCGA_data, MCO_data = MCO_data, DACHS_data = DACHS_data):
    cohort = paths[0].split('/')[-3]
    current_ID = []
    cleaned_path = []
    if cohort == 'TCGA_CRC':
        delete_ID = []
        #####based on tissue size threshold
        with open("/mnt/bulk-saturn/junhao/pathfinder/CRC/features/JUNHAO_CRC/TCGA_CRC/delete_path_list.pkl", "rb") as file:
            delete_paths = pickle.load(file)
        for delete_path in delete_paths:
            delete_ID.append(delete_path.split('/')[-1][:12])
        for seg_filepath in paths:
            ID = seg_filepath.split('/')[-1][:12]
            if TCGA_data['bcr_patient_barcode'].isin([ID]).any():
                if ID not in current_ID and ID not in delete_ID:
                    current_ID.append(ID)
                    cleaned_path.append(seg_filepath)
    elif cohort == 'DACHS':
        for seg_filepath in paths:
            ID = seg_filepath.split('/')[-1][:9]
            if DACHS_data['PATIENT'].isin([ID]).any():
                if ID not in current_ID:
                    current_ID.append(ID)
                    cleaned_path.append(seg_filepath)
    elif cohort == 'MCO':
        for seg_filepath in paths:
            ID = seg_filepath.split('/')[-1][:7]
            if MCO_data['FILENAME'].isin([ID]).any():
                if ID not in current_ID:
                    current_ID.append(ID)
                    cleaned_path.append(seg_filepath)
    elif cohort == 'CR07':
        delete_ID = []
        #####based on tissue size threshold
        with open("/mnt/bulk-saturn/junhao/pathfinder/CRC/features/JUNHAO_CRC/CR07/delete_path_list.pkl", "rb") as file:
            delete_paths = pickle.load(file)
        for delete_path in delete_paths:
            delete_ID.append(delete_path.split('/')[-1][:-4])
        for seg_filepath in paths:
            ID = seg_filepath.split('/')[-1][:-4]
            if CR07_data['FILENAME'].astype(str).isin([ID]).any():
                patient_index = CR07_data[CR07_data['FILENAME'].astype(str).isin([ID])].index.values[0]
                patient_ID = CR07_data['PATIENT'][patient_index]
                if patient_ID not in current_ID and ID not in delete_ID:
                    current_ID.append(patient_ID)
                    cleaned_path.append(seg_filepath)
    else:
        print('No {} cohort'.format(cohort))
    
    return cleaned_path


def get_args():
		
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--external_test', type=bool, default=True, help='make extrenal tests')
    parser.add_argument('--batch_size', type=int, default=36, help='batch size')
    parser.add_argument('--epoch', type=int, default=50, help='number of training epochs')
    parser.add_argument('--cv_fold', type=int, default=5, help='cross validation fold')
    parser.add_argument('--save_path', type=str, default='/mnt/bulk-saturn/junhao/pathfinder/CRC/hist/CNN_raw_20251011_external_test', help='ckpt save path')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--LAMBDA_COX', type=float, default=1, help='')
    parser.add_argument('--LAMBDA_REG', type=float, default=3e-4, help='')

    args = parser.parse_args() 

    return args


def get_files(path, rule=".npy"):
    all = []
    for fpathe,dirs,fs in os.walk(path): 
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule): 
                all.append(filename)
    return all


def all_path(dirname, file_type):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            # if file_type in apath[-4:]:
            if file_type in apath:
                result.append(apath)
    return result


def main():
    #load args
    args = get_args()
    args_dict = {k: v for k, v in args._get_kwargs()}
    print(args_dict)
    print(args)
    
    #load data path
    DACHS_path = '/mnt/bulk-saturn/junhao/pathfinder/CRC/features/JUNHAO_CRC/DACHS/prob_map'
    MCO_path = '/mnt/bulk-saturn/junhao/pathfinder/CRC/features/JUNHAO_CRC/MCO/prob_map'
    TCGA_path = '/mnt/bulk-saturn/junhao/pathfinder/CRC/features/JUNHAO_CRC/TCGA_CRC/prob_map'
    CR07_path = '/mnt/bulk-saturn/junhao/pathfinder/CRC/features/JUNHAO_CRC/CR07/prob_map'

    DACHS_raw = get_files(DACHS_path)
    MCO_raw = get_files(MCO_path)
    TCGA_raw = get_files(TCGA_path)
    CR07_raw = sorted(get_files(CR07_path),reverse=True)


    DACHS = path_clean(DACHS_raw)
    MCO = path_clean(MCO_raw)
    TCGA = path_clean(TCGA_raw)
    CR07 = path_clean(CR07_raw)
    print(len(DACHS))
    print(len(MCO))
    print(len(TCGA))
    print(len(CR07))


    #load trained model
    PATH = '/mnt/bulk-saturn/junhao/pathfinder/CRC/hist/CNN_raw'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnext50_32x4d()
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.DataParallel(model, device_ids=[0])
    model = model.to(device)
    model.eval()

    loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, MCO, args)
    print("[MCO] Apply model to test set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
    loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, TCGA, args)
    print("[TCGA] Apply model to test set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
    loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, CR07, args)
    print("[CR07] Apply model to test set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))

if __name__ == '__main__': 
    main()
