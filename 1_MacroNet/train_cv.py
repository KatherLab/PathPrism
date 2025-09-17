import os
import numpy as np
import pickle
from data_loaders import *
from train_test import train, test
import argparse
import logging
import pickle

#clinical info of different cohorts
DACHS_info_dir = '/mnt/bulk-saturn/junhao/pathfinder/CRC/info/DACHS_merged.csv'
DACHS_data = pd.read_csv(DACHS_info_dir)


def path_clean(paths, DACHS_data = DACHS_data):
    cohort = 'DACHS'
    current_ID = []
    cleaned_path = []

    if cohort == 'DACHS':
        for seg_filepath in paths:
            ID = seg_filepath.split('/')[-1][:9]
            if DACHS_data['PATIENT'].isin([ID]).any():
                if ID not in current_ID:
                    current_ID.append(ID)
                    cleaned_path.append(seg_filepath)
    else:
        print('No {} cohort'.format(cohort))
    
    return cleaned_path


def all_path(dirname, file_type):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            # if file_type in apath[-4:]:
            if file_type in apath:
                result.append(apath)
    return result


def get_args():
		
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--external_test', type=bool, default=False, help='make extrenal tests')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    # parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--cv_fold', type=int, default=5, help='cross validation fold')
    parser.add_argument('--save_path', type=str, default='/mnt/bulk-saturn/junhao/pathfinder/CRC/hist/CNN_raw', help='ckpt save path')
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


def path_k_fold(path_list, k):
    k_fold_path = []
    length = len(path_list)
    step_size = int(length/k)
    for i in range(k):
        if i<k-1:
            test_set = path_list[i*step_size:(i+1)*step_size]
            train_set = list(set(path_list).difference(set(test_set)))
        else:
            test_set = path_list[i*step_size:(i+1)*step_size]
            train_set = list(set(path_list).difference(set(test_set)))
        k_fold_path.append([train_set, test_set])
    
    return k_fold_path


def main():

    args = get_args()
    args_dict = {k: v for k, v in args._get_kwargs()}
    print(args_dict)
    print(args)
    
    DACHS_path = '/mnt/bulk-saturn/junhao/pathfinder/CRC/features/JUNHAO_CRC/DACHS/prob_map'
    DACHS_raw = get_files(DACHS_path)
    DACHS = path_clean(DACHS_raw)
    print(len(DACHS))
    cv_path = path_k_fold(DACHS,k = args.cv_fold)

    k = 0
    for path in cv_path:
        train_data = path[0]
        test_data = path[1]
        print(len(train_data))
        print(len(test_data))

        model, optimizer, metric_logger = train(train_data,test_data,k,args)
        k += 1
        loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train = test(model, train_data, args)
        loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, test_data, args)

        print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
        logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
        print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
        logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))


if __name__ == '__main__': 
    main()

