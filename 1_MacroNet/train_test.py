import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import RandomSampler

from data_loaders import SegHeatmapDatasetLoader
from networks import resnext50_32x4d, resnet18, regularize_path_weights
from utils import CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, count_parameters

import albumentations as A
from albumentations.pytorch import ToTensorV2


import os
import pickle


def train(train_data, test_data, k_th_fold, args, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print(device)
    cindex_test_max = 0
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2019)
    torch.manual_seed(2019)
    random.seed(2019)
    

    model = resnext50_32x4d()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=10)
    print("Number of Trainable Parameters: %d" % count_parameters(model))

    train_transform = A.Compose(
        [   
            A.Resize(256, 256),
            A.RandomRotate90(p = 0.75),
            ToTensorV2(),
        ]
    )

    custom_data_loader = SegHeatmapDatasetLoader(train_data, train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=args.batch_size, shuffle=True,drop_last=False)
    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[]}}
    
    for epoch in tqdm(range(args.epoch)):

        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        loss_epoch, grad_acc_epoch = 0, 0

        for batch_idx, (x_path, survtime, censor,_) in enumerate(train_loader):

            censor = censor.to(device)
            x_path = x_path.to(device).type(torch.cuda.FloatTensor)
            pred = model(x_path)

            loss_cox = CoxLoss(survtime, censor, pred, device)
            loss_reg = regularize_path_weights(model=model)
            loss = args.LAMBDA_COX*loss_cox + args.LAMBDA_REG*loss_reg
            loss_epoch += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information

        scheduler.step(loss)
        learning_rate = optimizer.param_groups[0]['lr']
        print(learning_rate)
        loss_epoch /= len(train_loader.dataset)

        cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)
        surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)
        grad_acc_epoch = None
        loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, test_data, args)

        metric_logger['train']['loss'].append(loss_epoch)
        metric_logger['train']['cindex'].append(cindex_epoch)
        metric_logger['train']['pvalue'].append(pvalue_epoch)
        metric_logger['train']['surv_acc'].append(surv_acc_epoch)
        metric_logger['train']['grad_acc'].append(grad_acc_epoch)

        metric_logger['test']['loss'].append(loss_test)
        metric_logger['test']['cindex'].append(cindex_test)
        metric_logger['test']['pvalue'].append(pvalue_test)
        metric_logger['test']['surv_acc'].append(surv_acc_test)
        metric_logger['test']['grad_acc'].append(grad_acc_test)


        print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}, {:s}: {:}'.format('Train', loss_epoch, 'C-Index', cindex_epoch, 'p-value', pvalue_epoch))
        print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}, {:s}: {:}\n'.format('Test', loss_test, 'C-Index', cindex_test, 'p-value', pvalue_test))


        save_path = args.save_path + '/{}th'.format(k_th_fold)
        if not os.path.exists(save_path): os.makedirs(save_path)

        epoch_idx = epoch
        if cindex_test_max < cindex_test: 
            cindex_test_max = cindex_test
            torch.save({
            'split':k_th_fold,
            'epoch':epoch_idx,
            'data': [train_data, test_data],
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metric_logger}, 
            save_path + '/best_weight.pkl')
            pickle.dump(pred_test, open(save_path + '/pred_test_{}.pkl'.format(k_th_fold), 'wb'))

        if epoch_idx == args.epoch-1:
            torch.save({
            'split':k_th_fold,
            'epoch':epoch_idx,
            'data': [train_data, test_data],
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metric_logger}, 
            save_path + '/{}.pkl'.format(epoch_idx))

        pickle.dump(pred_test, open(save_path + '/pred_test_{}.pkl'.format(epoch_idx), 'wb'))

    return model, optimizer, metric_logger


def test(model, data, args, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model.eval()

    test_transform = A.Compose(
        [   
            A.Resize(256, 256),
            ToTensorV2(),
        ]
    )
    custom_data_loader = SegHeatmapDatasetLoader(data, test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=args.batch_size, shuffle=False,drop_last=False)
    
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    file_path = []
    loss_test, grad_acc_test = 0, 0

    for batch_idx, (x_path, survtime, censor, seg_filepath) in enumerate(test_loader):

        censor = censor.to(device)
        x_path = x_path.to(device).type(torch.FloatTensor)
        pred = model(x_path)


        loss_cox = CoxLoss(survtime, censor, pred, device)
        loss_reg = regularize_path_weights(model=model)
        loss = args.LAMBDA_COX*loss_cox + args.LAMBDA_REG*loss_reg
        loss_test += loss.data.item()
        gt_all = None

        file_path.append(seg_filepath)
        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information

    
    ################################################### 
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader.dataset)
    
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    grad_acc_test = None
    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all, file_path]

    if args.external_test:
        cohort_name = file_path[0][0].split('/')[-3]
        if not os.path.exists(args.save_path): os.makedirs(args.save_path)
        pickle.dump(pred_test, open(args.save_path + '/pred_test_{}.pkl'.format(cohort_name), 'wb'))

    return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test
