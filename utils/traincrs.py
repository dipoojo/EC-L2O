
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn



from utils.preprocess import sliding_windows, load_power_shortage
from utils.loss import object_loss_cost, object_loss_cr
from utils.model import LSTM_unroll
from utils.dataset import TrajectCR_Dataset
from tqdm import tqdm
import json 
import argparse

n_iter = 0
n_iter_val = 0
use_cuda = False

def train_cr_vals(ml_model, optimizer, writer, train_dataloader, demand_validation, 
            num_epoch, switch_weight, min_cr, mtl_weight = 0.5, mute=True, l_1=1, l_2=1, l_3=1, calib=True):
    from utils.loss import object_loss_cost as objl
    from utils.loss import object_loss_cr as objlr
    global n_iter, n_iter_val, use_cuda
    
    if not mute:
        epoch_iter = tqdm(range(num_epoch))
    else:
        epoch_iter = range(num_epoch)
    
    for _ in epoch_iter:
        ml_model.train()
        for _, (demand,opt_cost) in enumerate(train_dataloader):
            demand = demand.float()
            if use_cuda: 
                demand = demand.cuda()
                opt_cost = opt_cost.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            action_ml = ml_model(demand, calib = calib)
            
            
            if mtl_weight == 1.0:
                loss_calib = torch.zeros((1,1))
                
                loss_ml = object_loss_cost(demand, action_ml, c=switch_weight)
                # loss_ml = object_loss_cr(demand, action_ml, opt_cost, min_cr = min_cr, c=switch_weight)
                loss = loss_ml

            else:
                loss_ml = objlr(demand, action_ml, opt_cost, min_cr = min_cr, c=switch_weight)
                
                action_calib = ml_model(demand, calib = calib)
                loss_calib = object_loss_cost(demand, action_calib, c = switch_weight)
                
                loss = mtl_weight*loss_ml + (1-mtl_weight)*loss_calib

            loss.backward()
            optimizer.step()
#updated names to include scalars for easy tracking
            writer.add_scalar(f'Loss_train/no_calib_{l_1}_{l_2}_{l_3}', loss_ml.item(), n_iter)
            writer.add_scalar(f'Loss_train/with_calib_{l_1}_{l_2}_{l_3}', loss_calib.item(), n_iter)
            writer.add_scalar(f'Loss_train/overall_{l_1}_{l_2}_{l_3}', loss.item(), n_iter)
            n_iter += 1

        writer.flush()
        
        # Calculate evaluation cost
        ml_model.eval()

        with torch.no_grad():
            #action_val_ml    = ml_model(demand_validation, mode="val", calib=False)
            action_val_calib = ml_model(demand_validation, mode="val", calib=True)

            #loss_val_ml = object_loss_cost(demand_validation, action_val_ml, c = switch_weight)
            #loss_val_calib = object_loss_cost(demand_validation, action_val_calib, c = switch_weight)
#updated names with lambdas for ease of tracking
        #writer.add_scalar(f'Loss_val/no_calib_{l_1}_{l_2}_{l_3}', loss_val_ml.item()/100, n_iter_val)
        #writer.add_scalar(f'Loss_val/with_calib_{l_1}_{l_2}_{l_3}', loss_val_calib.item()/100, n_iter_val)
        n_iter_val += 1

    writer.close()



def train_cr_vals_dynamic(ml_model, optimizer, writer, train_dataloader, val_dataloader,opt_cost_dynamic, 
            num_epoch, switch_weight, min_cr, mtl_weight = 0.5, mute=True, l_1=1, l_2=1, l_3=1, calib=True):
    from utils.loss import object_loss_cost as objl
    from utils.loss import object_loss_cr as objlr
    global n_iter, n_iter_val, use_cuda
    
    if not mute:
        epoch_iter = tqdm(range(num_epoch))
    else:
        epoch_iter = range(num_epoch)
    
    for j in epoch_iter:
        #print(f'epoch is {j}')
        ml_model.train()
        for k, (demand,opt_cost) in enumerate(train_dataloader):
            #print(f'batch is {k} ***********')
            demand = demand.float()
            
            opt_cost = torch.reshape(opt_cost, (opt_cost.shape[0], 1))
            #opt_cost = np.expand_dims(opt_cost, axis=1)
            #opt_cost = opt_cost.float()
            if use_cuda: 
                demand = demand.cuda()
                opt_cost = opt_cost.cuda()
            #print(demand.shape)
            #print(opt_cost.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            #demand = demand.type('torch.FloatTensor')
            #opt_cost = opt_cost.type('torch.FloatTensor')
            #action_ml = ml_model(demand, torch.from_numpy(opt_cost).float(), calib = calib)
            action_ml = ml_model(demand, opt_cost, calib = calib)
            
            
            if mtl_weight == 1.0:
                loss_calib = torch.zeros((1,1))
                
                loss_ml = object_loss_cost(demand, action_ml, c=switch_weight)
                # loss_ml = object_loss_cr(demand, action_ml, opt_cost, min_cr = min_cr, c=switch_weight)
                loss = loss_ml
                #print(f'loss is {loss}')

            else:
                loss_ml = objlr(demand, action_ml, opt_cost, min_cr = min_cr, c=switch_weight)
                #print(demand.dtype)
                #print(opt_cost.dytpe)
                action_calib = ml_model(demand, opt_cost, calib = calib)
                loss_calib = object_loss_cost(demand, action_calib, c = switch_weight)
                #print(f'loss calib is {loss_calib}')
                #print(f'loss is {loss_ml}')
                
                loss = mtl_weight*loss_ml + (1-mtl_weight)*loss_calib
                #print(f'final loss is {loss}')

            loss.backward()
            optimizer.step()
#updated names to include scalars for easy tracking
            writer.add_scalar(f'Loss_train/no_calib_{l_1}_{l_2}_{l_3}', loss_ml.item(), n_iter)
            writer.add_scalar(f'Loss_train/with_calib_{l_1}_{l_2}_{l_3}', loss_calib.item(), n_iter)
            writer.add_scalar(f'Loss_train/overall_{l_1}_{l_2}_{l_3}', loss.item(), n_iter)
            n_iter += 1

        writer.flush()
        
        # Calculate evaluation cost
        ml_model.eval()

        with torch.no_grad():
            for _, (demand,opt_cost) in enumerate(val_dataloader):
                demand = demand.float()
            
                opt_cost = torch.reshape(opt_cost, (opt_cost.shape[0], 1))
                #opt_cost = np.expand_dims(opt_cost, axis=1)
                #opt_cost = opt_cost.float()
                if use_cuda: 
                    demand = demand.cuda()
                    opt_cost = opt_cost.cuda()
                    #print(demand.shape)
                    #print(opt_cost.shape)
                    # zero the parameter gradients
                    #optimizer.zero_grad()
                    #demand = demand.type('torch.FloatTensor')
                    #opt_cost = opt_cost.type('torch.FloatTensor')
                    #action_ml = ml_model(demand, torch.from_numpy(opt_cost).float(), calib = calib)
                action_ml = ml_model(demand, opt_cost, calib = False)
                action_val_calib = ml_model(demand, opt_cost, calib = True)
            #action_val_ml    = ml_model(demand_validation, opt_cost_dynamic, mode="val", calib=False)
            #action_val_calib = ml_model(demand_validation, opt_cost_dynamic, mode="val", calib=True)

            #loss_val_ml = object_loss_cost(demand_validation, action_val_ml, c = switch_weight)
            #loss_val_calib = object_loss_cost(demand_validation, action_val_calib, c = switch_weight)
#updated names with lambdas for ease of tracking
        #writer.add_scalar(f'Loss_val/no_calib_{l_1}_{l_2}_{l_3}', loss_val_ml.item()/100, n_iter_val)
        #writer.add_scalar(f'Loss_val/with_calib_{l_1}_{l_2}_{l_3}', loss_val_calib.item()/100, n_iter_val)
        n_iter_val += 1

    writer.close()    