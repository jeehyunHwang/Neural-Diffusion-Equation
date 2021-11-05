from __future__ import division
from __future__ import print_function

import sys
import os

def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Neural Diffusion Equation Implementation',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file (YAML format)",
                        metavar="FILE",
                        # default="LA.yaml",
                        default="SD.yaml",
                        )
    
    parser.add_argument("--model_path",
                        dest="modelpath",
                        help="load pretrained model",
                        default=False)
    
    parser.add_argument("--comment",
                type=str,
                help="comment",
                default="")

    parser.add_argument("--gpu",
                type=str,
                help="gpu num",
                default="0")

    return parser

########## Device setting ##########
args = get_parser().parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
####################################

import logging
import pprint
# import socket
import datetime

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from torch_geometric.data import Data

from utils import get_laplacian
from model import ODENet
from blocks import ODEfunc, ODEBlock

# LA/SD
TIME_DIM = 384
LAT_DIM = 141    # vertical
LONG_DIM = 129   # horizontal

########## Device setting ##########
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
####################################

def main(cfg):

    use_mini = cfg['model']['MINI_NN']
    use_physics = cfg['model']['PHY_EQ']
    MODE_DESC = cfg['model']['MODE_desc']
    REGION = cfg['dataset']['REGION']    # LA or SD

    dirname = "_".join([cfg['comment'], MODE_DESC, REGION, "NN"+str(use_mini), "PHY"+str(cfg['model']['PHY_EQ']), "Enc"+str(cfg['model']['enc_node_feat']), "lr"+str(cfg['optimizer']['initial_lr']), "decay"+str(cfg['optimizer']['weight_decay']), datetime.datetime.now().isoformat()[:19]])
    logdir = os.path.join("results", dirname)
    modeldir = os.path.join(logdir, "model")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    logfilename = os.path.join(logdir, 'log.txt')
    # Print the configuration - just to make sure that you loaded what you wanted to load
    with open(logfilename, 'w') as f:
        pp = pprint.PrettyPrinter(indent=4, stream=f)
        pp.pprint(cfg)
    
    logging.basicConfig(filename=logfilename,
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG)
    writer = SummaryWriter(logdir)
    
    logging.info("USE MINI: {} USE PHYSICS: {} ({}) REGION: {}".format(use_mini, use_physics, MODE_DESC, REGION))
    logging.info("logdir: {}".format(logdir))
    logging.info("modeldir: {}".format(modeldir))

    ########## Load data and edge attributes ##########
    # edge attribute not used in current version
    X = np.load(cfg['dataset']['X_path'])
    edge_index = np.load(cfg['dataset']['edge_index_path'])
    edge_attr = np.load(cfg['dataset']['edge_attr_path'])
    edge_attr_type = len(set(edge_attr.tolist()))
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    edge_attr = torch.tensor(edge_attr)
    num_nodes = X.shape[1]
    ###################################################


    ########## Architecture setting ##########
    node_features = X.shape[2] - 1    # Temperature is not considered as input
    enc_node_features = cfg['model']['enc_node_feat']
    mininet_dim_size = cfg['model']['mininet_dim']
    output_size = 1    # predict Temperature
    sp_L = get_laplacian(edge_index, type="aug", sparse=False).to(device)
    time_dependent = cfg['model']['time_dependent']
    nonlinear = cfg['model']['nonlinear']
    activation = cfg['model']['activation']
    method = cfg['model']['method']
    tol = cfg['model']['tol']
    adjoint = cfg['model']['adjoint']
    use_initial = cfg['model']['use_initial']
    dropout_rate = cfg['model']['dropout_rate']
    ##########################################

    ############ Training setting ############
    num_processing_steps = cfg['train']['num_processing_steps']    # Forecast horizon
    num_iterations = cfg['train']['num_iter']
    multistep = cfg['train']['multistep']
    input_seq = cfg['train']['input_sequence']
    valid_iter = cfg['train']['valid_iter']
    ##########################################

    losses_save = []
    val_losses_save = []

    ####### create physics coefficient matrix using edge attribute shape ###########
    one_hot_encoder = torch.zeros(size=(num_nodes*num_nodes, edge_attr_type), device=device)
    for i in range(len(edge_attr)):
        one_hot_encoder[edge_index[:,i][0]*num_nodes + edge_index[:,i][1], edge_attr[i]] = 1

    #### Model ####
    if cfg['modelpath']:
        model = ODENet(node_features,
                enc_node_features,
                sp_L,
                one_hot_encoder,
                edge_attr_type,
                num_nodes,
                use_mini=use_mini,
                use_physics=use_physics, 
                augment_dim=0,
                enc_desc=[['relu']],
            #    enc_desc=None,
                dec_desc=[[32, 'relu'], [None]],
            #    dec_desc=None,
                mini_nn_desc=[[128, 'relu'], ['tanh']],
                k_enc_desc=None,
                time_dependent=time_dependent, 
                num_processing_steps=num_processing_steps, 
                use_initial=use_initial,
                multistep=multistep,
                method=method,
                adjoint=adjoint,
                tol=tol, 
                dropout_rate=dropout_rate, 
                device=device)
        model.load_state_dict(torch.load(cfg['modelpath'], map_location=device))
        logging.info("pretrained model is loaded. {}".format(cfg['modelpath']))

    else:
        model = ODENet(node_features * input_seq,
                       enc_node_features,
                       sp_L,
                       one_hot_encoder,
                       edge_attr_type,
                       num_nodes,
                       use_mini=use_mini,
                       use_physics=use_physics, 
                       augment_dim=0,
                       enc_desc=[['relu']],       #[TODO] LA
                    #    enc_desc=[['relu']],       #[TODO] SD
                       dec_desc=[[32, 'relu'], [None]],         #[TODO] LA
#                       dec_desc=[[32, 'relu'], [None]],       #[TODO] SD
                       mini_nn_desc=[[128, 'relu'], ['tanh']],          #[TODO] LA
                    #    mini_nn_desc=[[256, 'relu'], ['tanh']],          #[TODO] SD
                       time_dependent=time_dependent, 
                       num_processing_steps=num_processing_steps, 
                       use_initial=use_initial,
                       multistep=multistep,
                       method=method,
                       adjoint=adjoint,
                       tol=tol, 
                       dropout_rate=dropout_rate, 
                       device=device)
        logging.info("new model is initialized. {}".format(modeldir))

    num_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("# params in model: {}".format(num_total_params))
    logging.info("new model : \n{}".format(model))
    model.to(device)
    model.train()
    phy_params = []
    best_result = np.inf
    
    optimizer = optim.RMSprop(model.parameters(), 
                            lr=0.001, 
                            weight_decay=cfg['optimizer']['weight_decay'])
    tr_ind, val_ind, te_ind = 250, 300, TIME_DIM-1    # training/validation/test split

    epoch_iter = (tr_ind - num_processing_steps - multistep) // num_processing_steps
    
    #### Training
    for iter_ in range(num_iterations):
        t = np.random.randint(input_seq - 1, tr_ind - num_processing_steps - multistep)

        if input_seq == 1:
            input_data = [Data(x=torch.tensor(X[t+step_t,:,1:], dtype=torch.float32, device=device))
                           for step_t in range(num_processing_steps)]
        else:
            input_data = [Data(x=torch.tensor(np.concatenate([X[t+step_t+i,:,1:] for i in range(input_seq)], 1)\
                            , dtype=torch.float32, device=device))
                            for step_t in range(num_processing_steps)]

        eval_times = [Data(t=torch.tensor([t+step_t+input_seq-1,t+step_t+input_seq], dtype=torch.float32, device=device))
                           for step_t in range(num_processing_steps)]
        
        outputs, phy_params = model(input_data, eval_times, num_processing_steps)
        
        if use_initial:
            losses = [sum([torch.sum((out - torch.tensor(X[t+step_t+multi+input_seq-1,:,:1], dtype=torch.float32, device=device))**2)
                                for multi, out in enumerate(output)])/len(output) for step_t, output in enumerate(outputs)]
        else:
            losses = [sum([torch.sum((out - torch.tensor(X[t+1+step_t+multi+input_seq,:,:1], dtype=torch.float32, device=device))**2)
                                for multi, out in enumerate(output)])/len(output) for step_t, output in enumerate(outputs)]
        
        loss = sum(losses) / len(losses)
        losses_save.append(loss.item())
        
        writer.add_scalars('loss/train', {'loss': losses_save[-1]}, iter_)
        writer.add_scalars('loss/train', {'loss_per_node': losses_save[-1]/num_nodes}, iter_)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_ == 0:
            if use_mini and use_physics:
                torch.save(model.state_dict(), os.path.join(modeldir, "MINI NN + PHY"))
            elif use_mini and not use_physics:
                torch.save(model.state_dict(), os.path.join(modeldir, "MINI NN"))
            elif not use_mini and use_physics:
                torch.save(model.state_dict(), os.path.join(modeldir, "PHY"))
                
        #### Validation
        if iter_%valid_iter == 0:
            losses_val_save = []

            if input_seq == 1:
                input_data = [Data(x=torch.tensor(X[tr_ind+step_t,:,1:], dtype=torch.float32, device=device)) 
                                for step_t in range(50 - multistep)]
            else:
                input_data = [Data(x=torch.tensor(np.concatenate([X[tr_ind+step_t+i,:,1:] for i in range(input_seq)], 1)\
                                , dtype=torch.float32, device=device))
                                for step_t in range(51 - multistep - input_seq)]
            eval_times = [Data(t=torch.tensor([tr_ind+step_t+input_seq-1,tr_ind+step_t+input_seq], dtype=torch.float32, device=device))
                            for step_t in range(50 - multistep)]
            outputs, phy_params = model(input_data, eval_times, 51 - multistep - input_seq)
            
            if use_initial:
                val_losses = [sum([torch.sum((out - torch.tensor(X[tr_ind+multi+step_t+input_seq,:,:1], dtype=torch.float32, device=device))**2)
                                for multi, out in enumerate(output[1:])])/len(output[1:]) for step_t, output in enumerate(outputs)]
            else:
                val_losses = [sum([torch.sum((out - torch.tensor(X[tr_ind+multi+step_t+input_seq,:,:1], dtype=torch.float32, device=device))**2)
                                for multi, out in enumerate(output)])/len(output) for step_t, output in enumerate(outputs)]

            val_loss = sum(val_losses) / len(val_losses)
            losses_val_save.append(val_loss.item())

            #### Test
            if (len(val_losses_save)>0) and (np.mean(losses_val_save)<np.min(val_losses_save)):
                if use_mini and use_physics:
                    torch.save(model.state_dict(), os.path.join(modeldir, "MINI NN + PHY"))
                elif use_mini and not use_physics:
                    torch.save(model.state_dict(), os.path.join(modeldir, "MINI NN"))
                elif not use_mini and use_physics:
                    torch.save(model.state_dict(), os.path.join(modeldir, "PHY"))
                    
                if input_seq == 1:
                    input_data = [Data(x=torch.tensor(X[val_ind+step_t,:,1:], dtype=torch.float32, device=device)) 
                                    for step_t in range(te_ind - val_ind - multistep - 1)]
                else:
                    input_data = [Data(x=torch.tensor(np.concatenate([X[val_ind+step_t+i,:,1:] for i in range(input_seq)], 1)\
                                    , dtype=torch.float32, device=device))
                                    for step_t in range(te_ind - val_ind - multistep - input_seq)]
                eval_times = [Data(t=torch.tensor([val_ind+step_t+input_seq-1,val_ind+step_t+input_seq], dtype=torch.float32, device=device))
                                    for step_t in range(te_ind - val_ind - multistep - 1)]

                outputs, phy_params = model(input_data, eval_times, te_ind - val_ind - multistep - input_seq)

                if use_initial:
                    te_losses = [sum([torch.sum((out - torch.tensor(X[val_ind+multi+step_t+1,:,:1], dtype=torch.float32, device=device))**2) 
                                    for multi, out in enumerate(output[1:])])/len(output[1:]) for step_t, output in enumerate(outputs)]
                else:
                    te_losses = [sum([torch.sum((out - torch.tensor(X[val_ind+multi+step_t+1,:,:1], dtype=torch.float32, device=device))**2) 
                                    for multi, out in enumerate(output)])/len(output) for step_t, output in enumerate(outputs)]
                te_loss = sum(te_losses) / len(te_losses)

                if te_loss.item() <= best_result:
                    best_result = te_loss.item()

                writer.add_scalars('loss/test', {'loss_sup': te_loss.item()}, iter_)
                writer.add_scalars('loss/test', {'loss_sup_per_node': te_loss.item()/num_nodes}, iter_)
                logging.info("{}/{} iterations.".format(iter_, num_iterations))
                logging.info("[Train]Loss: {:.4f} [Vali]Loss_sup: {:.4f}({:.4f}) [Test]Loss_sup: {:.4f}({:.4f})"
                             .format(loss,
                                     np.mean(losses_val_save), np.mean(losses_val_save)/num_nodes, 
                                     te_loss.item(), te_loss.item()/num_nodes))

            val_losses_save.append(np.mean(losses_val_save))
            writer.add_scalars('loss/valid', {'loss_sup': val_losses_save[-1]}, iter_)
            writer.add_scalars('loss/valid', {'loss_sup_per_node': val_losses_save[-1]/num_nodes}, iter_)
            
            
        if iter_%epoch_iter == 0:
            logging.info("{}/{} iterations.".format(iter_, num_iterations))
            logging.info("[Train]Loss: {:.4f} [Vali]Loss_sup: {:.4f}({:.4f})"
                         .format(loss, np.mean(losses_val_save), np.mean(losses_val_save)/num_nodes))


    logging.info("[Training]The smallest supervised loss: {:.4e}({:.4e}) at {}/{}"
                 .format(np.min(losses), np.min(losses)/num_nodes, np.argmin(losses), len(losses)))
    logging.info("[Vali]The smallest supervised loss: {:.4e}({:.4e}) at {}/{}"
                 .format(np.min(val_losses_save), np.min(val_losses_save)/num_nodes, np.argmin(val_losses_save), len(val_losses_save)))

    
def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    root_path = os.path.dirname(os.path.abspath(__file__))
    yaml_filepath = os.path.join(root_path,str(yaml_filepath))
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    return cfg

def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg
    


    
if __name__=="__main__":
    cfg = load_cfg("./cfg_files_ode/" + args.filename)
    cfg['modelpath'] = args.modelpath
    cfg['comment'] = args.comment
    main(cfg)

