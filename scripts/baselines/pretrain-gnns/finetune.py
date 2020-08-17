import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from splitters import scaffold_split
import pandas as pd

import os
import shutil

import json

from tensorboardX import SummaryWriter

criterion = nn.BCEWithLogitsLoss(reduction = "none")
criterion = nn.MSELoss()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.gnn.state_dict(), self.path)
        self.val_loss_min = val_loss

def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # print(torch.is_tensor(batch.edge_index),batch.edge_index.dtype)
        batch.edge_index = batch.edge_index.long()
        # print(torch.is_tensor(batch.edge_index), batch.edge_index.dtype)
        batch = batch.to(device)
        # print(torch.is_tensor(batch.edge_index),batch.edge_index.dtype)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss = criterion(pred.double(), y)
        #loss matrix after removing null target
        # loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        #
        optimizer.zero_grad()
        # loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader, scaler, train = False):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        if train:
            y_scores.append(pred)
        else:
            y_scores.append(torch.Tensor(scaler.inverse_transform(pred.cpu())))

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    r2 = r2_score(y_true, y_scores)
    rmse = mean_squared_error(y_true, y_scores, squared = False)

    return r2, rmse #y_true.shape[1]



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--dataset', type=str, default = '../../../data/raw/baselines/pretrain_gnn/logp/train', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--val_dataset', type=str, default = '../../../data/raw/baselines/pretrain_gnn/logp/val', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--test_dataset', type=str, default = '../../../data/raw/baselines/pretrain_gnn/logp/test', help='root directory of dataset. For now, only classification.')

    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = 'exp_0', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="default", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--log_path', type=str, default='../../../data/raw/baselines/pretrain_gnn/logs2', help='root directory for logs')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if "tox21" in args.dataset:
        num_tasks = 12
    elif "hiv" in args.dataset:
        num_tasks = 1
    elif "pcba" in args.dataset:
        num_tasks = 128
    elif "muv" in args.dataset:
        num_tasks = 17
    elif "bace" in args.dataset:
        num_tasks = 1
    elif "bbbp" in args.dataset:
        num_tasks = 1
    elif "toxcast" in args.dataset:
        num_tasks = 617
    elif "sider" in args.dataset:
        num_tasks = 27
    elif "clintox" in args.dataset:
        num_tasks = 2
    elif "logp" in args.dataset:
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset(args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv(args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    elif args.split == "default":
        train_dataset = dataset
        valid_dataset = MoleculeDataset(args.val_dataset, dataset=args.dataset)
        test_dataset = MoleculeDataset(args.test_dataset, dataset=args.dataset)

    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])
    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))
    print(train_dataset.data.y)

    scaler = StandardScaler()
    scaled_y = torch.tensor(scaler.fit_transform(train_dataset.data.y.reshape(-1, 1)).reshape(-1))
    train_dataset.data.y = scaled_y
    print(train_dataset.data.y)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(os.path.join(args.input_model_file, args.filename), device)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []




    if not args.filename == "":

        fname = os.path.join(args.log_path, args.input_model_file, args.filename)
        os.makedirs(fname, exist_ok=True)
        # #delete the directory if there exists one
        # if os.path.exists(fname):
        #     shutil.rmtree(fname)
        #     print("removed the existing file.")
        writer = SummaryWriter(fname)
        with open(os.path.join(fname, 'parameters.json'), 'w') as f:
            json.dump(vars(args), f)

    early_stopping = EarlyStopping(patience=50, verbose=True, path=os.path.join(fname, args.filename + '.pth'))

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_r2, train_rmse = eval(args, model, device, train_loader, scaler, train = True)
        else:
            print("omit the training accuracy computation")
            train_r2, train_rmse = 0, 0
        val_r2, val_rmse = eval(args, model, device, val_loader, scaler)
        test_r2, test_rmse = eval(args, model, device, test_loader, scaler)

        print("train r2: %f\ntrain rmse: %f\n val r2: %f\n val rmse: %f\ntest r2: %f\ntest rmse: %f"\
              %(train_r2, train_rmse, val_r2, val_rmse, test_r2, test_rmse))

        # val_acc_list.append(val_acc)
        # test_acc_list.append(test_acc)
        # train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train r2', train_r2, epoch)
            writer.add_scalar('data/train rmse', train_rmse, epoch)

            writer.add_scalar('data/val r2', val_r2, epoch)
            writer.add_scalar('data/val rmse', val_rmse, epoch)
            writer.add_scalar('data/test r2', test_r2, epoch)
            writer.add_scalar('data/test rmse', test_rmse, epoch)

        early_stopping(val_rmse, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        print("")

    if not args.filename == "":
        writer.close()
        # torch.save(model.gnn.state_dict(), os.path.join(fname, args.filename+'.pth'))

if __name__ == "__main__":
    main()
