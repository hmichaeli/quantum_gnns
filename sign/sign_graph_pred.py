# [HM] This code is modified from the original source available at:
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/train.py
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
import dgl.function as fn
from dgl.transforms import SIGNDiffusion

from quantum_layers import PolyAct
import time
import random
from types import SimpleNamespace
from tqdm import tqdm

try:
    import wandb
except:
    print("Cannot import Weights and Biases")
    wandb = None


default_config = SimpleNamespace(
    dataset="MUTAG",
    num_epochs=350,
    batch_size=128,
    device="0",
    degrees_as_nlabel=False,
    model="qmodel_dense",
    
    sign_raw_hops=0,
    sign_rw_hops=0,
    sign_gcn_hops=3,
    sign_ppr_hops=0,

    num_hidden=16,
    activation="poly",
    poly_act_deg=2,
    norm=None,
    dropout=0.5,
    input_dropout=0,
    print_freq=50,
    wandb_project=None,
    wandb_group="default_group",
    wandb_name=None,
    wandb_sweep=False,
    seed=0,
    lr=0.01,
    weight_decay=0,
    num_train_folds=10,
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QModel(nn.Module):
    ''' Simpler implementation
        - Linear + activation per feature,
        - aggregation
        - Linear
    '''
    def __init__(self, in_feats, hidden, out_feats, R, dropout, activation='prelu', input_drop=0.0, norm=None, poly_act_deg=2):
        super(QModel, self).__init__()
        self.R = R
        self.input_drop = nn.Dropout(input_drop)

        self.feature_layers = nn.ModuleList()
        for i in range(R + 1):
            linear = nn.Linear(in_feats, hidden)
            # activation
            if activation == 'prelu':
                act = nn.PReLU()
            elif activation == 'poly':
                act = PolyAct(trainable=True, deg=poly_act_deg)
            else:
                act = nn.Identity()
            # norm
            if norm == 'batch':
                norm_layer = nn.BatchNorm1d(hidden)
            elif norm == 'layer':
                norm_layer = nn.LayerNorm(hidden)
            else:
                norm_layer = nn.Identity()

            self.feature_layers.append(nn.Sequential(linear, norm_layer, act))
    
        self.dropout = nn.Dropout(dropout)
        self.pool = SumPooling()
        self.project = nn.Linear(hidden * (R + 1), out_feats)

    def forward(self, g, feats):
        hidden = []
        feats = [self.input_drop(feat) for feat in feats]

        for feat, ff in zip(feats, self.feature_layers):
            hidden.append(ff(feat))
        hidden = torch.cat(hidden, dim=-1)
        hidden = self.pool(g, hidden)
        out = self.project(self.dropout(hidden))
        return out


class QModelDense(nn.Module):
    ''' Simpler implementation
        - Concat features
        - Linear + activation,
        - aggregation
        - Linear
    '''
    def __init__(self, in_feats, hidden, out_feats, R, dropout, activation='prelu', input_drop=0.0, norm=None, poly_act_deg=2):
        super(QModelDense, self).__init__()
        self.R = R
        self.input_drop = nn.Dropout(input_drop)

        linear = nn.Linear(in_feats * (R+1), hidden * (R+1))
        # activation
        if activation == 'prelu':
            act = nn.PReLU()
        elif activation == 'poly':
            act = PolyAct(trainable=True, deg=poly_act_deg)
        else:
            act = nn.Identity()
        # norm
        if norm == 'batch':
            norm_layer = nn.BatchNorm1d(hidden * (R+1))
        elif norm == 'layer':
            norm_layer = nn.LayerNorm(hidden * (R+1))
        else:
            norm_layer = nn.Identity()

        self.feature_layers = nn.Sequential(linear, norm_layer, act)
        

        self.dropout = nn.Dropout(dropout)
        self.pool = SumPooling()
        self.project = nn.Linear(hidden * (R + 1), out_feats)

    def forward(self, g, feats):
        
        feats = torch.cat(feats, dim=-1)
        feats = self.input_drop(feats)
        hidden = self.feature_layers(feats)
        
        hidden = self.pool(g, hidden)
        out = self.project(self.dropout(hidden))
        return out



class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(dropout)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer


def split_fold10(labels, fold_idx=0):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, valid_idx = idx_list[fold_idx]
    return train_idx, valid_idx


def evaluate(dataloader, device, model, sign_R=0):
    model.eval()
    total = 0
    total_correct = 0
    for batched_graph, labels in dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        feat = batched_graph.ndata.pop("attr")
        if sign_R > 0:
            feat = [feat]

            for k,v in batched_graph.ndata.items():
                if 'sign_attr' in k:
                    feat.append(v)
                
        total += len(labels)
        logits = model(batched_graph, feat)
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
    acc = 1.0 * total_correct / total
    return acc


def train(num_epochs, train_loader, val_loader, device, model, print_freq=50, sign_R=0, use_wandb=False, fold_idx=0, lr=0.01, weight_decay=0):
    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_val_acc = 0

    # training loop
    start_epoch_time = time.time()

    all_valid_acc = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels) in enumerate(train_loader):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            feat = batched_graph.ndata.pop("attr")
            if sign_R > 0:
                feat = [feat]

                for k,v in batched_graph.ndata.items():
                    if 'sign_attr' in k:
                        feat.append(v)
            
            logits = model(batched_graph, feat)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_acc = evaluate(train_loader, device, model, sign_R=sign_R)
        valid_acc = evaluate(val_loader, device, model, sign_R=sign_R)
        all_valid_acc.append(valid_acc)
        if use_wandb:
            wandb.log(
                {
                    f"fold_{fold_idx}/loss": total_loss / (batch + 1),
                    f"fold_{fold_idx}/train_acc": train_acc,
                    f"fold_{fold_idx}/val_acc": valid_acc,
                    f"fold_{fold_idx}/epoch": epoch,
                    f"fold_{fold_idx}/best_val_acc": best_val_acc,
                }
            )
        epoch_time = int(time.time() - start_epoch_time)
        if epoch % print_freq == 0 or valid_acc > best_val_acc:

            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
            print("Epoch {:05d} | Time {} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} | Best Acc. {:.4f} ".format(
                    epoch, epoch_time, total_loss / (batch + 1), train_acc, valid_acc, best_val_acc))
    print("Train time: {} [sec]".format(int(time.time() - start_epoch_time)))

    return best_val_acc, all_valid_acc


# create a parser to parse arguments, use config_defaults as default values
# use for reference the following code:
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_config.dataset,
        choices=["MUTAG", "PTC", "NCI1", "PROTEINS", "COLLAB", "IMDBBINARY", "IMDBMULTI", "REDDITBINARY", "REDDITMULTI5K"],
        help="name of dataset (default: MUTAG)",
    )
    parser.add_argument('--num-epochs', type=int, default=default_config.num_epochs, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=default_config.batch_size, help='batch size')
    parser.add_argument('--device', type=str, default=default_config.device, help='device to use')
    parser.add_argument('--degrees-as-nlabel', default=default_config.degrees_as_nlabel, action='store_true', help='use node degrees as input features')
    parser.add_argument('--model', type=str, default=default_config.model, help='model to use (GIN or QModel)')

    parser.add_argument('--sign-raw-hops', type=int, default=default_config.sign_raw_hops, help='number of raw hops')
    parser.add_argument('--sign-rw-hops', type=int, default=default_config.sign_rw_hops, help='number of random walk hops')
    parser.add_argument('--sign-gcn-hops', type=int, default=default_config.sign_gcn_hops, help='number of GCN hops')
    parser.add_argument('--sign-ppr-hops', type=int, default=default_config.sign_ppr_hops, help='number of PPR hops')

    parser.add_argument("--num-hidden", type=int, default=default_config.num_hidden)
    parser.add_argument('--activation', type=str, default=default_config.activation, help='activation function')
    parser.add_argument('--poly-act-deg', type=int, default=default_config.poly_act_deg, help='degree of polynomial activation')
    parser.add_argument('--norm', type=str, default=default_config.norm, help='normalization layer')
    parser.add_argument('--dropout', type=float, default=default_config.dropout, help='dropout rate')
    parser.add_argument("--input-dropout", type=float, default=default_config.input_dropout, help="dropout on input features")
    parser.add_argument('--print-freq', type=int, default=default_config.print_freq)
    parser.add_argument('--wandb-project', default=default_config.wandb_project, type=str, help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb-group', default=default_config.wandb_group, type=str, help='Group name for W&B')
    parser.add_argument('--wandb-name', default=default_config.wandb_name, type=str, help='Run name for W&B')
    parser.add_argument('--wandb-sweep', default=default_config.wandb_sweep, type=bool, help='Use wandb sweep')
    parser.add_argument('--seed', default=default_config.seed, type=int, help='Random seed for reproducibility')
    parser.add_argument('--lr', default=default_config.lr, type=float, help='Learning rate')
    parser.add_argument('--weight-decay', default=default_config.weight_decay, type=float, help='Weight decay')
    parser.add_argument('--num-train-folds', default=default_config.num_train_folds, type=int, help='Number of training folds (out of 10 splits)')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    use_wandb = False
    if args.wandb_project is not None and wandb is not None:
        use_wandb = True
        run = wandb.init(
                project=args.wandb_project,
                config=args,
                name=args.wandb_name,
                group=args.wandb_group,
                entity="quantun_dnns"
            )
        
        if args.wandb_sweep:
            # update args (wandb sweep)
            args = wandb.config
            
            # set run name to replace the sweep id
            random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=3))
            if args.activation == 'poly':
                print_act = f"poly{args.poly_act_deg}"
            else:
                print_act = args.activation
            wandb.run.name = f"{args.model}_raw{args.sign_raw_hops}_rw{args.sign_rw_hops}_gcn{args.sign_gcn_hops}_ppr{args.sign_ppr_hops}_\
            h{args.num_hidden}_{print_act}_{args.norm}_norm_\
            dp{args.dropout}_indp{args.input_dropout}_lr{args.lr}_wd{args.weight_decay}_seed{args.seed}_{random_str}"
            
    print("use wandb: ", use_wandb)

    set_random_seed(args.seed)


    print("args: ", args)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # load dataset
    print("loading dataset...")
    sign_transform = None
    sign_R = args.sign_raw_hops + args.sign_rw_hops + args.sign_gcn_hops + args.sign_ppr_hops
    
    if sign_R > 0:
        sign_transforms = []
        if args.sign_raw_hops > 0:
            sign_transforms.append(SIGNDiffusion(k=args.sign_raw_hops, in_feat_name='attr', out_feat_name='sign_attr_raw', diffuse_op='raw'))
        if args.sign_rw_hops > 0:
            sign_transforms.append(SIGNDiffusion(k=args.sign_rw_hops, in_feat_name='attr', out_feat_name='sign_attr_rw', diffuse_op='rw'))
        if args.sign_gcn_hops > 0:
            sign_transforms.append(SIGNDiffusion(k=args.sign_gcn_hops, in_feat_name='attr', out_feat_name='sign_attr_gcn', diffuse_op='gcn'))
        if args.sign_ppr_hops > 0:
            sign_transforms.append(SIGNDiffusion(k=args.sign_ppr_hops, in_feat_name='attr', out_feat_name='sign_attr_ppr', diffuse_op='ppr'))
                            

        print(f"Using SIGN: {sign_transforms}\n num hops: {sign_R}")
    else:
        print("Not using SIGN diffusion")
        sign_R = 0


    dataset = GINDataset(
        args.dataset, self_loop=True, degree_as_nlabel=args.degrees_as_nlabel,
    )  # add self_loop and disable one-hot encoding for input features

    if sign_transforms != []:
        print(f"apply transform  for all graphs:\n {sign_transforms}" )
        transformed_graphs = []
        for g in tqdm(dataset.graphs):
            g = g.to(device)
            for transform in sign_transforms:
                g = transform(g)
            transformed_graphs.append(g.cpu())
        dataset.graphs = transformed_graphs


    labels = [l for _, l in dataset]
    in_size = dataset.dim_nfeats
    out_size = dataset.gclasses
    print("done")


    # model 10-fold training/validating
    fold_accs = []
    print("Training...")
    folds_acc_list = []
    for fold_idx in range(min(args.num_train_folds, 10)):
        train_idx, valid_idx = split_fold10(labels, fold_idx)# split data
        train_loader = GraphDataLoader(
            dataset,
            sampler=SubsetRandomSampler(train_idx),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=0,
        )
        val_loader = GraphDataLoader(
            dataset,
            sampler=SubsetRandomSampler(valid_idx),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=0,
        )

        # create model
        if args.model == 'gin':
            model = GIN(in_size, args.num_hidden, out_size, dropout=args.dropout).to(device)
        elif args.model == 'qmodel':
            model = QModel(in_feats=in_size, hidden=args.num_hidden, out_feats=out_size, R=sign_R, dropout=args.dropout, 
                           activation=args.activation, input_drop=args.input_dropout, norm=args.norm,
                           poly_act_deg=args.poly_act_deg).to(device)
        elif args.model == 'qmodel_dense':
            model = QModelDense(in_feats=in_size, hidden=args.num_hidden, out_feats=out_size, R=sign_R, dropout=args.dropout, 
                           activation=args.activation, input_drop=args.input_dropout, norm=args.norm,
                           poly_act_deg=args.poly_act_deg).to(device)
        
        else:
            assert False, f"Model {args.model} not implemented"

        if fold_idx == 0:
            print(model)
        
        print("Fold {} training...".format(fold_idx))
        val_acc, fold_val_acc_list = train(args.num_epochs, train_loader, val_loader, device, model, sign_R=sign_R, 
                                           print_freq=args.print_freq, use_wandb=use_wandb, fold_idx=fold_idx, lr=args.lr,
                                            weight_decay=args.weight_decay)
        fold_accs.append(val_acc)
        print("Fold {} val acc: {:.4f}".format(fold_idx, val_acc))
        
        folds_acc_list.append(fold_val_acc_list)


    print("Average accuracy: {:.4f} std: {}".format(np.mean(fold_accs), np.std(fold_accs)))

    folds_acc_list = np.array(folds_acc_list)
    folds_avg_curve = np.mean(folds_acc_list, axis=0)
    folds_std_curve = np.std(folds_acc_list, axis=0)
    print("max avg val acc: ", np.max(folds_avg_curve))
    print("std of avg val acc: ", folds_std_curve[np.argmax(folds_avg_curve)])
          

    if use_wandb:
        
        if args.num_train_folds < 10:
            wandb.log({f"avg_val_acc_{args.num_train_folds}_folds": np.mean(fold_accs)})
            
        else:
            wandb.log({"avg_val_acc": np.mean(fold_accs)})
            wandb.log({"avg_val_curve/max": np.max(folds_avg_curve),
                        "avg_val_curve/std": folds_std_curve[np.argmax(folds_avg_curve)]
                        })

        wandb.finish()
