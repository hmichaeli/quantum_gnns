# [HM] This code is modified from the original source available at:
#  https://github.com/dmlc/dgl/blob/master/examples/pytorch/sign/sign.py
import argparse
import os
import time

import dgl
import dgl.function as fn
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import load_dataset
from quantum_layers import PolyAct

# decision problem import
from quantum_layers import class_pairs_mask, label_pairs_count

try:
    import wandb
except:
    print("Cannot import Weights and Biases")
    wandb = None


default_config = argparse.Namespace(
    num_epochs=1000,
    num_hidden=512,
    R=5,
    lr=0.001,
    dataset="amazon",
    dropout=0.4,
    input_dropout=0.3,
    gpu=0,
    weight_decay=0,
    eval_every=10,
    eval_first_epochs=0,
    eval_batch_size=250000,
    ff_layer=2,
    model='qmodel',
    activation='poly',
    poly_act_deg=2,
    train_batch_size=1000,
    wandb_project=None,
    wandb_group="default_group",
    wandb_name=None,
    wandb_sweep=False,
    seed=0,
    eval_decision=True,
    out_dir=None
)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x

class Model(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, R, n_layers, dropout):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        for hop in range(R + 1):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout)
            )
        # self.linear = nn.Linear(hidden * (R + 1), out_feats)
        self.project = FeedForwardNet(
            (R + 1) * hidden, hidden, out_feats, n_layers, dropout
        )

    def forward(self, feats):
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return out

class QModel(nn.Module):
    ''' Simpler implementation
        - Linear + activation per feature,
        - aggregation
        - Linear
    '''
    def __init__(self, in_feats, hidden, out_feats, R, dropout, activation='prelu', input_drop=0.0, poly_act_deg=2):
        super(QModel, self).__init__()
        self.in_feats = in_feats
        self.hidden = hidden
        self.R = R
        self.input_drop = nn.Dropout(input_drop)
        self.feature_layers = nn.ModuleList()
        for i in range(R + 1):
            linear = nn.Linear(in_feats, hidden)
            if activation == 'prelu':
                act = nn.PReLU()
            elif activation == 'poly':
                act = PolyAct(trainable=True, deg=poly_act_deg)
            else:
                act = nn.Identity()
            self.feature_layers.append(nn.Sequential(linear, act))
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(hidden * (R + 1), out_feats)

    def forward(self, feats):
        hidden = []
        feats = [self.input_drop(feat) for feat in feats]
        for feat, ff in zip(feats, self.feature_layers):
            hidden.append(ff(feat))
        out = self.project(self.dropout(torch.cat(hidden, dim=-1)))
        return out

    def reset_parameters(self):  # Add the missing reset_parameters method
        print("reset parameters")
        gain = nn.init.calculate_gain("relu")
        for layer in self.feature_layers:
            nn.init.xavier_uniform_(layer[0].weight, gain=gain)
            nn.init.zeros_(layer[0].bias)
    
        nn.init.xavier_uniform_(self.project.weight, gain=gain)
        nn.init.zeros_(self.project.bias)
    


def calc_weight(g):
    """
    Compute row_normalized(D^(-1/2)AD^(-1/2))
    """
    with g.local_scope():
        # compute D^(-0.5)*D(-1/2), assuming A is Identity
        g.ndata["in_deg"] = g.in_degrees().float().pow(-0.5)
        g.ndata["out_deg"] = g.out_degrees().float().pow(-0.5)
        g.apply_edges(fn.u_mul_v("out_deg", "in_deg", "weight"))
        # row-normalize weight
        g.update_all(fn.copy_e("weight", "msg"), fn.sum("msg", "norm"))
        g.apply_edges(fn.e_div_v("weight", "norm", "weight"))
        return g.edata["weight"]


def preprocess(g, features, args):
    """
    Pre-compute the average of n-th hop neighbors
    """
    with torch.no_grad():
        g.edata["weight"] = calc_weight(g)
        g.ndata["feat_0"] = features
        for hop in range(1, args.R + 1):
            g.update_all(
                fn.u_mul_e(f"feat_{hop-1}", "weight", "msg"),
                fn.sum("msg", f"feat_{hop}"),
            )
        res = []
        for hop in range(args.R + 1):
            res.append(g.ndata.pop(f"feat_{hop}"))
        return res


def prepare_data(device, args):
    data = load_dataset(args.dataset)
    g, n_classes, train_nid, val_nid, test_nid = data
    g = g.to(device)
    in_feats = g.ndata["feat"].shape[1]
    feats = preprocess(g, g.ndata["feat"], args)
    labels = g.ndata["label"]
    # move to device
    train_nid = train_nid.to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    train_feats = [x[train_nid] for x in feats]
    train_labels = labels[train_nid]
    return (
        feats,
        labels,
        train_feats,
        train_labels,
        in_feats,
        n_classes,
        train_nid,
        val_nid,
        test_nid,
    )

def decision_acc(preds, labels, num_classes):
    avg_pred = preds.sum(dim=0) / preds.shape[0]
    mask = class_pairs_mask(num_classes)
    decision_output = torch.matmul(avg_pred, mask)
    decision_labels = torch.tensor(label_pairs_count(labels, num_classes)).to(decision_output.device).float()
    pred = (decision_output > 0).float()
    correct = (pred == decision_labels).float()
    return correct.mean()

def evaluate(epoch, args, model, feats, labels, train, val, test, device, eval_decision=False, num_classes=None):
    with torch.no_grad():
        batch_size = args.eval_batch_size
        if batch_size <= 0:
            pred = model(feats)
        else:
            pred = []
            num_nodes = labels.shape[0]
            n_batch = (num_nodes + batch_size - 1) // batch_size
            for i in range(n_batch):

                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, num_nodes)
                batch_feats = [feat[batch_start:batch_end] for feat in feats]
                #[hm] move data to GPU eval time
                batch_feats = [t.to(device) for t in batch_feats]
                pred.append(model(batch_feats).cpu())
            pred = torch.cat(pred)

        pred_class = torch.argmax(pred, dim=1)
        correct = (pred_class == labels).float()
        train_acc = correct[train].sum() / len(train)
        val_acc = correct[val].sum() / len(val)
        test_acc = correct[test].sum() / len(test)

        # evaluate decision problem
        if eval_decision:
            assert num_classes is not None
            train_decision_acc = decision_acc(pred[train], labels[train], labels.max().item() + 1)
            val_decision_acc = decision_acc(pred[val], labels[val], labels.max().item() + 1)
            test_decision_acc = decision_acc(pred[test], labels[test], labels.max().item() + 1)
            return train_acc, val_acc, test_acc, train_decision_acc, val_decision_acc, test_decision_acc

        return train_acc, val_acc, test_acc


def main(args, use_wandb=False):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    data = prepare_data("cpu", args)
    (
        feats,
        labels,
        train_feats,
        train_labels,
        in_size,
        num_classes,
        train_nid,
        val_nid,
        test_nid,
    ) = data
    
    if args.eval_decision:
        print("data class statistics:")
        print("Train:", np.bincount(labels[train_nid], minlength=num_classes))
        print("Eval: ", np.bincount(labels[val_nid], minlength=num_classes))
        print("Test: ", np.bincount(labels[test_nid], minlength=num_classes))
              
    if args.model == 'qmodel':
        model = QModel(
            in_size,
            args.num_hidden,
            num_classes,
            args.R,
            args.dropout,
            args.activation,
            args.input_dropout,
            poly_act_deg=args.poly_act_deg
        )
    else:
        model = Model(
            in_size,
            args.num_hidden,
            num_classes,
            args.R,
            args.ff_layer,
            args.dropout,
        )
    print("model:\n", model)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_epoch = 0
    best_val = 0
    best_test = 0
    best_epoch_results = None

    best_val_decision = 0
    best_test_decision = 0


    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        model.train()

        train_feats = [t.to(device) for t  in train_feats]
        train_labels = train_labels.to(device)
        if args.train_batch_size is not None:
            optimizer.zero_grad()
            # split train data to batches
            num_nodes = len(train_nid)
            n_batch = (num_nodes + args.train_batch_size - 1) // args.train_batch_size
            for i in range(n_batch):
                batch_start = i * args.train_batch_size
                batch_end = min((i + 1) * args.train_batch_size, num_nodes)
                batch_feats = [feat[batch_start:batch_end] for feat in train_feats]
                batch_labels = train_labels[batch_start:batch_end]
                loss = loss_fcn(model(batch_feats), batch_labels)
                loss.backward()
            
            optimizer.step()

        else:
            loss = loss_fcn(model(train_feats), train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.eval_every == 0 or epoch <= args.eval_first_epochs:
            model.eval()
            acc = evaluate(
                epoch, args, model, feats, labels, train_nid, val_nid, test_nid, device, eval_decision=args.eval_decision, num_classes=num_classes
            )
            end = time.time()
            log = "Epoch {}, Times(s): {:.2f}".format(epoch, end - start)
            log += ", Loss: {:.4f}".format(loss.item())
            log += ", Accuracy: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(
                *acc
            )
            if args.eval_decision:
                log += ", Decision Accuracy: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(
                    acc[3], acc[4], acc[5]
                )
            print(log)
            log_dict = {
                        "train_loss": loss.item(),
                        "train_acc": acc[0],
                        "val_acc": acc[1],
                        "test_acc": acc[2],
                        "epoch": epoch,
                    }
            if args.eval_decision:
                log_dict.update({
                    "train_decision_acc": acc[3],
                    "val_decision_acc": acc[4],
                    "test_decision_acc": acc[5],
                })
            if acc[1] > best_val:
                best_val = acc[1]
                best_epoch = epoch
                best_test = acc[2]
                best_epoch_results = log_dict
                if args.out_dir is not None:
                    torch.save(model.state_dict(), os.path.join(args.out_dir, f"best_model.pt"))
            
            if acc[4] > best_val_decision:
                best_val_decision = acc[4]
                best_test_decision = acc[5]
                if args.out_dir is not None:
                    torch.save(model.state_dict(), os.path.join(args.out_dir, f"best_model_decision.pt"))
            
            if use_wandb:
                
                wandb.log(log_dict)

    print(
        "Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
            best_epoch, best_val, best_test
        )
    )
    print("Best epoch: ", best_epoch_results)
    if use_wandb:
        wandb.log({"final_val_acc": best_val, "final_test_acc": best_test, "final_val_decision_acc": best_val_decision, "final_test_decision_acc": best_test_decision})


def get_arg_parser(default_args):
    parser = argparse.ArgumentParser(description="SIGN")
    parser.add_argument("--num-epochs", type=int, default=default_args.num_epochs)
    parser.add_argument("--num-hidden", type=int, default=default_args.num_hidden)
    parser.add_argument("--R", type=int, default=default_args.R, help="number of hops")
    parser.add_argument("--lr", type=float, default=default_args.lr)
    parser.add_argument("--dataset", type=str, default=default_args.dataset)
    parser.add_argument("--dropout", type=float, default=default_args.dropout)
    parser.add_argument("--gpu", type=int, default=default_args.gpu)
    parser.add_argument("--weight-decay", type=float, default=default_args.weight_decay)
    parser.add_argument("--eval-every", type=int, default=default_args.eval_every)
    parser.add_argument("--eval-first-epochs", type=int, default=default_args.eval_first_epochs, help='Evaluate decision problem after this number of epochs')
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=default_args.eval_batch_size,
        help="evaluation batch size, -1 for full batch",
    )
    parser.add_argument(
        "--ff-layer", type=int, default=default_args.ff_layer, help="number of feed-forward layers"
    )
    parser.add_argument('--model', default=default_args.model, help='model to use (qmodel, model)')
    parser.add_argument('--activation', default=default_args.activation, help='activation to use (prelu, identity)')
    
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=default_args.train_batch_size,
        help="training batch size, -1 for full batch",
    )
    parser.add_argument(
        "--input-dropout",
        type=float,
        default=default_args.input_dropout,
        help="dropout on input features",
    )

    parser.add_argument('--wandb-project', default=default_args.wandb_project, type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb-group', default=default_args.wandb_group, type=str, help='Group name for W&B')
    parser.add_argument('--wandb-name', default=default_args.wandb_name, type=str, help='Run name for W&B')
    parser.add_argument('--wandb-sweep', type=bool, default=default_args.wandb_sweep, help='Use wandb sweep. To use sweep this must be set True in the default args')
    parser.add_argument('--seed', default=default_args.seed, type=int, help='Random seed for reproducibility')
    parser.add_argument('--eval-decision', type=bool, default=default_args.eval_decision, help='Evaluate decision problem')
    parser.add_argument('--poly-act-deg', type=int, default=default_args.poly_act_deg, help='Degree of the polynomial activation')
    parser.add_argument('--out-dir', type=str, default=default_args.out_dir, help='Output directory to save the best model')
    return parser


if __name__ == "__main__":
    args = get_arg_parser(default_args=default_config).parse_args()
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
        
        wandb.run.name = f"{args.model}_r{args.R}_h{args.num_hidden}_{print_act}_\
            dp{args.dropout}_indp{args.input_dropout}_lr{args.lr}_wd{args.weight_decay}_seed{args.seed}_{random_str}"
        
    print(args)
    set_random_seed(args.seed)        
    print("use wandb: ", use_wandb)
    print("args: ", args)
    
    main(args, use_wandb)
