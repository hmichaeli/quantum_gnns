
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolyAct(torch.nn.Module):
    '''
    Source: https://github.com/hmichaeli/alias_free_convnets/blob/5fe2dd28d64d366d2ec2e0f2e07c998a52f8efaa/models/activation.py#L61
    '''
    def __init__(self, trainable=False, init_coef=None, in_scale=1,
                 out_scale=1,
                 train_scale=False,
                 deg=2):
        super(PolyAct, self).__init__()
        if init_coef is None:
            init_coef = [0.0169394634313126, 0.5, 0.3078363963999393]
        
        # self.deg = len(init_coef) - 1
        self.deg = deg
        if deg > len(init_coef) - 1:
            init_coef = init_coef + [0.0] * (deg - len(init_coef) + 1)
        elif deg < len(init_coef) - 1:
            init_coef = init_coef[:deg + 1]

        self.trainable = trainable
        coef = torch.Tensor(init_coef)
        
        if trainable:
            self.coef = nn.Parameter(coef, requires_grad=True)
        else:
            self.register_buffer('coef', coef)

        if train_scale:
            self.in_scale = nn.Parameter(torch.tensor([in_scale * 1.0]), requires_grad=True)
            self.out_scale = nn.Parameter(torch.tensor([out_scale * 1.0]), requires_grad=True)

        else:
            if in_scale != 1:
                self.register_buffer('in_scale', torch.tensor([in_scale * 1.0]))
            else:
                self.in_scale = None

            if out_scale != 1:
                self.register_buffer('out_scale', torch.tensor([out_scale * 1.0]))
            else:
                self.out_scale = None

        
    def forward(self, x):
        if self.in_scale is not None:
            x = self.in_scale * x

        x = self.calc_polynomial(x)

        if self.out_scale is not None:
            x = self.out_scale * x

        return x

    def __repr__(self):
        print_coef = self.coef.cpu().detach().numpy()
        return "PolyAct(trainable={}, coef={})".format(
            self.trainable, print_coef)

    def calc_polynomial(self, x):
        res = self.coef[0] + self.coef[1] * x
        for i in range(2, self.deg + 1):
            res = res + self.coef[i] * (x ** i)

        return res



def label_pairs_count_iterative(labels, C):
    # Count the instances of each class
    class_counts = np.zeros(C, dtype=int)
    for label in labels:
        class_counts[label] += 1
    
    # Create the pairwise comparison vector
    pairwise_vector = []
    for i in range(C):
        for j in range(i+1, C):
            pairwise_vector.append(1 if class_counts[i] > class_counts[j] else 0)
    
    return np.array(pairwise_vector)

# fast vectorized implementation
def label_pairs_count(labels, C):
    # Count the instances of each class
    class_counts = np.bincount(labels, minlength=C)
    
    # Create indices for pairwise comparisons
    indices = np.triu_indices(C, k=1)
    
    # Create the pairwise comparison vector
    pairwise_vector = (class_counts[indices[0]] > class_counts[indices[1]]).astype(int)
    
    return pairwise_vector


def test_label_pairs_count():
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 3])
    C = 4
    pairwise_vector = label_pairs_count_iterative(labels, C)
    print( pairwise_vector)

    pairwise_vector = label_pairs_count(labels, C)
    print(pairwise_vector)



# helpers for decision accuracy
def class_pairs_mask(input_dim):
    num_pairs = input_dim * (input_dim - 1) // 2
    mask = torch.zeros((input_dim, num_pairs), dtype=torch.float32)
    col = 0
    for i in range(input_dim):
        for j in range(i+1, input_dim):
            mask[i, col] = 1
            mask[j, col] = -1
            col += 1
    return mask

