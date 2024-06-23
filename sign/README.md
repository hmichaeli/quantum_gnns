# Shallow Polynomial Graph Neural Netowrks



## Environment setup

Requirements:
CUDA 12.1
Python 3.10

```
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html
pip install pandas
pip install pyyaml
pip install pydantic
pip install ogb
 
```

## Results
### Node Classification

```bash
# Ogbn-producs
python sign_node_pred.py --dataset amazon --R 5 --num-hidden 512 --dr 0.4 --lr 0.001 --input-dropout 0.3 --activation poly --seed 0
# Reddit
python sign_node_pred.py --dataset amazon --R 5 --num-hidden 512 --dr 0.4 --lr 0.001 --input-dropout 0.3 --activation poly --seed 0
# Cora
python sign_node_pred.py --dataset amazon --R 5 --num-hidden 512 --dr 0.4 --lr 0.001 --input-dropout 0.3 --activation poly --seed 0
```
Results (10 seeds statistics):

| Dataset       | Test Accuracy        | 
|---------------|----------------------|
| Ogbn-products | 78.51 $\pm$ 0.05     | 
| Reddit        | 96.31 $\pm$ 0.03     | 
| Cora          | 78.69 $\pm$ 0.26     | 

### Graph Classification
```bash
# MUTAG
python sign_graph_pred.py --dataset MUTAG --model qmodel_dense --activation poly --num-hidden 64 --norm batch \
--sign-gcn-hops 6 --sign-ppr-hops 5 --sign-raw-hops 1 --sign-rw-hops 8 \
--dropout 0 --input-dropout 0 --batch-size 128 --lr 0.005 --weight-decay 0 \
--seed 2 --wandb-project default_project --device 2

# PTC         
python sign_graph_pred.py --dataset PTC --model qmodel_dense --activation poly --num-hidden 96 --norm batch \
--sign-gcn-hops 10 --sign-ppr-hops 3 --sign-raw-hops 0 --sign-rw-hops 2 \
--dropout 0 --input-dropout 0 --batch-size 32 --lr 0.003 --weight-decay 0.0000001 \
--seed 0

# NCI1     
python sign_graph_pred.py --dataset NCI1 --model qmodel_dense --activation poly --num-hidden 12 --norm layer \
--sign-gcn-hops 10 --sign-ppr-hops 2 --sign-raw-hops 2 --sign-rw-hops 7 \
--dropout 0.1 --input-dropout 0 --batch-size 64 --lr 0.03 --weight-decay 0.0001 \
--seed 0    

# PROTEINS    
python sign_graph_pred.py --dataset PROTEINS --model qmodel_dense --activation poly --num-hidden 128 --norm batch \
--sign-gcn-hops 9 --sign-ppr-hops 8 --sign-raw-hops 0 --sign-rw-hops 6 \
--dropout 0.1 --input-dropout 0 --batch-size 32 --lr 0.001 --weight-decay 0.0001 \
--seed 0

# COLLAB        
python sign_graph_pred.py --dataset COLLAB --model qmodel_dense --activation poly --num-hidden 148 --norm batch \
--sign-gcn-hops 8 --sign-ppr-hops 1 --sign-raw-hops 0 --sign-rw-hops 4 \
--dropout 0.1 --input-dropout 0 --batch-size 512 --lr 0.05 --weight-decay 0.00000001 \
--degrees-as-nlabel \
--seed 0

# IMDB-B       
python sign_graph_pred.py --dataset IMDBBINARY --model qmodel_dense --activation poly --num-hidden 12 --norm layer \
--sign-gcn-hops 7 --sign-ppr-hops 10 --sign-raw-hops 1 --sign-rw-hops 7 \
--dropout 0.3 --input-dropout 0 --batch-size 32 --lr 0.01 --weight-decay 0.0001 \
--degrees-as-nlabel \
--seed 0

# IMDB-M
python sign_graph_pred.py --dataset IMDBMULTI --model qmodel_dense --activation poly --num-hidden 8 --norm batch \
--sign-gcn-hops 9 --sign-ppr-hops 7 --sign-raw-hops 2 --sign-rw-hops 8 \
--dropout 0.3 --input-dropout 0 --batch-size 32 --lr 0.003 --weight-decay 0.0000001 \
--degrees-as-nlabel \
--seed 0 

# REDDIT-B     
python sign_graph_pred.py --dataset REDDITBINARY --model qmodel_dense --activation poly --num-hidden 8 --norm batch \
--sign-gcn-hops 2 --sign-ppr-hops 9 --sign-raw-hops 0 --sign-rw-hops 9 \
--dropout 0 --input-dropout 0 --batch-size 64 --lr 0.1 --weight-decay 0 \
--seed 0

# REDDIT-M  
python sign_graph_pred.py --dataset REDDITMULTI5K --model qmodel_dense --activation poly --num-hidden 32 --norm layer \
--sign-gcn-hops 10 --sign-ppr-hops 1 --sign-raw-hops 2 --sign-rw-hops 10 \
--dropout 0 --input-dropout 0 --batch-size 1024 --lr 0.003 --weight-decay 0.00001 \
--seed 0

```
| Dataset       | Test Accuracy        | 
|---------------|----------------------|
| MUTAG         | 92.0  $\pm$ 6.5      | 
| PTC           | 68.0  $\pm$ 8.1      | 
| NCI1          | 77.2 $\pm$ 1.4       | 
| PROTEINS      | 76.7 $\pm$ 4.6       | 
| COLLAB        | 81.8 $\pm$ 1.4       |
| IMDB-B        | 76.0 $\pm$ 2.5       |
| IMDB-M        | 53.1 $\pm$ 2.8       |
| REDDIT-B      | 78.9 $\pm$ 2.7       |
| REDDIT-M      | 54.1 $\pm$ 1.7       |



## Acknowledgement
* [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/pdf/2004.11198.pdf)
* DGL Examples (commit: 15d05be)
  - https://github.com/dmlc/dgl/tree/master/examples/pytorch/sign 
  - https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/


