# 01. [GCN and Oversmoothing](./01-gcn-oversmoothings.ipynb)

Subproject notes only. See the main README for full project context.

## Files

- [01-gcn-oversmoothings.ipynb](./01-gcn-oversmoothings.ipynb): Stand-alone notebook (Colab-friendly)
- [SimpleGCN.py](./SimpleGCN.py): Basic GCN baseline
- [NormalizedGCN.py](./NormalizedGCN.py): GCN with graph normalization + residual connection
- [BSGCN.py](./BSGCN.py): Blurring-Sharpening GCN variant
- [cora_load.py](./cora_load.py): CORA data loader
- [analysis.py](./analysis.py): Dirichlet energy / singular value analysis
- [gcn_train.py](./gcn_train.py): Training template

## Requirements

- torch
- torch_geometric
- numpy
- matplotlib