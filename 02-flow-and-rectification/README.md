# 02. [Flow and Rectified Flow](./02-flow-and-rectification.ipynb)

Subproject notes only. See the main README for full project context.

## Files

- [02-flow-and-rectification.ipynb](./02-flow-and-rectification.ipynb): Stand-alone notebook (Colab-friendly)
- [FFJORD.py](./FFJORD.py): FFJORD Model for initial Flow Model
- [flow_train.py](./flow_train.py): FFJORD training + Reflow Training
- [swissroll.py](./swissroll.py): Swiss Roll Dataset Generator
- [utils.py](./utils.py): Plotting Flow ODE trajectories


## Requirements

- torch
- torchdiffeq
- numpy
- matplotlib
- sklearn
- tqdm