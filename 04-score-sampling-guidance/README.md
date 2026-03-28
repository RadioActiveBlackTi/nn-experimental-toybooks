# 04. [Score Matching: Sampling and Guidance](./04-score-sampling-guidance.ipynb)

Subproject notes only. See the main README for full project context.

## Files

- [04-score-sampling-guidance.ipynb](./04-score-sampling-guidance.ipynb): Stand-alone notebook (Colab-friendly)
- [n_halfmoons.py](./n_halfmoons.py): N Halfmoons Generator
- [score_net.py](./score_net.py): NCSN model and training
- [schedulers.py](./schedulers.py): sigma schedulers
- [samplers.py](./samplers.py): NCSN sampler, DDPM sampler, DDIM sampler
- [classifer_guidance.py](./classifier_guidance.py): Classifier for guidance + Guided Score by Classifier
- [cfg.py](./cfg.py): Classifier-Free Guidance + Guided Score by CFG

## Requirements

- torch
- numpy
- matplotlib
- sklearn
- tqdm