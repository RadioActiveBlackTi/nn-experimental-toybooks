# 05. [Point Cloud Autoencoder](./05-point-cloud-autoencoder.ipynb)

Subproject notes only. See the main README for full project context.

## Files

- [05-point-cloud-autoencoder.ipynb](./05-point-cloud-autoencoder.ipynb): Stand-alone notebook (Colab-friendly)
- [snowman.py](./snowman.py): Three types of ellipsoidal snowmans Generator
- [PointNet.py](./PointNet.py): T-Net and PointNet Feature Extractor
- [encoders.py](./encoders.py): Deterministic Encoder, Variational Encoder
- [decoders.py](./decoders.py): Simple MLP Decoder, PointGMM Decoder
- [autoencoders.py](./autoencoders.py): PointNet AE, PointNet VAE, PointGMM VAE
- [utils.py](./utils.py): Latent Visualizations, Reconstructions, GMM Visualizations, GMM Clustering


## Requirements

- torch
- numpy
- matplotlib
- tqdm