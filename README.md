<div align=center>
  <h1>
  nn-experimental-toybooks
  </h1>

</div>

## 🔎 About this repo
This repository is a collection of my toy experiments for various deep learning topics. Each idea is within each one Jupyter notebook.

## 🧪 Experiment Lists

### 01. [GCN and Oversmoothings](./01-gcn-oversmoothings.ipynb)

In general, it is know that GCN cannot be stacked deeply since neighborhood aggregation works as a low-pass filter. On the other hand, Transformers can be stacked deeply although self-attention is equivalent to GCN on complete graph; thanks to normalizations and residual connections.

In this notebook `01-gcn-oversmoothings.ipynb`, I worked on CORA for inspecting oversmoothing, with testing normalization and residual connections, along with sharpening operations.

<h3>Reference works</h3>

**[DeepGCNs: Residual Connections for GNN](https://arxiv.org/abs/1904.03751)**

**[GREAD: Reaction-Diffusion based GNN](https://arxiv.org/abs/2211.14208)**

---


### 02. [Flow Model and Rectified Flow](./02-flow-and-rectification.ipynb)

Although flow-based model's concept is very nice and intuitive, it's early phase's obstacle was calculation of Jacobians for calculating log likelihood. FFJORD solved network constraints by applying Hutchinson estimator for Jacobian Trace calculation. Further, it is also found that the intuitive approach that connects start point to end point works well, even leading reasonable one-step generation.

<div align=center>
   <img src="./assets/02/ffjord.gif" width="200">
   <img src="./assets/02/reflow.gif" width="200">
   <img src="./assets/02/reflow2.gif" width="200">
</div>

In this notebook `02-flow-and-rectification.ipynb`, first worked on gaussian-to-swiss-roll generation using FFJORD, and applied reflow algorithms for the flow model.

<h3>Reference works</h3>

**[FFJORD: Free Form CNF model](https://arxiv.org/abs/1810.01367)**

**[Rectified Flow: Flow Straight and Fast](https://arxiv.org/abs/2209.03003)**

---

### 03. [Meta Optimization](./03-meta-optimization.ipynb)

Meta-learning is a methodology for finding versatile initialization point that can quickly adapt to any few-shot task only with a few gradient steps. While MAML suggested fundamental and model-agnostic way as its name, it has crucial overhead of calculating second-order Hessian. So in Reptile, rather than using gradient descent for inner loop gradient steps, it updates the initialization simply towards the fine-tuned weights.

<div align=center>
    <img src="./assets/03/base.gif" width="200">
    <img src="./assets/03/maml.gif" width="200">
    <img src="./assets/03/reptile.gif" width="200">
</div>

In this notebook `03-meta-optimization.ipynb`, I worked for simple sine regression tasks for baseline pre-trained MLP, MAML, and Reptile.

<h3>Reference works</h3>

**[MAML: Model-Agnostic Meta Learning](https://arxiv.org/abs/1703.03400)**

**[Reptile: First-Order Fast approximation of MAML](https://arxiv.org/abs/1803.02999)**

---

### 04. [Score Matching: Sampling and Guidance]()

TBD