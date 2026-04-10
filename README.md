# Low-pass Personalized Subgraph Federated Recommendation (LPSFed)

## **ICLR 2026** | [Paper](https://openreview.net/forum?id=SSd3GENRAU)

This repository contains the implementation of **LPSFed**, a robust personalized federated recommender system that addresses subgraph structural imbalance through low-pass spectral filtering and localized popularity bias-aware optimization.

## Overview

LPSFed leverages graph Fourier transforms and low-pass spectral filtering to extract stable structural signals across heterogeneous subgraphs, enabling robust personalized parameter updates. The method incorporates a localized popularity bias-aware margin to mitigate recommendation bias in federated settings.

### Key Features

- **Low-pass Spectral Filtering**: Extracts denoised structural signals from client subgraphs
- **Adaptive Margin**: Personalized bias correction based on local popularity patterns
- **Structural Similarity**: KL-divergence based similarity for federated aggregation
- **BC-Loss**: Bias-aware Contrastive Loss combining popularity and preference learning

## Requirements

- Python 3.10.13
- CUDA 12.4 (or compatible version)
- At least 10GB GPU Memory
- PyTorch 2.1.0+

## Installation

### 1. Environment Setup

Create a conda environment and install dependencies:

```bash
# Create conda environment
conda create -n LPSFed python=3.10
conda activate LPSFed

# Install required packages
pip install -r requirements.txt
```

### 2. Dataset Preparation

Place your dataset files in the `data/` directory. Supported datasets:
- Amazon-Book
- Gowalla
- MovieLens-1M
- Yelp2018
- Tmall-Buy

## Usage

### Step 1: Data Preprocessing - Create Subgraphs

First, partition your dataset into subgraphs for federated learning:

```bash
cd src/utils
python make_subgraph.py
```

This script will:
- Partition the global graph into client-specific subgraphs using spectral clustering
- Save subgraph files (`subgraphs{i}.gml`) and degree information (`degrees_{i}.json`) for each client
- Generate a random reference graph for structural similarity computation

**Configuration**: Edit `make_subgraph.py` to set:
- `NUM_CLIENTS`: Number of client subgraphs (default: 4)
- `DATASET`: Dataset name
- `RANDOMGRAPH_TYPE`: Type of random graph ('random' or 'gnmk')

### Step 2: Eigen-decomposition Preprocessing

Compute graph embeddings using eigen-decomposition:

```bash
cd src/utils
python subgraph_embeddings_gpu.py
```

This script will:
- Compute Laplacian matrices for each client subgraph
- Perform eigenvalue decomposition to extract spectral features
- Save graph embeddings (`graph_embeddings_1d{i}.json`) and eigenvalues (`eigenvalues_{i}.json`) for each client
- Generate embeddings for the random reference graph

**Configuration**: Edit `subgraph_embeddings_gpu.py` to set:
- `DATASET`: Dataset index (0: Amazon, 1: Gowalla, 2: ML-1M, 3: Yelp2018, 4: Tmall)
- `FREQUENCY`: Dimensionality of spectral embeddings (default: 64)
- `NUM_CLIENTS`: Number of clients
- `GRAPH_CONV`: Graph convolution type ('1d' or '2d')
- `RANDOMGRAPH_TYPE`: Random graph type ('random' or 'gnmk')

**Note**: This step requires GPU for efficient computation. The script uses scipy's `eigsh` for sparse eigenvalue decomposition.

### Step 3: Parameter Configuration

Configure experiment parameters in `src/utils/params.py`:

```python
# Dataset selection
dataset = 2  # 0:Amazon, 1:Gowalla, 2:ML-1M, 3:Yelp, 4:Tmall

# Model configuration
model = 1  # 0:MF, 1:LGCN
pred_dim = 128  # Embedding dimensionality
LAYER = 2  # Number of GCN layers

# Federated learning settings
NUM_CLIENTS = 4
FED_METHOD = 'lpsfed'  # 'split', 'fedavg', or 'lpsfed'
GLOBAL_UPDATE_EPOCH = 15  # Global aggregation frequency

# BC-Loss parameters
tau1 = 0.07  # Temperature for main contrastive loss
tau2 = 0.08  # Temperature for bias contrastive loss
w_lambda = 0.5  # Weight for bias vs preference learning
margin_ratio = 0.5  # Margin interpolation ratio
gamma = 1.0  # Margin strength (Eq. 8)
omega = 0.5  # Refined margin weight (Eq. 15)

# Model architecture
POOLING = 'MLP3'  # Pooling strategy: 'Concat', 'Sum', 'Max', 'Product', 'MLP3'
PREDICTION = 'MLP3'  # Prediction method: 'InnerProduct' or 'MLP3'
LOSS_FUNCTION = 'BC'  # Loss function: 'BPR', 'CrossEntropy', 'MSE', 'BC'
```

### Step 4: Run Experiments

Execute the main training script:

```bash
cd src
python main.py
```

The script will:
- Load preprocessed subgraphs and graph embeddings
- Train client models with LPSFed framework
- Perform federated aggregation based on structural similarity
- Evaluate on test sets and save results

**Output**: Results are saved in `src/experiment_result/` directory.

## Project Structure

```
LPSFed/
├── data/                    # Dataset directory
│   └── {DATASET}/
│       └── {NUM_CLIENTS}_clients/
│           ├── subgraphs{i}.gml          # Client subgraphs
│           ├── graph_embeddings_1d{i}.json  # Spectral embeddings
│           ├── eigenvalues_{i}.json      # Eigenvalues
│           ├── degrees_{i}.json         # Node degrees
│           └── train/test/val_data{i}.json
├── src/
│   ├── main.py             # Main entry point
│   ├── train_model.py      # Training loop
│   ├── test.py             # Evaluation
│   ├── model_LPSFed.py     # LPSFed model implementation
│   └── utils/
│       ├── params.py                    # Parameter configuration
│       ├── make_subgraph.py             # Subgraph generation
│       ├── subgraph_embeddings_gpu.py   # Eigen-decomposition
│       ├── fed_data_preprocessing.py    # Data preprocessing
│       └── comparison_eigenvalues.py   # Structural similarity
└── requirements.txt        # Python dependencies
```

## Key Components

### 1. Low-pass Spectral Filtering (Eq. 6)
Extracts denoised structural signals by retaining only the first Φ low-frequency components:
```
Kc = Λc ⊙ f̃1:Φ
```

### 2. Adaptive Margin (Eq. 8)
Computes personalized margin based on popularity bias:
```
Mc_ui = min {γ · ξ̂_ui, π - R̂_ui}
```

### 3. BC-Loss (Eq. 9)
Bias-aware Contrastive Loss combining popularity and preference learning:
```
L_BC = -Σ log(exp(cos(R̂_ui + fMc_ui)/τ) / (exp(cos(R̂_ui + fMc_ui)/τ) + Σ exp(cos(R̂_uj)/τ)))
```

### 4. Structural Similarity (Eq. 11-12)
KL-divergence based similarity for personalized federated aggregation:
```
ρc = DKL(KR ∥ Kc)
ρ̄c = 1 - (ρc - min(ρ)) / (max(ρ) - min(ρ))
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{sim2026lpsfed,
  title={Low-pass Personalized Subgraph Federated Recommendation},
  author={Sim, Wooseok and Park, Hogun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please contact:
- Wooseok Sim: dntjr41@skku.edu

## Acknowledgments

This work was supported by Sungkyunkwan University.
