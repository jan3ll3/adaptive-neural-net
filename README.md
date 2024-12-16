# Adaptive Neural Network with Dynamic Topology

This repository implements an adaptive neural network architecture capable of dynamically adjusting its topology (e.g., neurons, layers, attention heads) during training. It supports both basic neural network architectures and transformer-based language models, including selective pruning, weight quantization, and adaptive growth of network components.

## Key Features

- **Dynamic Topology**: Networks can grow or shrink based on training conditions:
  - **Neural MLPs**: Add or remove neurons and layers.
  - **Transformers**: Prune or reintroduce attention heads, consider layer size growth.
- **Adaptive Attention (Transformers)**: Heads are pruned based on importance metrics and can be masked out or reactivated.
- **Weight Quantization**: Periodic quantization reduces memory footprint and may improve inference speed.
- **Pruning**: Inactive neurons, attention heads, and low-magnitude connections are removed to streamline the model.
- **Visualization Tools**: Visualize network structure, learned patterns, latent spaces (for VAE), and attention distributions.

## Repository Structure

- **Core Neural Network** (`neural_network.py`):  
  Implements an adaptive feedforward network (e.g., for parity or MNIST).  
  Features:
  - Custom initialization inspired by XOR detection for parity tasks.
  - Dynamic neuron growth triggered by activation patterns.
  - Pruning of low-importance weights and neurons.
  - Optional weight quantization to reduce precision.
  - Support for skip connections.
  
- **Language Model** (`neural_network_LLM.py`):  
  Implements a transformer-based language model with adaptive capabilities.  
  Features:
  - Multi-head self-attention with pruning of less important heads.
  - Growth considerations for hidden states if high variance is detected.
  - Integrates positional embeddings and feed-forward networks with dynamic capacity.
  - Compatible with GPT-2 style tokenizers.
  
- **VAE for MNIST** (`vae.py`):  
  A Variational Autoencoder with adaptive channel and latent dimension adjustments.  
  Features:
  - Convolutional layers that can grow or prune channels based on activation patterns.
  - Adaptive latent space dimensionality and structure.
  - Quantization and pruning to maintain efficiency.
  - Visualization of generated digits, latent space interpolation, and channel activity.

- **Examples and Training Scripts**:
  - **MNIST** (`mnist_example.py`):  
    Simple digit classification showcasing adaptive network growth and pruning.
    
  - **Parity** (`parity_example.py`):  
    N-bit parity problem with XOR-inspired initialization to demonstrate the adaptive MLP.
    
  - **Language Model Training** (`train_llm.py`):  
    Train a transformer-based language model on a text dataset (e.g., BabyLM). Integrates adaptive attention and layer growth.
    
  - **VAE Training**:  
    Run `python vae.py` to train the adaptive VAE on MNIST and visualize outputs.
  
- **Visualization Tools**:
  - **Network Topology** (`visualize_model.py`):  
    Visualize the network’s layer structure and pruning/growth events.
    
  - **MNIST Pattern Visualization** (`visualize_mnist.py`):  
    Display learned features and outputs from the MLP or VAE.
    
  - **VAE Latent Space Visualization** (within `vae.py`):  
    Generate latent space plots, sample generations, and interpolations.

- **Interactive Interface** (`chat.py`):
  - Provides a command-line chat interface to interact with trained language models.
  - Adjustable generation parameters such as temperature, top-k, and nucleus sampling.

## Installation

```bash
# Clone the repository
git clone https://github.com/jan3ll3/adaptiveNN.git
cd adaptiveNN

# Install dependencies
pip install torch torchvision transformers rich wandb
```

## Quick Start

### Training a Basic Network

Train an adaptive MLP on MNIST or parity tasks:

```bash
# Train on MNIST
python mnist_example.py

# Train on an N-bit parity problem
python parity_example.py
```

### Training a Language Model

```bash
python train_llm.py --data_dir path/to/data \
    --hidden_size 256 \
    --num_layers 6 \
    --num_heads 8 \
    --batch_size 2 \
    --max_seq_length 512
```

### Using the Chat Interface

After training a language model:

```bash
python chat.py --model_path outputs/best_model.pt
```

Interactively generate text and adjust parameters such as temperature.

## Adaptive Mechanisms

### Growth and Pruning

**Neural MLPs:**

- **Neuron Growth:**  
  If layer activations are consistently saturated (e.g., >90% are active), the network adds new neurons to handle complexity.
  
- **Weight and Neuron Pruning:**  
  Periodically remove weights with magnitudes below a threshold. Inactive neurons (no strong connections) are pruned to reduce overparameterization.

**Transformers:**

- **Attention Head Pruning:**  
  Heads are pruned if their importance (based on weight norms) falls below a certain ratio of the maximum. Pruned heads are masked out, reducing computation and focusing on critical information paths.

- **Potential Layer Growth:**  
  If activation variance is high and layer sizes are under their maximum limit, hidden dimensions or intermediate projections may grow. This is currently a stubbed feature in the example code and can be extended.

**VAE:**

- **Channel Growth/Pruning:**  
  Convolutional channels grow if variance is high, and prune if some channels remain inactive. Latent space dimensions can also adapt to complexity.

### Quantization

- **Weight Quantization:**  
  Every few steps, weights are quantized to reduce memory usage. This can act as a form of regularization and improve inference speed.

### Stability Controls

- **Growth Cooldown:**  
  After growing, a cooldown period prevents immediate subsequent expansions, allowing the network to stabilize.

- **Activation Statistics:**  
  Exponential moving averages of mean and variance guide decisions on when to grow layers or prune heads.

### Example Schedules

- **Weight Quantization:** Every 100 steps.
- **Head Pruning (Transformers):** Every 500 steps.
- **Growth Checks:** Every 1000 steps, if enough samples have been processed.

These schedules can be customized in the training scripts.

## Training Parameters

**Language Model (Default):**
- Learning Rate: 3e-4
- Batch Size: 2-4
- Sequence Length: 512
- Hidden Size: 256
- Number of Layers: 6
- Attention Heads: 8

**Optimization:**
- AdamW optimizer
- Cosine LR schedule
- Gradient clipping for stability
- Periodic pruning and quantization

**VAE (Default):**
- Latent dimension: 16
- Initial channels: [32, 64, 64]
- Growth threshold: 0.5
- Prune threshold: 0.05
- Batch size: 128
- Learning rate: 1e-3

## Visualization

- **Topology Visualization:**  
  Inspect how layers and heads change over time.
  
- **Attention Patterns (Transformers):**  
  Analyze which heads survive pruning and how attention distributions evolve.
  
- **VAE Latent Space:**  
  Visualize digit clusters, interpolations, and emerging structures as channels and latent dimensions adapt.

## Chat Interface Features

- Interactive prompt for language model queries.
- Adjustable generation parameters (`temp=X` for temperature, `help` for commands).
- Allows on-the-fly experimentation with pruning and quantization settings if integrated.

## Acknowledgments

- PyTorch team for the foundational deep learning framework.
- Hugging Face team for the Transformers library.
- The `rich` library for improved console output.
- The BabyLM project for providing training data for the language model.

