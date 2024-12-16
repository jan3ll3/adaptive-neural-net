# Adaptive Neural Network with Dynamic Topology

This repository implements an adaptive neural network architecture that can dynamically adjust its topology during training. It includes implementations for both basic neural networks and transformer-based language models with adaptive capabilities.

## Features

- **Dynamic Topology**: Networks can grow or shrink based on training needs
- **Adaptive Attention**: Transformer models with prunable attention heads
- **Quantization**: Weight quantization for reduced memory footprint
- **Pruning**: Automatic pruning of inactive neurons and connections
- **Visualization**: Tools for visualizing network structure and learned patterns

## Components

### Core Neural Network (`neural_network.py`)
- Base implementation of adaptive neural network
- Support for skip connections
- Dynamic layer growth and pruning
- Weight quantization

### Language Model (`neural_network_LLM.py`)
- Transformer-based architecture with adaptive topology
- Dynamic attention head pruning
- Adaptive layer growth based on activation patterns
- Compatible with GPT2 tokenizer

### Training Examples
- **MNIST** (`mnist_example.py`): Handwritten digit classification
- **Parity** (`parity_example.py`): N-bit parity problem
- **Language Model** (`train_llm.py`): Text generation with BabyLM dataset

### MNIST VAE Generator (`vae.py`)
- Variational Autoencoder with adaptive topology
- Features:
  - Dynamic convolutional layer growth
  - Adaptive latent space dimensionality
  - Automatic pruning of inactive channels
  - Weight quantization for efficiency
- Capabilities:
  - Generate new MNIST-like digits
  - Interpolate between digits in latent space
  - Visualize learned digit patterns
  - Analyze latent space structure
- Training:
  ```bash
  python vae.py
  ```
- Generated outputs:
  - `vae_samples_epoch_*.png`: Generated samples per epoch
  - `interpolation_*to*.png`: Digit interpolations
  - `latent_space.png`: 2D visualization of latent space
- Parameters:
  - Latent dimension: 16
  - Initial channels: [32, 64, 64]
  - Growth threshold: 0.5
  - Prune threshold: 0.05
  - Batch size: 128
  - Learning rate: 1e-3

### Visualization Tools
- Network topology visualization (`visualize_model.py`)
- MNIST pattern visualization (`visualize_mnist.py`)
- VAE latent space visualization (`vae.py`)

### Interactive Interface
- Chat interface for language models (`chat.py`)
- Real-time temperature adjustment
- Interactive text generation

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
```bash
# Train on MNIST
python mnist_example.py

# Train on parity problem
python parity_example.py
```

### Training a Language Model
```bash
# Train on text data
python train_llm.py --data_dir path/to/data \
    --hidden_size 256 \
    --num_layers 6 \
    --num_heads 8 \
    --batch_size 2 \
    --max_seq_length 512
```

### Using the Chat Interface
```bash
# Chat with trained language model
python chat.py --model_path outputs/best_model.pt
```

## Model Architecture

### Adaptive Neural Network
- Dynamic layer sizing based on training needs
- Skip connection management
- Pruning of inactive neurons
- Weight quantization for efficiency

### Adaptive Transformer
- Base transformer architecture with:
  - Multi-head self-attention
  - Position embeddings
  - Layer normalization
  - Feed-forward networks
- Adaptive features:
  - Dynamic head pruning
  - Layer growth based on activation patterns
  - Attention pattern analysis

### Adaptive VAE
- Convolutional architecture:
  - Encoder: Conv2d layers with increasing channels
  - Decoder: ConvTranspose2d layers with decreasing channels
  - Adaptive channel growth based on activation patterns
- Latent space features:
  - Dynamic dimensionality adjustment
  - Automatic structure discovery
  - KL divergence balancing
- Adaptive mechanisms:
  - Channel pruning based on activation statistics
  - Layer growth triggered by high variance
  - Weight quantization for memory efficiency
  - Stability-aware growth with cooldown periods
- Visualization capabilities:
  - Generated samples tracking
  - Latent space interpolation
  - t-SNE visualization of digit clusters
  - Channel activation analysis

## Training Parameters

### Language Model
- Learning rate: 3e-4
- Batch size: 2-4 (adjustable)
- Sequence length: 512
- Hidden size: 256
- Number of layers: 6
- Attention heads: 8

### Optimization
- AdamW optimizer
- Cosine learning rate schedule
- Gradient clipping
- Weight quantization every 100 steps
- Pruning every 500 steps
- Growth check every 1000 steps

## Visualization

The repository includes several visualization tools:
- Network topology visualization
- Attention pattern visualization
- Training progress monitoring
- Latent space visualization (for VAE)

## Chat Interface Features

- Interactive command-line interface
- Adjustable generation parameters:
  - Temperature (0.1-2.0)
  - Top-k filtering
  - Nucleus (top-p) sampling
- Special commands:
  - `help`: Show available commands
  - `temp=X`: Adjust temperature
  - `quit`: Exit interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{adaptiveNN,
  title = {Adaptive Neural Network with Dynamic Topology},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/adaptiveNN}
}
```

## Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- The Hugging Face team for the transformers library
- The rich library for beautiful console output 