import torch
import numpy as np
import torch.nn as nn

class AdaptiveNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device='cuda'):
        super(AdaptiveNetwork, self).__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Store layer dimensions
        self.layer_sizes = [input_size, hidden_size, hidden_size//2, output_size]
        
        # Initialize weights and biases
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        # Create standard layer connections with parity-focused initialization
        for i in range(len(self.layer_sizes) - 1):
            if i == 0:  # Input to first hidden: initialize with XOR-like patterns
                w = torch.zeros(self.layer_sizes[i+1], self.layer_sizes[i])
                # Initialize with XOR-like patterns
                for j in range(min(hidden_size, input_size * (input_size-1))):
                    pattern = torch.zeros(input_size)
                    idx1 = j % input_size
                    idx2 = (j // input_size + 1) % input_size
                    pattern[idx1] = 1.0
                    pattern[idx2] = -1.0  # Opposite sign for XOR detection
                    w[j] = pattern
                # Rest get small random initialization
                if hidden_size > input_size * (input_size-1):
                    w[input_size * (input_size-1):] = torch.randn(
                        hidden_size - input_size * (input_size-1), 
                        input_size
                    ) * 0.1
                # Add bias to make XOR threshold
                b = torch.ones(self.layer_sizes[i+1]) * 0.5
            
            elif i == 1:  # First to second hidden: combine XOR patterns
                w = torch.randn(self.layer_sizes[i+1], self.layer_sizes[i]) * 0.1
                # Initialize some neurons to combine pairs of XOR detectors
                for j in range(min(self.layer_sizes[i+1], hidden_size//2)):
                    w[j, j*2:j*2+2] = 1.0
                b = torch.zeros(self.layer_sizes[i+1])
            
            else:  # Second hidden to output: count active XOR patterns
                w = torch.ones(self.layer_sizes[i+1], self.layer_sizes[i])
                w = w / self.layer_sizes[i]  # Normalize to detect proportion of active units
                b = torch.zeros(self.layer_sizes[i+1]) - 0.5  # Threshold at 50%
            
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(b))
        
        # Skip connections: (from_layer, to_layer) -> weight matrix
        self.skip_connections = nn.ModuleDict()
        
        # Active neurons in each layer (1 = active, 0 = pruned)
        self.active_neurons = nn.ParameterList([
            nn.Parameter(torch.ones(size, dtype=torch.bool), requires_grad=False)
            for size in self.layer_sizes
        ])
        
        # Store activations for analysis
        self.last_activations = None
        self.activation_correlations = {}
        
        # Move to device
        self.to(self.device)
    
    def to(self, device):
        """Override to method to handle custom parameters"""
        super().to(device)
        self.device = device
        return self
    
    def train(self, mode=True):
        """Set the network in training mode"""
        super().train(mode)
        return self
    
    def eval(self):
        """Set the network in evaluation mode"""
        return self.train(False)
    
    def forward(self, x):
        """Forward pass through the network"""
        activations = [x]
        
        # Process each layer
        for i in range(len(self.weights)):
            # Start with standard connections
            layer_input = torch.mm(activations[-1], self.weights[i].t()) + self.biases[i]
            
            # Add skip connections to this layer
            for (from_layer, to_layer), weight in self.skip_connections.items():
                if to_layer == i + 1:
                    skip_input = torch.mm(activations[from_layer], weight.t())
                    layer_input += skip_input
            
            # Apply activation (ReLU for hidden, none for output)
            output = torch.relu(layer_input) if i < len(self.weights) - 1 else layer_input
            
            # Apply neuron masking
            output = output * self.active_neurons[i + 1].float()
            
            activations.append(output)
        
        # Store activations for analysis only during training
        if self.training:
            self.last_activations = activations
        return activations[-1]
    
    def add_skip_connection(self, from_layer, to_layer):
        """Add a skip connection between layers"""
        if from_layer >= to_layer or from_layer < 0 or to_layer >= len(self.layer_sizes):
            raise ValueError("Invalid skip connection")
        
        key = f"{from_layer}_{to_layer}"
        if key not in self.skip_connections:
            w = torch.randn(self.layer_sizes[to_layer], self.layer_sizes[from_layer]) / np.sqrt(self.layer_sizes[from_layer])
            self.skip_connections[key] = nn.Parameter(w)
    
    def quantize_weights(self, bits=8):
        """Quantize weights to specified number of bits"""
        with torch.no_grad():
            for w in self.weights:
                if w is not None:
                    max_val = torch.max(torch.abs(w.data))
                    scale = (2 ** (bits - 1) - 1) / max_val
                    w.data = torch.round(w.data * scale) / scale
    
    def prune_weights(self):
        """Prune weights below threshold"""
        with torch.no_grad():
            for i, w in enumerate(self.weights):
                if w is not None:
                    threshold = torch.std(w.data) * 0.1
                    mask = (torch.abs(w.data) > threshold).float()
                    w.data *= mask
                    
                    # Update active neurons based on remaining connections
                    in_connections = torch.any(mask > 0, dim=0)
                    out_connections = torch.any(mask > 0, dim=1)
                    self.active_neurons[i].data &= in_connections
                    self.active_neurons[i + 1].data &= out_connections
    
    def grow_network(self, max_opportunities=2):
        """Add neurons based on activation patterns"""
        if self.last_activations is None:
            return 0
            
        changes = 0
        for layer_idx in range(1, len(self.layer_sizes) - 1):
            if changes >= max_opportunities:
                break
                
            layer_activations = self.last_activations[layer_idx]
            if layer_activations.shape[1] >= self.layer_sizes[layer_idx] * 2:
                continue
                
            # Check if layer is saturated
            activation_means = torch.mean((layer_activations > 0).float(), dim=0)
            if torch.mean(activation_means) > 0.9:
                # Add neurons
                new_size = min(self.layer_sizes[layer_idx] * 2, self.layer_sizes[layer_idx] + 10)
                self.add_layer(new_size, layer_idx)
                changes += 1
        
        return changes
    
    def get_network_stats(self):
        """Get statistics about the network structure"""
        stats = []
        for i in range(len(self.layer_sizes)):
            layer_stats = {
                'total_neurons': self.layer_sizes[i],
                'active_neurons': torch.sum(self.active_neurons[i]).item()
            }
            if i > 0:
                w = self.weights[i-1]
                layer_stats.update({
                    'total_weights': w.numel(),
                    'nonzero_weights': torch.sum(w != 0).item()
                })
            stats.append(layer_stats)
        return stats