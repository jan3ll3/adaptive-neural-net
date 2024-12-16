import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_network(model_state, title="Network Visualization"):
    """Visualize network topology with active neurons and connections"""
    plt.figure(figsize=(15, 10))
    
    # Get layer sizes
    layer_sizes = model_state['layer_sizes']
    weights = model_state['weights']
    active_neurons = model_state['active_neurons']
    skip_connections = model_state['skip_connections']
    
    max_size = max(layer_sizes)
    
    # Calculate positions for neurons
    layer_positions = []
    for i, size in enumerate(layer_sizes):
        x = i
        positions = np.linspace(-size/max_size, size/max_size, size)
        layer_positions.append([(x, y) for y in positions])
    
    # Draw connections between layers
    for i in range(len(weights)):
        weight_matrix = weights[i]
        max_weight = torch.max(torch.abs(weight_matrix)).item()
        
        for out_idx in range(weight_matrix.shape[0]):
            if active_neurons[i+1][out_idx]:  # Only draw connections to active neurons
                for in_idx in range(weight_matrix.shape[1]):
                    if active_neurons[i][in_idx]:  # Only draw connections from active neurons
                        weight = weight_matrix[out_idx, in_idx].item()
                        if abs(weight) > 1e-6:  # Only draw non-zero connections
                            color = 'red' if weight > 0 else 'blue'
                            alpha = float(min(1.0, abs(weight) / max_weight))
                            plt.plot([layer_positions[i][in_idx][0], layer_positions[i+1][out_idx][0]],
                                   [layer_positions[i][in_idx][1], layer_positions[i+1][out_idx][1]],
                                   color=color, alpha=alpha, linewidth=alpha*2,
                                   zorder=1)
    
    # Draw skip connections
    for (from_layer, to_layer), weight_matrix in skip_connections.items():
        max_weight = torch.max(torch.abs(weight_matrix)).item()
        
        for out_idx in range(weight_matrix.shape[0]):
            if active_neurons[to_layer][out_idx]:
                for in_idx in range(weight_matrix.shape[1]):
                    if active_neurons[from_layer][in_idx]:
                        weight = weight_matrix[out_idx, in_idx].item()
                        if abs(weight) > 1e-6:
                            color = 'green'
                            alpha = float(min(1.0, abs(weight) / max_weight))
                            plt.plot([layer_positions[from_layer][in_idx][0], layer_positions[to_layer][out_idx][0]],
                                   [layer_positions[from_layer][in_idx][1], layer_positions[to_layer][out_idx][1]],
                                   color=color, alpha=alpha, linewidth=alpha*2,
                                   linestyle='--', zorder=1)
    
    # Draw neurons
    for i, positions in enumerate(layer_positions):
        for j, (x, y) in enumerate(positions):
            if active_neurons[i][j]:
                if i == 0:  # Input layer
                    color = 'lightgreen'
                    label = f'in_{j}'
                elif i == len(layer_sizes)-1:  # Output layer
                    color = 'yellow'
                    label = 'out'
                else:  # Hidden layers
                    color = 'lightblue'
                    label = f'h{i}_{j}'
                
                plt.scatter(x, y, c=color, s=100, zorder=2)
                plt.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Add layer labels
    layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(layer_sizes)-2)] + ['Output']
    for i, name in enumerate(layer_names):
        plt.text(i, 1.1, name, ha='center')
        active_count = torch.sum(active_neurons[i]).item()
        total_count = layer_sizes[i]
        plt.text(i, -1.1, f'Active: {active_count}/{total_count}', ha='center')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', label='Positive Weight'),
        plt.Line2D([0], [0], color='blue', label='Negative Weight'),
        plt.Line2D([0], [0], color='green', linestyle='--', label='Skip Connection'),
        plt.scatter([0], [0], c='lightgreen', label='Input Neuron'),
        plt.scatter([0], [0], c='lightblue', label='Hidden Neuron'),
        plt.scatter([0], [0], c='yellow', label='Output Neuron')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()

def main():
    print("Loading model from parity_model.pt...")
    model_state = torch.load('parity_model.pt')
    
    print("\nNetwork structure:")
    print(f"Layer sizes: {model_state['layer_sizes']}")
    for i, (active, total) in enumerate(zip(
        [torch.sum(n).item() for n in model_state['active_neurons']], 
        model_state['layer_sizes']
    )):
        print(f"Layer {i}: {active}/{total} neurons active")
    
    print("\nGenerating visualization...")
    visualize_network(model_state, "Parity Network Structure")

if __name__ == "__main__":
    main() 