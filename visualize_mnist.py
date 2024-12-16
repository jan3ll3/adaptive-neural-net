import torch
import torchvision
import torchvision.transforms as transforms
from neural_network import AdaptiveNetwork
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection
import torch.nn as nn

def visualize_network(network, max_neurons_per_layer=50):
    """Visualize network structure with weights and skip connections"""
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=0.9)  # Add space for title
    
    # Get network statistics
    stats = network.get_network_stats()
    max_layer_size = max(s['total_neurons'] for s in stats)
    
    # Calculate positions
    layer_positions = []
    neuron_positions = []
    
    for layer_idx, layer_stats in enumerate(stats):
        # Calculate vertical spacing
        active_neurons = min(layer_stats['total_neurons'], max_neurons_per_layer)
        y_positions = np.linspace(-active_neurons/2, active_neurons/2, active_neurons)
        
        # Store positions
        layer_positions.append(layer_idx * 3)  # Horizontal position
        neuron_positions.append(y_positions)
        
        # Draw neurons
        for i, y in enumerate(y_positions):
            # Color neurons based on whether they're active
            is_active = i < layer_stats['active_neurons']
            color = 'lightblue' if is_active else 'lightgray'
            circle = Circle((layer_idx * 3, y), 0.2, color=color, zorder=2)
            plt.gca().add_patch(circle)
            
            # Add neuron count if showing subset
            if layer_stats['total_neurons'] > max_neurons_per_layer and i == len(y_positions) - 1:
                plt.text(layer_idx * 3, y + 0.5, 
                        f"+{layer_stats['total_neurons'] - max_neurons_per_layer} more",
                        ha='center', va='bottom')
    
    # Draw connections
    for layer_idx in range(len(stats) - 1):
        if layer_idx >= len(network.weights):
            continue
            
        weights = network.weights[layer_idx].detach().cpu()
        max_weight = torch.max(torch.abs(weights)).item()  # Convert to Python float
        
        # Sample connections if too many
        source_neurons = min(len(neuron_positions[layer_idx]), max_neurons_per_layer)
        target_neurons = min(len(neuron_positions[layer_idx + 1]), max_neurons_per_layer)
        
        # Calculate connection density
        total_connections = source_neurons * target_neurons
        max_connections = 1000  # Maximum number of connections to show
        sample_rate = max_connections / total_connections if total_connections > max_connections else 1.0
        
        for i in range(source_neurons):
            for j in range(target_neurons):
                if np.random.random() > sample_rate:
                    continue
                    
                weight = weights[j, i].item() if j < weights.shape[0] and i < weights.shape[1] else 0
                if abs(weight) > max_weight * 0.1:  # Only show significant weights
                    alpha = float(min(1.0, abs(weight) / max_weight))  # Convert to Python float
                    color = 'red' if weight < 0 else 'blue'
                    arrow = FancyArrowPatch(
                        (layer_positions[layer_idx], neuron_positions[layer_idx][i]),
                        (layer_positions[layer_idx + 1], neuron_positions[layer_idx + 1][j]),
                        arrowstyle='-',
                        alpha=alpha,
                        color=color,
                        linewidth=1,
                        zorder=1
                    )
                    plt.gca().add_patch(arrow)
    
    # Draw skip connections
    for key, weight in network.skip_connections.items():
        from_layer, to_layer = map(int, key.split('_'))
        weights = weight.detach().cpu()
        max_weight = torch.max(torch.abs(weights)).item()  # Convert to Python float
        
        source_neurons = min(len(neuron_positions[from_layer]), max_neurons_per_layer)
        target_neurons = min(len(neuron_positions[to_layer]), max_neurons_per_layer)
        
        # Calculate connection density for skip connections
        total_connections = source_neurons * target_neurons
        max_connections = 500  # Maximum number of skip connections to show
        sample_rate = max_connections / total_connections if total_connections > max_connections else 1.0
        
        for i in range(source_neurons):
            for j in range(target_neurons):
                if np.random.random() > sample_rate:
                    continue
                    
                w = weights[j, i].item() if j < weights.shape[0] and i < weights.shape[1] else 0
                if abs(w) > max_weight * 0.1:
                    alpha = float(min(1.0, abs(w) / max_weight))  # Convert to Python float
                    arrow = FancyArrowPatch(
                        (layer_positions[from_layer], neuron_positions[from_layer][i]),
                        (layer_positions[to_layer], neuron_positions[to_layer][j]),
                        arrowstyle='->',
                        alpha=alpha,
                        color='green',
                        linewidth=1,
                        connectionstyle='arc3,rad=.3',
                        zorder=1
                    )
                    plt.gca().add_patch(arrow)
    
    # Set plot limits and labels
    plt.xlim(-1, max(layer_positions) + 1)
    plt.ylim(-max_layer_size/2 - 1, max_layer_size/2 + 1)
    plt.axis('equal')
    plt.axis('off')
    
    # Add layer labels
    for i, x in enumerate(layer_positions):
        plt.text(x, -max_layer_size/2 - 0.5, f'Layer {i+1}\n{stats[i]["active_neurons"]}/{stats[i]["total_neurons"]}',
                ha='center', va='top')
    
    plt.title('Network Structure\nBlue: positive weights, Red: negative weights, Green: skip connections')
    plt.show()

def visualize_activations(network, sample_input, max_neurons=20):
    """Visualize neuron activations for a sample input"""
    network.eval()
    with torch.no_grad():
        # Forward pass storing activations
        activations = [sample_input]
        x = sample_input
        
        for i in range(len(network.weights)):
            layer_input = torch.mm(activations[-1], network.weights[i].t()) + network.biases[i]
            
            # Add skip connections
            for (from_layer, to_layer), weight in network.skip_connections.items():
                if to_layer == i + 1:
                    skip_input = torch.mm(activations[from_layer], weight.t())
                    layer_input += skip_input
            
            # Apply activation
            output = torch.relu(layer_input) if i < len(network.weights) - 1 else layer_input
            activations.append(output)
    
    # Plot activations
    plt.figure(figsize=(15, 8))
    plt.subplots_adjust(top=0.9)  # Add space for title
    
    # Calculate positions
    layer_positions = np.arange(len(activations)) * 3
    max_neurons_shown = min(max_neurons, max(act.shape[1] for act in activations))
    
    # Plot each layer's activations
    for layer_idx, act in enumerate(activations):
        act = act.cpu().numpy()[0]  # Get first sample
        neurons_to_show = min(max_neurons, act.shape[0])
        y_positions = np.linspace(-neurons_to_show/2, neurons_to_show/2, neurons_to_show)
        
        # Normalize activations for visualization
        act_norm = (act - act.min()) / (act.max() - act.min() + 1e-6)
        
        for i, y in enumerate(y_positions):
            if i < act.shape[0]:
                activation = act_norm[i]
                circle = Circle((layer_positions[layer_idx], y), 0.2, 
                              color=plt.cm.viridis(activation),
                              zorder=2)
                plt.gca().add_patch(circle)
                
                # Add activation value
                plt.text(layer_positions[layer_idx] + 0.3, y, 
                        f'{act[i]:.2f}', 
                        va='center', ha='left',
                        fontsize=8)
        
        # Show number of hidden neurons if truncated
        if act.shape[0] > max_neurons:
            plt.text(layer_positions[layer_idx], max_neurons_shown/2 + 0.5,
                    f'+{act.shape[0] - max_neurons} more',
                    ha='center', va='bottom')
    
    # Set plot limits
    plt.xlim(-1, max(layer_positions) + 1)
    plt.ylim(-max_neurons_shown/2 - 1, max_neurons_shown/2 + 1)
    plt.axis('equal')
    plt.axis('off')
    
    # Add layer labels
    for i, x in enumerate(layer_positions):
        plt.text(x, -max_neurons_shown/2 - 0.5, f'Layer {i+1}',
                ha='center', va='top')
    
    plt.title('Neuron Activations\nColor intensity represents activation strength')
    plt.show()

def visualize_learned_patterns(network, layer_idx=0, top_k=10):
    """Visualize the patterns learned by neurons in a specific layer"""
    if layer_idx >= len(network.weights):
        print(f"Layer {layer_idx} does not exist")
        return
    
    # Move all weights to CPU for visualization
    weights = [w.detach().cpu() for w in network.weights]
    patterns = weights[layer_idx]
    
    # For input layer, weights directly represent patterns
    if layer_idx == 0:
        patterns = patterns
    else:
        # For hidden layers, we need to compute effective input patterns
        patterns = patterns.clone()
        for i in range(layer_idx - 1, -1, -1):
            patterns = torch.mm(patterns, weights[i])
    
    # Get top-k neurons by weight magnitude
    magnitudes = torch.norm(patterns, dim=1)
    top_indices = torch.argsort(magnitudes, descending=True)[:top_k]
    
    # Plot patterns
    n_cols = 5
    n_rows = (top_k + n_cols - 1) // n_cols
    plt.figure(figsize=(2*n_cols, 2*n_rows))
    plt.subplots_adjust(top=0.9)  # Add space for title
    
    for i, idx in enumerate(top_indices):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Reshape and normalize pattern for visualization
        pattern = patterns[idx].view(28, 28)
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        plt.imshow(pattern, cmap='viridis')
        plt.axis('off')
        plt.title(f'Neuron {idx}')
    
    plt.suptitle(f'Top {top_k} Learned Patterns in Layer {layer_idx + 1}')
    plt.show()

def show_digit_predictions(network, device='cuda'):
    """Show predictions for each digit"""
    # Load MNIST test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Find one example of each digit
    digit_examples = {}
    digit_activations = {}
    
    network.eval()
    with torch.no_grad():
        for img, label in test_dataset:
            # Convert label to int if it's a tensor
            label = label.item() if torch.is_tensor(label) else int(label)
            if label not in digit_examples:
                # Get network prediction and activations
                input_tensor = img.view(1, -1).to(device)
                output = network(input_tensor)
                pred = output.argmax().item()
                
                if pred == label:  # Only show correct predictions
                    digit_examples[label] = img
                    digit_activations[label] = output[0].cpu()  # Move to CPU for plotting
                
                if len(digit_examples) == 10:
                    break
    
    # Plot digits and their activations
    plt.figure(figsize=(15, 6))
    plt.subplots_adjust(top=0.9, bottom=0.1)  # Add space for title and labels
    
    for i in range(10):
        # Plot digit
        plt.subplot(2, 10, i + 1)
        plt.imshow(digit_examples[i].squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f'Digit {i}')
        
        # Plot activations
        plt.subplot(2, 10, i + 11)
        activations = digit_activations[i].numpy()
        plt.bar(range(10), activations)
        plt.ylim(min(0, activations.min()), activations.max() * 1.1)
        plt.xticks([])
        if i == 0:
            plt.ylabel('Activation')
    
    plt.suptitle('Network Predictions and Activations for Each Digit')
    plt.show()

def main():
    # Load the trained model
    try:
        model_state = torch.load('mnist_model.pt', weights_only=True)  # Add weights_only=True to avoid warning
    except FileNotFoundError:
        print("Error: mnist_model.pt not found. Please train the model first using mnist_example.py")
        return
    
    # Create network with same architecture
    device = torch.device(model_state.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    network = AdaptiveNetwork(
        input_size=model_state['layer_sizes'][0],
        hidden_size=model_state['layer_sizes'][1],
        output_size=model_state['layer_sizes'][-1],
        device=device
    ).to(device)
    
    # Load weights and biases
    for i, (w, b) in enumerate(zip(model_state['weights'], model_state['biases'])):
        network.weights[i].data = w.to(device)
        network.biases[i].data = b.to(device)
    
    # Load skip connections
    for key, weight in model_state['skip_connections'].items():
        network.skip_connections[key] = nn.Parameter(weight.to(device))
    
    # Load active neurons
    for i, active in enumerate(model_state['active_neurons']):
        network.active_neurons[i].data = active.to(device)
    
    print(f"Loaded model with {model_state['accuracy']:.2f}% accuracy")
    
    # Show network structure
    print("\nNetwork Structure:")
    visualize_network(network)
    
    # Show learned patterns for each layer
    for layer_idx in range(len(network.weights)):
        print(f"\nLearned Patterns in Layer {layer_idx + 1}:")
        visualize_learned_patterns(network, layer_idx=layer_idx)
    
    # Show digit predictions
    print("\nNetwork Predictions:")
    show_digit_predictions(network, device)
    
    # Print statistics
    print("\nNetwork Statistics:")
    stats = network.get_network_stats()
    for i, layer_stats in enumerate(stats):
        print(f"\nLayer {i+1}:")
        print(f"  Neurons: {layer_stats['active_neurons']}/{layer_stats['total_neurons']} active")
        if i > 0:
            sparsity = 1 - (layer_stats['nonzero_weights'] / layer_stats['total_weights'])
            print(f"  Weights: {layer_stats['nonzero_weights']}/{layer_stats['total_weights']} nonzero ({sparsity*100:.1f}% sparse)")

if __name__ == "__main__":
    main() 