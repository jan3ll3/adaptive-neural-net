import torch
import numpy as np
from neural_network import AdaptiveNetwork

def generate_parity_data(input_size, num_samples):
    """Generate data for the parity problem"""
    X = torch.randint(0, 2, (num_samples, input_size), dtype=torch.float32)
    y = torch.sum(X, dim=1) % 2
    return X, y.reshape(-1, 1)

def main():
    # Problem parameters
    input_size = 6
    hidden_size = input_size * 4  # More neurons to detect XOR patterns
    output_size = 1
    num_samples = 4000  # More training samples
    
    print(f"Generating {num_samples} samples for {input_size}-input parity problem...\n")
    
    # Generate some example patterns to verify correctness
    print("Example patterns testing each input:")
    print("Inputs -> Output")
    print("-" * 50)
    for i in range(input_size):
        x = torch.zeros(input_size)
        x[i] = 1
        y = (torch.sum(x) % 2).item()
        print(f"{x.numpy().astype(int)} -> {y}")
        x = torch.zeros(input_size)
        y = (torch.sum(x) % 2).item()
        print(f"{x.numpy().astype(int)} -> {y}")
    print()
    
    # Generate training data
    X, y = generate_parity_data(input_size, num_samples)
    
    # Create and train network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create network with XOR-focused architecture
    network = AdaptiveNetwork(input_size=input_size, 
                            hidden_size=hidden_size, 
                            output_size=output_size, 
                            device=device)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=0.002)  # Higher learning rate for XOR patterns
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Move data to device
    X = X.to(device)
    y = y.to(device)
    
    print("Training network...")
    best_loss = float('inf')
    epochs_without_improvement = 0
    max_epochs = 2000
    check_growth_interval = 150  # Less frequent growth with better initialization
    growth_threshold = 0.90  # Lower threshold since task is harder
    max_growth_attempts = 4
    growth_attempts = 0
    
    for epoch in range(max_epochs):
        try:
            # Forward pass
            optimizer.zero_grad()
            output = network.forward(X)
            loss = criterion(output, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = (torch.sigmoid(output) > 0.5).float()
                accuracy = (predictions == y).float().mean().item()
            
            # Print progress
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
            
            # Check for growth opportunities
            if epoch % check_growth_interval == 0 and epoch > 0:
                if accuracy < growth_threshold and growth_attempts < max_growth_attempts:
                    print("\nNetwork seems stuck. Analyzing growth opportunities...")
                    changes = network.grow_network(max_opportunities=2)
                    if changes > 0:
                        print(f"Made {changes} structural changes to the network")
                        growth_attempts += 1
                        # Reset optimizer with lower learning rate after growth
                        optimizer = torch.optim.Adam(network.parameters(), lr=0.002 * (0.9 ** growth_attempts))
                    else:
                        print("No growth opportunities found")
            
            # Analyze and prune network periodically
            if epoch % 200 == 0 and epoch > 0:
                try:
                    print("\nAnalyzing network structure...")
                    stats = network.prune_weights(threshold=0.05)  # Lower pruning threshold
                    print(f"Pruned {stats['pruned_weights']} weights, {stats['pruned_neurons']} neurons")
                    
                    print("\nNetwork Statistics:")
                    for i, stats in enumerate(network.get_network_stats()):
                        print(f"Layer {i+1}:")
                        print(f"  Neurons: {stats['active_neurons']}/{stats['total_neurons']} active")
                        if i > 0:  # Skip input layer
                            sparsity = 1 - (stats['nonzero_weights'] / stats['total_weights'])
                            print(f"  Weights: {stats['nonzero_weights']}/{stats['total_weights']} nonzero ({sparsity*100:.1f}% sparse)")
                    print(f"\nOverall network sparsity: {sparsity*100:.1f}%")
                except Exception as e:
                    print(f"Error during pruning: {str(e)}")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= 500:
                print(f"\nEarly stopping: No improvement for {epochs_without_improvement} epochs")
                break
                
            # Stop if we achieve very high accuracy
            if accuracy > 0.99:
                print(f"\nReached high accuracy ({accuracy:.4f}) at epoch {epoch}")
                break
                
        except Exception as e:
            print(f"Error during training: {str(e)}")
            break
    
    print("\nFinal network structure:\n")
    print("Network Statistics:")
    for i, stats in enumerate(network.get_network_stats()):
        print(f"Layer {i+1}:")
        print(f"  Neurons: {stats['active_neurons']}/{stats['total_neurons']} active")
        if i > 0:  # Skip input layer
            sparsity = 1 - (stats['nonzero_weights'] / stats['total_weights'])
            print(f"  Weights: {stats['nonzero_weights']}/{stats['total_weights']} nonzero ({sparsity*100:.1f}% sparse)")
    print(f"\nOverall network sparsity: {sparsity*100:.1f}%")
    
    # Test some specific patterns
    print("\nTesting specific patterns:")
    test_patterns = [
        torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32),  # All zeros
        torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32),  # All ones
        torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32),  # Single one
        torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float32),  # Two ones
        torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32),  # Three ones
        torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32),  # Alternating
    ]
    
    network.eval()  # Set to evaluation mode
    with torch.no_grad():
        for pattern in test_patterns:
            pattern = pattern.to(device)
            output = network.forward(pattern.unsqueeze(0))
            pred = (torch.sigmoid(output) > 0.5).float().item()
            actual = (torch.sum(pattern) % 2).item()
            print(f"Input: {pattern.cpu().numpy().astype(int)} -> Predicted: {int(pred)}, Actual: {actual}")
    
    # After all testing, save the model
    print("\nSaving model...")
    model_state = {
        'layer_sizes': network.layer_sizes,
        'weights': [w.data.cpu() for w in network.weights],
        'biases': [b.data.cpu() for b in network.biases],
        'active_neurons': [n.cpu() for n in network.active_neurons],
        'skip_connections': {k: v.data.cpu() for k, v in network.skip_connections.items()}
    }
    torch.save(model_state, 'parity_model.pt')
    print("Model saved to parity_model.pt")

if __name__ == "__main__":
    main() 