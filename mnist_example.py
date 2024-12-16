import torch
import torchvision
import torchvision.transforms as transforms
from neural_network import AdaptiveNetwork
import numpy as np

def print_cuda_info():
    print("\nCUDA Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB\n")

def main():
    # Print CUDA information
    print_cuda_info()
    
    # Parameters
    input_size = 28 * 28
    hidden_size = 128
    output_size = 10
    batch_size = 128
    
    # Training parameters
    quantize_every = 100
    prune_every = 500
    growth_check_interval = 1000
    growth_threshold = 0.5
    prune_threshold = 0.05
    
    print("Loading MNIST dataset...")
    
    # CUDA setup
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("WARNING: CUDA is not available. Running on CPU will be much slower!")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load training data
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # Load test data
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # Create network
    network = AdaptiveNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        device=device
    ).to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}\n")
    
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("\nTraining network...")
    best_accuracy = 0
    epochs_without_improvement = 0
    max_epochs = 50
    
    # Track loss history for stability
    loss_history = []
    
    try:
        # Training loop
        for epoch in range(max_epochs):
            network.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Train on batches
            for i, (images, labels) in enumerate(train_loader):
                # Move data to device
                images = images.view(-1, input_size).to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = network(images)
                loss = criterion(outputs, labels)
                
                # Track loss for stability
                loss_history.append(loss.item())
                if len(loss_history) > 100:
                    loss_history.pop(0)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Periodic operations with stability checks
                if i % quantize_every == 0:
                    network.quantize_weights()
                
                if i % prune_every == 0 and epoch > 5:
                    network.prune_weights()
                
                if i % growth_check_interval == 0 and epoch > 3:
                    if len(loss_history) >= 100:
                        recent_loss_std = np.std(loss_history[-100:])
                        recent_loss_mean = np.mean(loss_history[-100:])
                        if recent_loss_std / recent_loss_mean < 0.1:
                            changes = network.grow_network()
                            if changes > 0:
                                print(f"\nNetwork grew by {changes} neurons")
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Print progress
                if i % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss/100:.3f}, '
                          f'Accuracy: {100 * correct/total:.2f}%')
                    if device.type == 'cuda':
                        print(f'GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB')
                    running_loss = 0.0
            
            # Evaluate on test set
            network.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.view(-1, input_size).to(device)
                    labels = labels.to(device)
                    outputs = network(images)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            test_accuracy = 100 * test_correct / test_total
            print(f'\nEpoch {epoch + 1} Test Accuracy: {test_accuracy:.2f}%')
            
            # Save best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                epochs_without_improvement = 0
                print("\nSaving best model...")
                model_state = {
                    'epoch': epoch,
                    'layer_sizes': network.layer_sizes,
                    'weights': [w.data.cpu() for w in network.weights],
                    'biases': [b.data.cpu() for b in network.biases],
                    'active_neurons': [n.cpu() for n in network.active_neurons],
                    'skip_connections': {k: v.data.cpu() for k, v in network.skip_connections.items()},
                    'accuracy': test_accuracy,
                    'device': device.type
                }
                torch.save(model_state, 'mnist_model.pt')
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= 10:
                print(f"\nEarly stopping: No improvement for {epochs_without_improvement} epochs")
                break
            
            # Print GPU memory after each epoch
            if device.type == 'cuda':
                print(f'GPU Memory after epoch {epoch + 1}: {torch.cuda.memory_allocated() / 1024**2:.1f}MB')
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current state...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': best_accuracy,
            'device': device.type
        }, 'mnist_model_interrupted.pt')
    
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Print final statistics
    stats = network.get_network_stats()
    print("\nFinal network structure:")
    for i, layer_stats in enumerate(stats):
        print(f"\nLayer {i+1}:")
        print(f"  Neurons: {layer_stats['active_neurons']}/{layer_stats['total_neurons']} active")
        if i > 0:
            sparsity = 1 - (layer_stats['nonzero_weights'] / layer_stats['total_weights'])
            print(f"  Weights: {layer_stats['nonzero_weights']}/{layer_stats['total_weights']} nonzero ({sparsity*100:.1f}% sparse)")
    
    print(f"\nBest test accuracy: {best_accuracy:.2f}%")
    print("Model saved as mnist_model.pt")
    print("\nRun visualize_mnist.py to see network visualizations")
    
    if device.type == 'cuda':
        print(f"\nFinal GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")

if __name__ == "__main__":
    main() 