import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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

class AdaptiveConvVAE(nn.Module):
    def __init__(self, latent_dim=16, growth_threshold=0.5, prune_threshold=0.05):
        super(AdaptiveConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.growth_threshold = growth_threshold
        self.prune_threshold = prune_threshold
        
        # Track layer sizes for dynamic growth
        self.layer_sizes = {
            'conv1': 32,
            'conv2': 64,
            'conv3': 64,
            'hidden': 256,
            'latent': latent_dim
        }
        
        # Track maximum sizes to prevent over-growth
        self.max_sizes = {
            'conv1': 64,   # Limit maximum size
            'conv2': 128,
            'conv3': 128,
            'hidden': 512,
            'latent': latent_dim
        }
        
        self.build_network()
        
        # Initialize masks for pruning (will be moved to correct device later)
        self.masks = {}
        
        # Track activation statistics for growth
        self.activation_stats = {}
        self.gradient_stats = {}
        
        # Track loss history for stability
        self.loss_history = []
        self.growth_cooldown = 0  # Cooldown period after growth
    
    def to(self, device):
        """Override to method to handle masks"""
        super().to(device)
        # Initialize or move masks to device
        self.initialize_masks(device)
        return self
        
    def initialize_masks(self, device=None):
        """Initialize binary masks for each layer"""
        if device is None:
            device = next(self.parameters()).device
            
        self.masks = {
            'conv1': torch.ones_like(self.conv1.weight.data, device=device),
            'conv2': torch.ones_like(self.conv2.weight.data, device=device),
            'conv3': torch.ones_like(self.conv3.weight.data, device=device),
            'fc_hidden': torch.ones_like(self.fc_hidden.weight.data, device=device),
            'fc_mu': torch.ones_like(self.fc_mu.weight.data, device=device),
            'fc_var': torch.ones_like(self.fc_var.weight.data, device=device),
            'decoder_linear1': torch.ones_like(self.decoder_linear1.weight.data, device=device),
            'decoder_linear2': torch.ones_like(self.decoder_linear2.weight.data, device=device),
            'deconv1': torch.ones_like(self.deconv1.weight.data, device=device),
            'deconv2': torch.ones_like(self.deconv2.weight.data, device=device),
            'deconv3': torch.ones_like(self.deconv3.weight.data, device=device)
        }
    
    def quantize_weights(self, bits=8):
        """Quantize weights to specified number of bits"""
        device = next(self.parameters()).device  # Get current device
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    # Calculate scale for quantization
                    max_val = torch.max(torch.abs(param.data))
                    scale = (2 ** (bits - 1) - 1) / max_val
                    
                    # Quantize
                    param.data = torch.round(param.data * scale) / scale
                    
                    # Apply mask if exists
                    if name.split('.')[0] in self.masks:
                        mask = self.masks[name.split('.')[0]]
                        if mask.device != device:
                            mask = mask.to(device)
                            self.masks[name.split('.')[0]] = mask
                        param.data *= mask
    
    def prune_weights(self):
        """Prune weights below threshold with stability checks"""
        device = next(self.parameters()).device
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    mask_name = name.split('.')[0]
                    if mask_name in self.masks:
                        # More conservative threshold based on distribution
                        mean_abs = torch.mean(torch.abs(param.data))
                        std_abs = torch.std(torch.abs(param.data))
                        threshold = mean_abs - self.prune_threshold * std_abs
                        
                        # Ensure we don't prune too many weights
                        new_mask = (torch.abs(param.data) > threshold).float()
                        prune_ratio = 1.0 - torch.mean(new_mask).item()
                        
                        if prune_ratio < 0.5:  # Don't prune more than 50% of weights
                            self.masks[mask_name] = new_mask.to(device)
                            param.data *= self.masks[mask_name]
    
    def grow_neurons(self):
        """Add neurons based on activation patterns with stability checks"""
        if self.growth_cooldown > 0:
            self.growth_cooldown -= 1
            return
            
        with torch.no_grad():
            growth_happened = False
            for name, stats in self.activation_stats.items():
                if stats['mean'] > self.growth_threshold:
                    # Check if we're under max size
                    if name in self.layer_sizes and self.layer_sizes[name] < self.max_sizes[name]:
                        # Calculate more conservative growth
                        current_size = self.layer_sizes[name]
                        growth = max(1, int(0.05 * current_size))  # Reduced from 0.1 to 0.05
                        new_size = min(current_size + growth, self.max_sizes[name])
                        
                        if new_size > current_size:
                            self.layer_sizes[name] = new_size
                            print(f"Growing {name} layer from {current_size} to {new_size} neurons")
                            growth_happened = True
            
            if growth_happened:
                # Rebuild network with new sizes
                device = next(self.parameters()).device
                self.build_network()
                self.to(device)
                self.growth_cooldown = 2  # Set cooldown period
    
    def update_stats(self, x, layer_name):
        """Update activation statistics for a layer"""
        with torch.no_grad():
            mean_activation = torch.mean(torch.abs(x))
            if layer_name not in self.activation_stats:
                self.activation_stats[layer_name] = {'mean': mean_activation}
            else:
                # Exponential moving average
                self.activation_stats[layer_name]['mean'] = (
                    0.9 * self.activation_stats[layer_name]['mean'] + 
                    0.1 * mean_activation
                )
    
    def encode(self, x):
        # Apply masks and update stats through encoder
        x = F.relu(self.conv1(x))
        self.update_stats(x, 'conv1')
        
        x = F.relu(self.conv2(x))
        self.update_stats(x, 'conv2')
        
        x = F.relu(self.conv3(x))
        self.update_stats(x, 'conv3')
        
        x = x.view(-1, self.layer_sizes['conv3'] * 7 * 7)
        x = F.relu(self.fc_hidden(x))
        self.update_stats(x, 'hidden')
        
        return self.fc_mu(x), self.fc_var(x)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.decoder_linear1(z))
        h = F.relu(self.decoder_linear2(h))
        h = h.view(-1, self.layer_sizes['conv3'], 7, 7)
        
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        return torch.sigmoid(self.deconv3(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def build_network(self):
        """Build the network architecture with current layer sizes"""
        # Encoder
        self.conv1 = nn.Conv2d(1, self.layer_sizes['conv1'], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.layer_sizes['conv1'], self.layer_sizes['conv2'], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(self.layer_sizes['conv2'], self.layer_sizes['conv3'], kernel_size=3, stride=1, padding=1)
        self.fc_hidden = nn.Linear(self.layer_sizes['conv3'] * 7 * 7, self.layer_sizes['hidden'])
        
        # Latent space
        self.fc_mu = nn.Linear(self.layer_sizes['hidden'], self.layer_sizes['latent'])
        self.fc_var = nn.Linear(self.layer_sizes['hidden'], self.layer_sizes['latent'])
        
        # Decoder
        self.decoder_linear1 = nn.Linear(self.layer_sizes['latent'], self.layer_sizes['hidden'])
        self.decoder_linear2 = nn.Linear(self.layer_sizes['hidden'], self.layer_sizes['conv3'] * 7 * 7)
        
        self.deconv1 = nn.ConvTranspose2d(self.layer_sizes['conv3'], self.layer_sizes['conv2'], 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(self.layer_sizes['conv2'], self.layer_sizes['conv1'], 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(self.layer_sizes['conv1'], 1, 4, stride=2, padding=1)

def loss_function(recon_x, x, mu, log_var, kld_weight=1.0):
    # Reconstruction loss (binary cross entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + kld_weight * KLD

def train(model, device, train_loader, optimizer, epoch, quantize_every=100, prune_every=500):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        # Track loss for stability
        model.loss_history.append(loss.item())
        if len(model.loss_history) > 100:
            model.loss_history.pop(0)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        # Periodic operations with stability checks
        if batch_idx % quantize_every == 0:
            model.quantize_weights()
        
        if batch_idx % prune_every == 0 and epoch > 5:  # Start pruning after epoch 5
            model.prune_weights()
            
        if batch_idx % 1000 == 0 and epoch > 3:  # Start growth after epoch 3
            # Check if loss is stable before growing
            if len(model.loss_history) >= 100:
                recent_loss_std = np.std(model.loss_history[-100:])
                recent_loss_mean = np.mean(model.loss_history[-100:])
                if recent_loss_std / recent_loss_mean < 0.1:  # Only grow if loss is stable
                    model.grow_neurons()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
            if device.type == 'cuda':
                print(f'GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB')
    
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def generate_samples(model, device, num_samples=10):
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, model.fc_mu.out_features).to(device)
        # Decode the latent vectors
        sample = model.decode(z)
        return sample

def interpolate_digits(model, device, start_digit, end_digit, steps=10):
    """Generate interpolation between two digits"""
    model.eval()
    
    # Get a batch of test images
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True,
        transform=transforms.ToTensor()
    )
    
    # Find examples of start and end digits
    start_img = None
    end_img = None
    for img, label in test_dataset:
        if label == start_digit and start_img is None:
            start_img = img
        if label == end_digit and end_img is None:
            end_img = img
        if start_img is not None and end_img is not None:
            break
    
    with torch.no_grad():
        # Encode both images
        start_mu, _ = model.encode(start_img.unsqueeze(0).to(device))
        end_mu, _ = model.encode(end_img.unsqueeze(0).to(device))
        
        # Create interpolation steps
        alphas = np.linspace(0, 1, steps)
        interpolated = []
        
        for alpha in alphas:
            # Interpolate in latent space
            z = alpha * end_mu + (1 - alpha) * start_mu
            # Decode
            decoded = model.decode(z)
            interpolated.append(decoded.squeeze().cpu())
        
        # Plot
        plt.figure(figsize=(15, 3))
        for i, img in enumerate(interpolated):
            plt.subplot(1, steps, i + 1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Interpolation from {start_digit} to {end_digit}')
        plt.savefig(f'interpolation_{start_digit}to{end_digit}.png')
        plt.close()

def visualize_latent_space(model, device):
    """Visualize the structure of the latent space"""
    try:
        # Get test data with explicit transforms
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        test_dataset = torchvision.datasets.MNIST(
            './data', train=False, download=True,
            transform=transform
        )
        
        # Use smaller batch size and enable pin memory
        test_loader = DataLoader(
            test_dataset,
            batch_size=100,  # Reduced batch size
            shuffle=True,
            pin_memory=True,
            num_workers=0  # Disable multiprocessing for stability
        )
        
        model.eval()
        latent_vectors = []
        labels = []
        
        # Get latent vectors for test images
        print("Collecting latent vectors...")
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                if i >= 20:  # Limit to 2000 samples (20 batches of 100)
                    break
                    
                data = data.to(device)
                mu, _ = model.encode(data)
                latent_vectors.append(mu.cpu())
                labels.extend(target.numpy())
        
        print("Processing latent space...")
        latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
        labels = np.array(labels)
        
        # Use t-SNE for dimensionality reduction if latent_dim > 2
        if latent_vectors.shape[1] > 2:
            try:
                from sklearn.manifold import TSNE
                print("Performing t-SNE dimensionality reduction...")
                tsne = TSNE(n_components=2, random_state=42)
                latent_2d = tsne.fit_transform(latent_vectors)
            except ImportError:
                print("Warning: sklearn not found, using PCA instead")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                latent_2d = pca.fit_transform(latent_vectors)
        else:
            latent_2d = latent_vectors
        
        # Plot with better styling
        plt.figure(figsize=(12, 12))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                            c=labels, cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Digit Class')
        plt.title('Latent Space Visualization (t-SNE)', pad=20)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=plt.cm.tab10(i/10),
                                    label=f'Digit {i}', markersize=10)
                          for i in range(10)]
        plt.legend(handles=legend_elements, loc='center left',
                  bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig('latent_space.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Latent space visualization saved.")
        
    except Exception as e:
        print(f"Error in latent space visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # Print CUDA information
    print_cuda_info()
    
    # Hyperparameters
    batch_size = 128
    epochs = 20
    latent_dim = 16
    
    # CUDA setup with benchmark for potential speed improvement
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("WARNING: CUDA is not available. Running on CPU will be much slower!")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    
    # Use pin_memory for faster data transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4 if device.type == 'cuda' else 0  # Use multiple workers for data loading
    )
    
    # Initialize model and optimizer
    model = AdaptiveConvVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Print model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}\n")
    
    try:
        # Train the model
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            
            # Generate and save sample images
            if epoch % 2 == 0:
                with torch.no_grad():
                    # Clear cache before generation
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Generate samples
                    sample = generate_samples(model, device)
                    
                    # Plot generated samples
                    plt.figure(figsize=(10, 4))
                    for i in range(10):
                        plt.subplot(2, 5, i + 1)
                        plt.imshow(sample[i].cpu().squeeze(), cmap='gray')
                        plt.axis('off')
                    plt.suptitle(f'Generated Samples (Epoch {epoch})')
                    plt.savefig(f'vae_samples_epoch_{epoch}.png')
                    plt.close()
                    
                    # Save model with additional info
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'layer_sizes': model.layer_sizes,
                        'masks': model.masks,
                        'device': device.type,
                    }, 'vae_model.pt')
                    
                    # Print current GPU memory usage
                    if device.type == 'cuda':
                        print(f'GPU Memory after epoch {epoch}: {torch.cuda.memory_allocated() / 1024**2:.1f}MB')
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'layer_sizes': model.layer_sizes,
            'masks': model.masks,
            'device': device.type,
        }, 'vae_model_interrupted.pt')
        
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print("\nGenerating interpolations...")
    interpolate_digits(model, device, 1, 7)
    interpolate_digits(model, device, 0, 6)
    interpolate_digits(model, device, 3, 8)
    
    print("\nVisualizing latent space...")
    visualize_latent_space(model, device)
    
    print("\nTraining complete! Check the generated images:")
    print("1. Sample digits: vae_samples_epoch_*.png")
    print("2. Interpolations: interpolation_*to*.png")
    print("3. Latent space visualization: latent_space.png")
    
    # Final memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"\nFinal GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")

if __name__ == '__main__':
    main() 