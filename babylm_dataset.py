import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import os
import glob
import random

class BabyLMDataset(Dataset):
    def __init__(self, data_path, max_length=1024, split=None, file_pattern="*.test"):
        """
        Initialize BabyLM dataset
        
        Args:
            data_path: Path to data file or directory
            max_length: Maximum sequence length for training
            split: Optional split name (train/valid/test)
            file_pattern: Pattern to match files if data_path is a directory
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_length = max_length
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Handle both single file and directory inputs
        if os.path.isfile(data_path):
            self.files = [data_path]
        else:
            # Load all text files from the directory
            pattern = os.path.join(data_path, file_pattern)
            self.files = glob.glob(pattern)
            
        if not self.files:
            raise ValueError(f"No text files found at {data_path}")
        
        print(f"Found {len(self.files)} files to process")
        for f in self.files:
            print(f"  {os.path.basename(f)}")
        
        # Load and tokenize all text
        self.examples = []
        self._load_data()
    
    def _load_data(self):
        """Load and tokenize all text data"""
        for file_path in self.files:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Tokenize the entire text
            tokens = self.tokenizer.encode(text)
            
            # Split into chunks of max_length with overlap
            stride = self.max_length // 2  # 50% overlap between chunks
            for i in range(0, len(tokens) - self.max_length + 1, stride):
                chunk = tokens[i:i + self.max_length]
                if len(chunk) == self.max_length:  # Only keep full-length chunks
                    self.examples.append(chunk)
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Convert to tensors
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)  # All tokens except last
        labels = torch.tensor(tokens[1:], dtype=torch.long)    # All tokens except first
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones_like(input_ids)
        }

def create_dataloaders(data_dir, batch_size=4, max_length=1024, num_workers=4):
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Directory containing data files
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, valid_loader
    """
    # Create dataset from all files
    dataset = BabyLMDataset(data_dir, max_length=max_length)
    
    # Split into train/valid (90/10 split)
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader

if __name__ == "__main__":
    # Test the dataset
    data_path = "text_data/test/test/simple_wiki.test"
    
    print("Testing dataset loading...")
    dataset = BabyLMDataset(data_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Test a batch
    batch = dataset[0]
    print("\nSample batch:")
    for k, v in batch.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
    
    # Decode a sample
    tokens = batch['input_ids'].tolist()
    text = dataset.tokenizer.decode(tokens)
    print("\nSample text:")
    print(text[:100] + "...") 