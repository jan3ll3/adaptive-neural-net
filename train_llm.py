import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from neural_network_LLM import AdaptiveTransformer
from babylm_dataset import create_dataloaders
import os
from tqdm import tqdm
import wandb
import math

def train(args):
    # Initialize wandb
    wandb.init(project="adaptive-transformer", config=args)
    
    # Create model
    model = AdaptiveTransformer(
        vocab_size=50257,  # GPT2 tokenizer vocab size
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_length=args.max_seq_length,
        dropout=0.1
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Create dataloaders
    train_loader, valid_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        num_workers=4
    )
    
    num_training_steps = len(train_loader) * args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    # Training loop
    best_valid_loss = float('inf')
    step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        # Training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for batch in train_pbar:
            # Move batch to device
            batch = {k: v.cuda() if torch.cuda.is_available() else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log to wandb
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0],
                'step': step
            })
            
            step += 1
            
            # Adaptive topology updates
            if step % args.update_every == 0:
                model.update_topology()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_valid_loss = 0
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Valid]")
        
        with torch.no_grad():
            for batch in valid_pbar:
                batch = {k: v.cuda() if torch.cuda.is_available() else v for k, v in batch.items()}
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                total_valid_loss += loss.item()
                valid_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        
        # Log epoch metrics
        wandb.log({
            'epoch': epoch + 1,
            'avg_train_loss': avg_train_loss,
            'avg_valid_loss': avg_valid_loss,
            'perplexity': math.exp(avg_valid_loss)
        })
        
        print(f"\nEpoch {epoch+1}")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_valid_loss:.4f}")
        print(f"Perplexity: {math.exp(avg_valid_loss):.2f}")
        
        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f"New best model saved with validation loss: {best_valid_loss:.4f}")
        
        # Print network statistics
        model.print_statistics()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--max_seq_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--update_every', type=int, default=100)
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main() 