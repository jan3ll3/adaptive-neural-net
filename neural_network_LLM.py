import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, max_seq_length=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.max_seq_length = max_seq_length
        
        # Initialize with learnable parameters
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)
        
        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, hidden_size))
        
        # Active head mask (for pruning heads)
        self.register_buffer('active_heads', torch.ones(num_heads, dtype=torch.bool))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.query, self.key, self.value, self.out]:
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.shape
        
        # Add positional embeddings
        positions = self.pos_embedding[:, :seq_length, :]
        x = x + positions
        
        # Linear projections and split into heads
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        
        # Apply head masking
        q = q * self.active_heads.view(1, -1, 1, 1)
        k = k * self.active_heads.view(1, -1, 1, 1)
        v = v * self.active_heads.view(1, -1, 1, 1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        # Apply attention mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))  # Note the ~ to invert boolean mask
        
        # Compute attention weights and apply dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Store for analysis
        self.last_attention = attn_weights.detach()
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Combine heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        output = self.out(context)
        
        if return_attention:
            return output, attn_weights
        return output
    
    def prune_heads(self, threshold=0.1):
        """Prune attention heads based on their average attention weight magnitude"""
        with torch.no_grad():
            # Compute head importance based on weight magnitudes
            q_weights = torch.norm(self.query.weight.view(self.num_heads, -1), dim=1)
            k_weights = torch.norm(self.key.weight.view(self.num_heads, -1), dim=1)
            v_weights = torch.norm(self.value.weight.view(self.num_heads, -1), dim=1)
            
            head_importance = (q_weights + k_weights + v_weights) / 3
            max_importance = torch.max(head_importance)
            
            # Update active heads mask
            self.active_heads.data = (head_importance > (threshold * max_importance))
            
            return torch.sum(self.active_heads).item()

class AdaptiveTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, activation='gelu'):
        super().__init__()
        self.attention = AdaptiveAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # FFN with residual
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        
        return x
    
    def prune_heads(self, threshold=0.1):
        """Prune attention heads in this block"""
        return self.attention.prune_heads(threshold)

class AdaptiveTransformer(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 hidden_size=768, 
                 num_layers=12, 
                 num_heads=12, 
                 max_seq_length=1024,
                 dropout=0.1,
                 activation='gelu'):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_heads = num_heads
        
        # Track layer sizes and growth limits
        self.layer_sizes = {
            'hidden': hidden_size,
            'heads': num_heads,
            'intermediate': hidden_size * 4
        }
        
        self.max_sizes = {
            'hidden': hidden_size * 2,
            'heads': num_heads * 2,
            'intermediate': hidden_size * 8
        }
        
        # Store activations for analysis
        self.last_activations = None
        self.activation_stats = {}
        self.growth_cooldown = 0
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdaptiveTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Output head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_length = input_ids.shape
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), device=input_ids.device, dtype=torch.bool),
            diagonal=1
        )
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) & ~causal_mask
        
        # Embed tokens
        x = self.token_embedding(input_ids)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Apply final norm and language model head
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # If labels are provided, compute loss
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                          shift_labels.view(-1))
        
        return type('TransformerOutput', (), {'loss': loss, 'logits': logits})()
    
    def update_topology(self):
        """Update network topology by pruning and potentially growing"""
        # Prune heads
        total_pruned = self.prune_heads(threshold=0.1)
        
        # Update activation statistics
        for i, block in enumerate(self.blocks):
            if hasattr(block.attention, 'last_attention'):
                self.update_activation_stats(f'block_{i}_attention', block.attention.last_attention)
        
        # Consider growth if cooldown is over
        if self.growth_cooldown == 0:
            growth_happened = self.consider_growth()
            if growth_happened:
                self.growth_cooldown = 1000  # Reset cooldown
        else:
            self.growth_cooldown -= 1
    
    def prune_heads(self, threshold=0.1):
        """Prune attention heads across all layers"""
        total_pruned = 0
        for block in self.blocks:
            total_pruned += block.prune_heads(threshold)
        return total_pruned
    
    def update_activation_stats(self, name, tensor):
        """Update running statistics for activations"""
        if name not in self.activation_stats:
            self.activation_stats[name] = {
                'mean': torch.zeros_like(tensor.mean(dim=0)),
                'var': torch.zeros_like(tensor.var(dim=0)),
                'count': 0
            }
        
        stats = self.activation_stats[name]
        curr_mean = tensor.mean(dim=0)
        curr_var = tensor.var(dim=0)
        n = stats['count']
        m = tensor.size(0)
        
        # Update running statistics
        new_mean = (stats['mean'] * n + curr_mean * m) / (n + m)
        new_var = (stats['var'] * n + curr_var * m) / (n + m)
        
        stats['mean'] = new_mean
        stats['var'] = new_var
        stats['count'] += m
    
    def consider_growth(self):
        """Consider growing the network based on activation statistics"""
        growth_happened = False
        
        # Check each layer's statistics
        for name, stats in self.activation_stats.items():
            if stats['count'] < 1000:  # Need enough samples
                continue
            
            # Check if variance is high in any dimension
            high_var_dims = (stats['var'] > stats['var'].mean() * 2).sum()
            if high_var_dims > 0 and self.layer_sizes['hidden'] < self.max_sizes['hidden']:
                # Grow the layer size
                self._grow_layer(name)
                growth_happened = True
        
        return growth_happened
    
    def _grow_layer(self, layer_name):
        """Grow a specific layer"""
        # For now, just increase hidden size by 10%
        current_size = self.layer_sizes['hidden']
        new_size = min(int(current_size * 1.1), self.max_sizes['hidden'])
        
        if new_size > current_size:
            self.layer_sizes['hidden'] = new_size
            # Implementation of actual growth would go here
            # This would involve creating new weights and copying old ones
            print(f"Growing {layer_name} from {current_size} to {new_size}")
    
    def print_statistics(self):
        """Print current network statistics"""
        print("\nNetwork Statistics:")
        print(f"Layer sizes: {self.layer_sizes}")
        active_heads = sum(block.attention.active_heads.sum().item() for block in self.blocks)
        total_heads = self.num_layers * self.max_heads
        print(f"Active attention heads: {active_heads}/{total_heads}")
        print(f"Growth cooldown: {self.growth_cooldown}")