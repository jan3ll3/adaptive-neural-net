import torch
from transformers import GPT2Tokenizer
from neural_network_LLM import AdaptiveTransformer
import argparse
import os
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
import textwrap

class ChatInterface:
    def __init__(self, model_path, device='cuda'):
        self.console = Console()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Only set pad_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.console.print("Loading model...", style="yellow")
        self.model = self._load_model(model_path)
        self.console.print("Model loaded!", style="green")
        
        # Set max length for generation
        self.max_length = 512
        self.max_new_tokens = 100
        self.temperature = 0.7  # Default temperature
        
    def _load_model(self, model_path):
        # Load model configuration from saved state
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Create model with same configuration
        model = AdaptiveTransformer(
            vocab_size=50257,  # Standard GPT2 vocab size
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            max_seq_length=512
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    def generate_response(self, prompt, temperature=None):
        # Use instance temperature if none provided
        if temperature is None:
            temperature = self.temperature
            
        # Format prompt
        input_text = f"{prompt}"
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, 
                              max_length=self.max_length)
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = []
            
            for _ in range(self.max_new_tokens):
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                top_k = 50
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                top_p = 0.9
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if we predict the end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                generated_ids.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()
    
    def chat_loop(self):
        self.console.print("\n=== Chat Interface ===", style="bold blue")
        self.console.print("Type 'quit' to exit, 'help' for commands\n")
        
        while True:
            try:
                user_input = Prompt.ask("[bold blue]You[/bold blue]")
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower().startswith('temp='):
                    try:
                        temp = float(user_input.split('=')[1])
                        self.temperature = max(0.1, min(2.0, temp))
                        self.console.print(f"Temperature set to {self.temperature}", style="yellow")
                    except:
                        self.console.print("Invalid temperature value", style="red")
                    continue
                
                # Generate response
                self.console.print("\n[bold green]Assistant[/bold green]: ", end="")
                with self.console.status("[yellow]Thinking...[/yellow]"):
                    response = self.generate_response(user_input)
                
                # Print wrapped response
                wrapped_response = textwrap.fill(response, width=80)
                self.console.print(wrapped_response + "\n")
                
            except KeyboardInterrupt:
                self.console.print("\nUse 'quit' to exit properly", style="yellow")
            except Exception as e:
                self.console.print(f"\nError: {str(e)}", style="red")
    
    def show_help(self):
        help_text = """
        Available commands:
        - quit: Exit the chat
        - help: Show this help message
        - temp=X: Set temperature (0.1-2.0) for generation
        
        Generation parameters:
        - Temperature: Controls randomness (lower = more focused)
        - Max new tokens: {self.max_new_tokens}
        - Top-p: 0.9 (nucleus sampling)
        - Top-k: 50 (diversity control)
        """
        self.console.print(Markdown(textwrap.dedent(help_text)))

def main():
    parser = argparse.ArgumentParser(description="Chat interface for the trained model")
    parser.add_argument('--model_path', type=str, default='outputs/best_model.pt',
                      help='Path to the trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to run the model on (cuda/cpu)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    chat = ChatInterface(args.model_path, args.device)
    chat.chat_loop()

if __name__ == "__main__":
    main() 