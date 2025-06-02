import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import pickle
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from cvae_model import ConditionalVAE, create_condition_tensor
from data_preprocessing import LeukemiaDataPreprocessor

class CVAETrainer:
    """
    Trainer class for Conditional VAE on Leukemia Gene Expression Data
    """
    
    def __init__(self, 
                 model: ConditionalVAE,
                 device: torch.device = None,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 beta_schedule: str = 'constant',
                 beta_start: float = 1.0,
                 beta_end: float = 1.0):
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Beta scheduling for β-VAE
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': [],
            'beta_values': []
        }
        
        print(f"cVAE Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def get_beta(self, epoch: int, total_epochs: int) -> float:
        """
        Get beta value for current epoch (β-VAE scheduling)
        """
        if self.beta_schedule == 'constant':
            return self.beta_start
        elif self.beta_schedule == 'linear':
            return self.beta_start + (self.beta_end - self.beta_start) * (epoch / total_epochs)
        elif self.beta_schedule == 'cyclical':
            cycle_length = total_epochs // 4
            cycle_progress = (epoch % cycle_length) / cycle_length
            return self.beta_start + (self.beta_end - self.beta_start) * cycle_progress
        else:
            return self.beta_start
    
    def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        Train for one epoch
        """
        self.model.train()
        epoch_losses = {'total': [], 'recon': [], 'kl': []}
        
        # Get beta for this epoch
        beta = self.get_beta(epoch, total_epochs)
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs}')
        
        for batch_idx, (x, labels) in enumerate(progress_bar):
            x = x.to(self.device)
            labels = labels.to(self.device)
            
            # Create condition tensor
            condition = create_condition_tensor(
                labels.cpu().numpy(), 
                self.model.condition_dim, 
                self.device
            )
            
            # Forward pass and compute loss
            loss_dict = self.model.compute_loss(x, condition, beta=beta)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Record losses
            epoch_losses['total'].append(loss_dict['total_loss'].item())
            epoch_losses['recon'].append(loss_dict['recon_loss'].item())
            epoch_losses['kl'].append(loss_dict['kl_loss'].item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'Recon': f"{loss_dict['recon_loss'].item():.4f}",
                'KL': f"{loss_dict['kl_loss'].item():.4f}",
                'β': f"{beta:.3f}"
            })
        
        # Calculate average losses
        avg_losses = {
            'total_loss': np.mean(epoch_losses['total']),
            'recon_loss': np.mean(epoch_losses['recon']),
            'kl_loss': np.mean(epoch_losses['kl']),
            'beta': beta
        }
        
        return avg_losses
    
    def validate_epoch(self, dataloader: DataLoader, beta: float) -> Dict[str, float]:
        """
        Validate for one epoch
        """
        self.model.eval()
        epoch_losses = {'total': [], 'recon': [], 'kl': []}
        
        with torch.no_grad():
            for x, labels in dataloader:
                x = x.to(self.device)
                labels = labels.to(self.device)
                
                # Create condition tensor
                condition = create_condition_tensor(
                    labels.cpu().numpy(), 
                    self.model.condition_dim, 
                    self.device
                )
                
                # Compute loss
                loss_dict = self.model.compute_loss(x, condition, beta=beta)
                
                # Record losses
                epoch_losses['total'].append(loss_dict['total_loss'].item())
                epoch_losses['recon'].append(loss_dict['recon_loss'].item())
                epoch_losses['kl'].append(loss_dict['kl_loss'].item())
        
        # Calculate average losses
        avg_losses = {
            'total_loss': np.mean(epoch_losses['total']),
            'recon_loss': np.mean(epoch_losses['recon']),
            'kl_loss': np.mean(epoch_losses['kl']),
            'beta': beta
        }
        
        return avg_losses
    
    def train(self, 
              train_data: Dict[str, Any],
              val_data: Dict[str, Any] = None,
              epochs: int = 100,
              batch_size: int = 8,
              save_dir: str = 'models',
              save_every: int = 20,
              early_stopping_patience: int = 20) -> Dict[str, List[float]]:
        """
        Complete training loop
        """
        print("=" * 60)
        print("STARTING cVAE TRAINING")
        print("=" * 60)
        
        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_data['features']),
            torch.LongTensor(train_data['labels'])
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_data is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(val_data['features']),
                torch.LongTensor(val_data['labels'])
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training samples: {len(train_dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Number of epochs: {epochs}")
        
        # Training loop
        for epoch in range(epochs):
            # Train
            train_losses = self.train_epoch(train_loader, epoch, epochs)
            
            # Validate
            val_losses = None
            if val_loader:
                val_losses = self.validate_epoch(val_loader, train_losses['beta'])
            
            # Record history
            self.history['train_loss'].append(train_losses['total_loss'])
            self.history['train_recon_loss'].append(train_losses['recon_loss'])
            self.history['train_kl_loss'].append(train_losses['kl_loss'])
            self.history['beta_values'].append(train_losses['beta'])
            
            if val_losses:
                self.history['val_loss'].append(val_losses['total_loss'])
                self.history['val_recon_loss'].append(val_losses['recon_loss'])
                self.history['val_kl_loss'].append(val_losses['kl_loss'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train - Total: {train_losses['total_loss']:.4f}, "
                  f"Recon: {train_losses['recon_loss']:.4f}, "
                  f"KL: {train_losses['kl_loss']:.4f}")
            
            if val_losses:
                print(f"  Val   - Total: {val_losses['total_loss']:.4f}, "
                      f"Recon: {val_losses['recon_loss']:.4f}, "
                      f"KL: {val_losses['kl_loss']:.4f}")
                
                # Early stopping check
                if val_losses['total_loss'] < best_val_loss:
                    best_val_loss = val_losses['total_loss']
                    patience_counter = 0
                    # Save best model
                    self.save_model(os.path.join(save_dir, 'best_cvae_model.pth'))
                    print(f"  💾 Best model saved (Val Loss: {best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"  🛑 Early stopping triggered (patience: {early_stopping_patience})")
                        break
            
            # Save model periodically
            if (epoch + 1) % save_every == 0:
                self.save_model(os.path.join(save_dir, f'cvae_model_epoch_{epoch+1}.pth'))
        
        # Save final model
        self.save_model(os.path.join(save_dir, 'final_cvae_model.pth'))
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        
        return self.history
    
    def save_model(self, path: str):
        """
        Save model state
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'condition_dim': self.model.condition_dim,
                'latent_dim': self.model.latent_dim
            },
            'history': self.history
        }, path)
    
    def load_model(self, path: str):
        """
        Load model state
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Model loaded from {path}")
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('cVAE Training History', fontsize=16)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Total Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        if self.history['val_loss']:
            axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction Loss
        axes[0, 1].plot(epochs, self.history['train_recon_loss'], 'b-', label='Train')
        if self.history['val_recon_loss']:
            axes[0, 1].plot(epochs, self.history['val_recon_loss'], 'r-', label='Validation')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # KL Divergence
        axes[1, 0].plot(epochs, self.history['train_kl_loss'], 'b-', label='Train')
        if self.history['val_kl_loss']:
            axes[1, 0].plot(epochs, self.history['val_kl_loss'], 'r-', label='Validation')
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Beta Values
        axes[1, 1].plot(epochs, self.history['beta_values'], 'g-', label='β value')
        axes[1, 1].set_title('Beta Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('β')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved: {save_path}")
        
        plt.show()

def main():
    """
    Main function to demonstrate cVAE training
    """
    print("Loading preprocessed data...")
    
    # Load preprocessor and prepare data
    preprocessor = LeukemiaDataPreprocessor()
    df = preprocessor.load_data('datasets/raw_data.csv')
    processed_data = preprocessor.prepare_data_for_cvae(df)
    
    # Model configuration
    model_config = {
        'input_dim': processed_data['metadata']['n_genes'],
        'condition_dim': processed_data['metadata']['n_classes'],
        'latent_dim': 256,
        'encoder_hidden_dims': [2048, 1024, 512, 256],
        'decoder_hidden_dims': [256, 512, 1024, 2048]
    }
    
    # Create model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalVAE(**model_config)
    
    trainer = CVAETrainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        weight_decay=1e-5,
        beta_schedule='linear',
        beta_start=0.1,
        beta_end=1.0
    )
    
    # Training configuration
    training_config = {
        'epochs': 100,
        'batch_size': 8,
        'save_dir': 'models',
        'save_every': 20,
        'early_stopping_patience': 20
    }
    
    # Start training
    history = trainer.train(
        train_data=processed_data['train'],
        val_data=processed_data.get('validation'),
        **training_config
    )
    
    # Plot and save training history
    trainer.plot_training_history('visualizations/cvae_training_history.png')
    
    return trainer, history

if __name__ == "__main__":
    trainer, history = main() 