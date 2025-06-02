import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class Encoder(nn.Module):
    """
    Conditional VAE Encoder for gene expression data
    """
    
    def __init__(self, input_dim: int, condition_dim: int, latent_dim: int, hidden_dims: list = None):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim  # 22283 genes
        self.condition_dim = condition_dim  # 5 classes
        self.latent_dim = latent_dim  # latent space dimension
        
        # Default hidden dimensions for gene expression data
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512, 256]
        
        self.hidden_dims = hidden_dims
        
        # Input layer combines gene expression + condition
        combined_input_dim = input_dim + condition_dim
        
        # Build encoder layers
        layers = []
        prev_dim = combined_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder
        
        Args:
            x: Gene expression data [batch_size, input_dim]
            condition: One-hot encoded conditions [batch_size, condition_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        # Concatenate input with condition
        combined_input = torch.cat([x, condition], dim=1)
        
        # Pass through encoder
        encoded = self.encoder(combined_input)
        
        # Get mean and log variance
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        return mu, logvar

class Decoder(nn.Module):
    """
    Conditional VAE Decoder for gene expression data
    """
    
    def __init__(self, latent_dim: int, condition_dim: int, output_dim: int, hidden_dims: list = None):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim  # 22283 genes
        
        # Default hidden dimensions (reverse of encoder)
        if hidden_dims is None:
            hidden_dims = [256, 512, 1024, 2048]
        
        self.hidden_dims = hidden_dims
        
        # Input layer combines latent vector + condition
        combined_input_dim = latent_dim + condition_dim
        
        # Build decoder layers
        layers = []
        prev_dim = combined_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Output layer (no activation for gene expression)
        self.fc_output = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            condition: One-hot encoded conditions [batch_size, condition_dim]
            
        Returns:
            reconstructed: Reconstructed gene expression [batch_size, output_dim]
        """
        # Concatenate latent vector with condition
        combined_input = torch.cat([z, condition], dim=1)
        
        # Pass through decoder
        decoded = self.decoder(combined_input)
        
        # Generate output (no activation for gene expression)
        reconstructed = self.fc_output(decoded)
        
        return reconstructed

class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for Leukemia Gene Expression Data
    """
    
    def __init__(self, 
                 input_dim: int = 22283,
                 condition_dim: int = 5,
                 latent_dim: int = 256,
                 encoder_hidden_dims: list = None,
                 decoder_hidden_dims: list = None):
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Initialize encoder and decoder
        self.encoder = Encoder(
            input_dim=input_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            output_dim=input_dim,
            hidden_dims=decoder_hidden_dims
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
            
        Returns:
            z: Sampled latent vector [batch_size, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through cVAE
        
        Args:
            x: Gene expression data [batch_size, input_dim]
            condition: One-hot encoded conditions [batch_size, condition_dim]
            
        Returns:
            Dictionary containing reconstruction, mu, logvar, and latent vector
        """
        # Encode
        mu, logvar = self.encoder(x, condition)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z, condition)
        
        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space
        
        Args:
            x: Gene expression data [batch_size, input_dim]
            condition: One-hot encoded conditions [batch_size, condition_dim]
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        return self.encoder(x, condition)
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to gene expression
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            condition: One-hot encoded conditions [batch_size, condition_dim]
            
        Returns:
            reconstructed: Reconstructed gene expression
        """
        return self.decoder(z, condition)
    
    def generate(self, condition: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Generate new samples conditioned on given classes
        
        Args:
            condition: One-hot encoded conditions [n_conditions, condition_dim]
            n_samples: Number of samples to generate per condition
            
        Returns:
            generated: Generated gene expression samples
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # Expand conditions if needed
            if condition.dim() == 1:
                condition = condition.unsqueeze(0)
            
            # Repeat conditions for multiple samples
            condition = condition.repeat_interleave(n_samples, dim=0)
            batch_size = condition.size(0)
            
            # Sample from prior distribution
            z = torch.randn(batch_size, self.latent_dim, device=device)
            
            # Generate samples
            generated = self.decode(z, condition)
            
        return generated
    
    def compute_loss(self, x: torch.Tensor, condition: torch.Tensor, 
                    beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute cVAE loss (reconstruction + KL divergence)
        
        Args:
            x: Input gene expression data
            condition: One-hot encoded conditions
            beta: Weight for KL divergence term (β-VAE)
            
        Returns:
            Dictionary containing total loss and individual components
        """
        # Forward pass
        outputs = self.forward(x, condition)
        reconstructed = outputs['reconstructed']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss (MSE for gene expression)
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta': beta
        }

def create_condition_tensor(labels: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    """
    Convert class labels to one-hot encoded tensors
    
    Args:
        labels: Class labels [batch_size]
        n_classes: Total number of classes
        device: Device to place tensor on
        
    Returns:
        condition: One-hot encoded conditions [batch_size, n_classes]
    """
    condition = torch.zeros(len(labels), n_classes, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(1)
    condition.scatter_(1, labels_tensor, 1)
    return condition

def test_cvae_architecture():
    """
    Test the cVAE architecture with dummy data
    """
    print("Testing cVAE Architecture...")
    
    # Model parameters
    input_dim = 22283
    condition_dim = 5
    latent_dim = 256
    batch_size = 8
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalVAE(
        input_dim=input_dim,
        condition_dim=condition_dim,
        latent_dim=latent_dim
    ).to(device)
    
    print(f"Model created on device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data
    x = torch.randn(batch_size, input_dim, device=device)
    labels = np.random.randint(0, condition_dim, size=batch_size)
    condition = create_condition_tensor(labels, condition_dim, device)
    
    print(f"Input shape: {x.shape}")
    print(f"Condition shape: {condition.shape}")
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(x, condition)
        print(f"Reconstruction shape: {outputs['reconstructed'].shape}")
        print(f"Latent mu shape: {outputs['mu'].shape}")
        print(f"Latent logvar shape: {outputs['logvar'].shape}")
        
        # Test loss computation
        loss_dict = model.compute_loss(x, condition)
        print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
        print(f"Reconstruction loss: {loss_dict['recon_loss'].item():.4f}")
        print(f"KL loss: {loss_dict['kl_loss'].item():.4f}")
        
        # Test generation
        test_condition = create_condition_tensor([0, 1, 2], condition_dim, device)
        generated = model.generate(test_condition, n_samples=2)
        print(f"Generated samples shape: {generated.shape}")
    
    print("✅ cVAE architecture test completed successfully!")
    return model

if __name__ == "__main__":
    # Test the architecture
    model = test_cvae_architecture() 