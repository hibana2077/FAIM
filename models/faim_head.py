"""
Finsler-α Information Manifold Classification Head

This module implements the FAIM-Head as described in the research documentation,
providing a direct replacement for traditional Linear/ClassifierHead in PyTorch models.

The FAIM-Head uses Randers-type Finsler metric for ultra-fine-grained visual classification,
enabling directional-dependent distance metrics that can amplify intra-class variations
while maintaining or increasing inter-class distances.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FAIMHead(nn.Module):
    """Finsler-α Information Manifold Classification Head.
    
    A novel classification head that replaces traditional Linear layers with a 
    Finsler manifold-based distance computation. This enables directional-dependent
    metrics that are particularly effective for ultra-fine-grained visual classification.
    
    The head computes geodesic distances in a Randers-type Finsler space:
    F_x(v) = sqrt(v^T Σ v) + λ |β^T v|
    
    where:
    - Σ is a positive definite Fisher information matrix approximation
    - β is a directional 1-form vector
    - λ controls the weight of the directional component
    
    Args:
        in_features (int): Input feature dimension d
        num_classes (int): Number of classes C  
        lambda_init (float): Initial value for λ parameter. Default: 0.1
        scale_init (float): Initial value for temperature γ. Default: 10.0
        full_sigma (bool): If True, learn full Σ matrix; otherwise diagonal only. Default: False
        eps (float): Small constant for numerical stability. Default: 1e-6
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        lambda_init: float = 0.1,
        scale_init: float = 10.0,
        full_sigma: bool = False,
        eps: float = 1e-6
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.full_sigma = full_sigma
        self.eps = eps
        
        # Class prototypes μ_k [C, d]
        self.mu = nn.Parameter(torch.randn(num_classes, in_features))
        
        # Directional 1-form β [d] 
        self.beta = nn.Parameter(torch.randn(in_features))
        
        # Randers coefficient λ (scalar)
        self.lmbda = nn.Parameter(torch.tensor(lambda_init))
        
        # Temperature parameter γ for softmax scaling
        self.scale = nn.Parameter(torch.tensor(scale_init))
        
        # Σ matrix parameterization
        if full_sigma:
            # L is lower triangular (including diagonal) => Σ = L L^T + εI
            self.L = nn.Parameter(torch.eye(in_features))
        else:
            # Learn only diagonal elements (positive)
            self.log_sigma_diag = nn.Parameter(torch.zeros(in_features))
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with appropriate distributions."""
        # Initialize prototypes with Xavier normal
        nn.init.xavier_normal_(self.mu)
        
        # Initialize beta with small values
        nn.init.normal_(self.beta, mean=0.0, std=0.1)
        
        # Initialize L matrix to be close to identity
        if self.full_sigma:
            nn.init.eye_(self.L)
            # Add small random perturbation
            with torch.no_grad():
                self.L.add_(torch.tril(torch.randn_like(self.L) * 0.01))
    
    def _sigma(self) -> torch.Tensor:
        """Compute the positive definite Σ matrix.
        
        Returns:
            torch.Tensor: Positive definite matrix [d, d]
        """
        if self.full_sigma:
            # Ensure lower triangular and compute Σ = L L^T
            L = torch.tril(self.L)
            sigma = L @ L.T
        else:
            # Diagonal matrix with positive elements
            sigma = torch.diag(self.log_sigma_diag.exp())
        
        # Add small identity for numerical stability
        eye = torch.eye(
            self.in_features, 
            device=sigma.device, 
            dtype=sigma.dtype
        )
        return sigma + self.eps * eye
    
    @staticmethod
    def _smooth_abs(z: torch.Tensor, eps: float) -> torch.Tensor:
        """Smooth approximation of absolute value to ensure differentiability.
        
        |z| ≈ sqrt(z² + ε)
        
        Args:
            z: Input tensor
            eps: Small constant for smoothing
            
        Returns:
            Smoothed absolute value
        """
        return torch.sqrt(z * z + eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing FAIM-based logits.
        
        Args:
            x: Input features [B, d]
            
        Returns:
            logits: Class logits [B, C] computed as -γ * d_F(x, μ_k)
        """
        batch_size = x.size(0)
        
        # Get current Σ matrix [d, d]
        sigma = self._sigma()
        
        # Compute differences: x - μ_k for all classes [B, C, d]
        diff = x.unsqueeze(1) - self.mu  # [B, 1, d] - [C, d] -> [B, C, d]
        
        # Compute quadratic form: (x - μ_k)^T Σ (x - μ_k) [B, C]
        quad = torch.einsum('bcd,df,bcf->bc', diff, sigma, diff)
        
        # Compute directional term: β · (x - μ_k) [B, C]
        beta_dot = torch.einsum('d,bcd->bc', self.beta, diff)
        
        # Compute FAIM geodesic distance [B, C]
        riemannian_term = torch.sqrt(quad + self.eps)
        directional_term = self.lmbda * self._smooth_abs(beta_dot, self.eps)
        d_f = riemannian_term + directional_term
        
        # Convert to logits with temperature scaling
        logits = -self.scale * d_f
        
        return logits
    
    def get_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw FAIM distances without temperature scaling.
        
        Args:
            x: Input features [B, d]
            
        Returns:
            distances: FAIM distances to each class prototype [B, C]
        """
        with torch.no_grad():
            sigma = self._sigma()
            diff = x.unsqueeze(1) - self.mu
            quad = torch.einsum('bcd,df,bcf->bc', diff, sigma, diff)
            beta_dot = torch.einsum('d,bcd->bc', self.beta, diff)
            
            riemannian_term = torch.sqrt(quad + self.eps)
            directional_term = self.lmbda * self._smooth_abs(beta_dot, self.eps)
            distances = riemannian_term + directional_term
            
        return distances
    
    def get_sigma_eigenvalues(self) -> torch.Tensor:
        """Get eigenvalues of the Σ matrix for analysis.
        
        Returns:
            eigenvalues: Eigenvalues of Σ matrix
        """
        with torch.no_grad():
            sigma = self._sigma()
            eigenvalues = torch.linalg.eigvals(sigma).real
        return eigenvalues
    
    def get_prototype_distances(self) -> torch.Tensor:
        """Compute pairwise distances between class prototypes.
        
        Returns:
            pairwise_distances: Matrix of distances between prototypes [C, C]
        """
        with torch.no_grad():
            sigma = self._sigma()
            
            # Compute all pairwise differences [C, C, d]
            mu_expanded = self.mu.unsqueeze(1)  # [C, 1, d]
            diff = mu_expanded - self.mu  # [C, C, d]
            
            # Compute quadratic forms [C, C]
            quad = torch.einsum('ccd,df,ccf->cc', diff, sigma, diff)
            
            # Compute directional terms [C, C]
            beta_dot = torch.einsum('d,ccd->cc', self.beta, diff)
            
            # Compute distances
            riemannian_term = torch.sqrt(quad + self.eps)
            directional_term = self.lmbda * self._smooth_abs(beta_dot, self.eps)
            distances = riemannian_term + directional_term
            
        return distances
    
    def freeze_metric_params(self):
        """Freeze metric parameters (Σ, β, λ) for warm-up training."""
        if self.full_sigma:
            self.L.requires_grad_(False)
        else:
            self.log_sigma_diag.requires_grad_(False)
        self.beta.requires_grad_(False)
        self.lmbda.requires_grad_(False)
    
    def unfreeze_metric_params(self):
        """Unfreeze metric parameters for metric tuning phase."""
        if self.full_sigma:
            self.L.requires_grad_(True)
        else:
            self.log_sigma_diag.requires_grad_(True)
        self.beta.requires_grad_(True)
        self.lmbda.requires_grad_(True)
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f'in_features={self.in_features}, num_classes={self.num_classes}, '
            f'lambda={self.lmbda.item():.3f}, scale={self.scale.item():.3f}, '
            f'full_sigma={self.full_sigma}'
        )


class FAIMContrastiveLoss(nn.Module):
    """Contrastive loss using FAIM geodesic distances.
    
    This loss can be used as an alternative or supplement to the standard
    CrossEntropy loss, particularly effective for metric learning scenarios.
    
    Args:
        margin (float): Margin for contrastive loss. Default: 1.0
        temperature (float): Temperature for InfoNCE-style computation. Default: 0.1
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self, 
        distances: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss using FAIM distances.
        
        Args:
            distances: FAIM distances [B, C]
            targets: Target class indices [B]
            
        Returns:
            loss: Contrastive loss value
        """
        batch_size, num_classes = distances.shape
        
        # Get positive distances (distances to true class)
        pos_distances = distances.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Create negative mask (all classes except true class)
        neg_mask = torch.ones_like(distances).bool()
        neg_mask.scatter_(1, targets.unsqueeze(1), False)
        
        # Get negative distances
        neg_distances = distances.masked_select(neg_mask).view(batch_size, -1)
        
        # Compute contrastive loss: margin + d(anchor, pos) - d(anchor, neg)
        # We want to minimize positive distances and maximize negative distances
        pos_loss = pos_distances.mean()
        neg_loss = torch.relu(self.margin - neg_distances).mean()
        
        return pos_loss + neg_loss


def create_faim_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    lambda_init: float = 0.1,
    scale_init: float = 10.0,
    full_sigma: bool = False
) -> nn.Module:
    """Create a model with FAIM-Head replacing the original classifier.
    
    Args:
        model_name: Name of the timm model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        lambda_init: Initial lambda parameter for FAIM-Head
        scale_init: Initial scale parameter for FAIM-Head
        full_sigma: Whether to use full Sigma matrix
        
    Returns:
        model: Model with FAIM-Head
    """
    import timm
    
    # Create backbone without classifier
    model = timm.create_model(
        model_name, 
        pretrained=pretrained, 
        num_classes=0  # No default head
    )
    
    # Get feature dimension
    feat_dim = model.num_features
    
    # Add FAIM-Head
    faim_head = FAIMHead(
        in_features=feat_dim,
        num_classes=num_classes,
        lambda_init=lambda_init,
        scale_init=scale_init,
        full_sigma=full_sigma
    )
    
    # Replace the classifier/head
    if hasattr(model, 'head'):
        model.head = faim_head
    elif hasattr(model, 'classifier'):
        model.classifier = faim_head
    elif hasattr(model, 'fc'):
        model.fc = faim_head
    else:
        # If no standard head attribute found, add head
        model.head = faim_head
    
    return model


if __name__ == "__main__":
    # Example usage and testing
    print("Testing FAIM-Head implementation...")
    
    # Test parameters
    batch_size = 8
    feat_dim = 512
    num_classes = 100
    
    # Create FAIM-Head
    head = FAIMHead(
        in_features=feat_dim,
        num_classes=num_classes,
        lambda_init=0.1,
        full_sigma=False
    )
    
    # Test forward pass
    x = torch.randn(batch_size, feat_dim)
    logits = head(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Test loss computation
    targets = torch.randint(0, num_classes, (batch_size,))
    loss = F.cross_entropy(logits, targets)
    print(f"CrossEntropy loss: {loss.item():.3f}")
    
    # Test distances
    distances = head.get_distances(x)
    print(f"Distance shape: {distances.shape}")
    print(f"Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
    
    # Test model creation
    try:
        import timm
        model = create_faim_model('resnet18', num_classes=10, pretrained=False)
        print(f"Created model with FAIM-Head: {type(model.head)}")
        
        # Test full model
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        print(f"Full model output shape: {output.shape}")
        
    except ImportError:
        print("timm not available, skipping model creation test")
    
    print("FAIM-Head testing completed successfully!")
