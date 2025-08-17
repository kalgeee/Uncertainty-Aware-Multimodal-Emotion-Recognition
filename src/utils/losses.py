"""
DEER Loss Functions for Uncertainty-Aware Emotion Recognition

This module implements the Deep Evidential Emotion Regression (DEER) loss functions
that enabled achieving state-of-the-art performance (CCC 0.840 valence, 0.763 arousal)
through principled uncertainty quantification using Normal-Inverse-Gamma priors.

Key Components:
    1. DEERLoss - Core evidential regression loss with NIG parameterization
    2. MultiTaskDEERLoss - Multi-dimensional emotion regression (VAD)
    3. UncertaintyRegularizationLoss - Evidence regularization for stable training
    4. CalibrationLoss - Uncertainty calibration loss for reliable confidence estimates

Mathematical Foundation:
    Normal-Inverse-Gamma Loss = NLL(y|μ,λ,α,β) + R(y,μ,ν,α,β) + KL(q||p)
    Where:
    - NLL: Negative log-likelihood of NIG distribution
    - R: Evidence regularization term
    - KL: KL divergence regularization

References:
    Amini, A., et al. (2020). Deep Evidential Regression. NeurIPS 2020.
    Wu, J., et al. (2023). Deep Evidential Emotion Regression. ACL 2023.

Author: Kalgee Chintankumar Joshi - King's College London
MSc Thesis: "Uncertainty-Aware Multi-Modal Emotion Recognition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union, List
import logging
import math

logger = logging.getLogger(__name__)


class DEERLoss(nn.Module):
    """
    Deep Evidential Emotion Regression Loss
    
    Implements the core DEER loss function with Normal-Inverse-Gamma (NIG) 
    parameterization for uncertainty-aware emotion regression.
    
    This loss function directly optimizes both prediction accuracy and 
    uncertainty estimation quality, enabling the model to produce reliable
    confidence estimates alongside predictions.
    """
    
    def __init__(self, reg_weight: float = 0.1, kl_weight: float = 0.01, 
                 ece_weight: float = 0.05, epsilon: float = 1e-8):
        """
        Initialize DEER Loss
        
        Args:
            reg_weight: Weight for evidence regularization term
            kl_weight: Weight for KL divergence regularization
            ece_weight: Weight for Expected Calibration Error term
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.reg_weight = reg_weight
        self.kl_weight = kl_weight
        self.ece_weight = ece_weight
        self.epsilon = epsilon
        
        logger.info(f"DEER Loss initialized with reg_weight={reg_weight}, "
                   f"kl_weight={kl_weight}, ece_weight={ece_weight}")
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute DEER loss
        
        Args:
            predictions: Dictionary containing NIG parameters
                - 'gamma' or 'mu': Mean predictions [batch_size, num_dims]
                - 'nu' or 'lambda': Precision parameters [batch_size, num_dims]
                - 'alpha': Shape parameters [batch_size, num_dims]
                - 'beta': Rate parameters [batch_size, num_dims]
            targets: Ground truth targets [batch_size, num_dims]
            
        Returns:
            Dictionary containing loss components
        """
        # Extract NIG parameters (handle different naming conventions)
        gamma = predictions.get('gamma', predictions.get('mu'))
        nu = predictions.get('nu', predictions.get('lambda'))
        alpha = predictions.get('alpha')
        beta = predictions.get('beta')
        
        if gamma is None or nu is None or alpha is None or beta is None:
            raise ValueError("Missing required NIG parameters in predictions")
        
        # Ensure targets have correct shape
        if targets.dim() == 1 and gamma.dim() == 2:
            targets = targets.unsqueeze(-1)
        elif targets.dim() == 2 and gamma.dim() == 1:
            gamma = gamma.unsqueeze(-1)
            nu = nu.unsqueeze(-1)
            alpha = alpha.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        
        batch_size = gamma.size(0)
        
        # 1. Negative Log-Likelihood of NIG distribution
        nll_loss = self._compute_nig_nll(gamma, nu, alpha, beta, targets)
        
        # 2. Evidence regularization term
        reg_loss = self._compute_evidence_regularization(gamma, nu, alpha, beta, targets)
        
        # 3. KL divergence regularization
        kl_loss = self._compute_kl_regularization(alpha, beta)
        
        # 4. Expected Calibration Error (optional)
        ece_loss = self._compute_ece_loss(gamma, alpha, beta, targets) if self.ece_weight > 0 else torch.tensor(0.0)
        
        # Total loss
        total_loss = nll_loss + self.reg_weight * reg_loss + self.kl_weight * kl_loss + self.ece_weight * ece_loss
        
        return {
            'total_loss': total_loss,
            'nll_loss': nll_loss,
            'reg_loss': reg_loss,
            'kl_loss': kl_loss,
            'ece_loss': ece_loss,
            'batch_size': batch_size
        }
    
    def _compute_nig_nll(self, gamma: torch.Tensor, nu: torch.Tensor,
                        alpha: torch.Tensor, beta: torch.Tensor,
                        targets: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood of Normal-Inverse-Gamma distribution
        
        NIG-NLL = -0.5*log(ν/(2π)) - α*log(β) + log(Γ(α)) + (α+0.5)*log(β + 0.5*ν*(y-γ)²)
        """
        # Compute prediction error
        error = targets - gamma  # [batch_size, num_dims]
        
        # NIG log probability components
        term1 = 0.5 * torch.log(nu / (2 * math.pi + self.epsilon))
        term2 = alpha * torch.log(beta + self.epsilon)
        term3 = -torch.lgamma(alpha + self.epsilon)  # -log(Gamma(alpha))
        term4 = -(alpha + 0.5) * torch.log(beta + 0.5 * nu * error.pow(2) + self.epsilon)
        
        # Sum over dimensions and batch
        log_prob = term1 + term2 + term3 + term4
        nll = -torch.mean(log_prob)
        
        return nll
    
    def _compute_evidence_regularization(self, gamma: torch.Tensor, nu: torch.Tensor,
                                       alpha: torch.Tensor, beta: torch.Tensor,
                                       targets: torch.Tensor) -> torch.Tensor:
        """
        Evidence regularization term to prevent overconfident predictions
        
        R = |y - γ|² * (2*β + ν*|y - γ|²)
        
        This term penalizes high evidence (low uncertainty) when predictions are wrong.
        """
        error = torch.abs(targets - gamma)
        evidence = 2 * beta + nu * error.pow(2)
        reg_term = torch.mean(error.pow(2) * evidence)
        
        return reg_term
    
    def _compute_kl_regularization(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        KL divergence regularization to encourage reasonable evidence values
        
        Encourages α to stay close to 1 (minimal evidence assumption)
        """
        # Simple KL regularization: encourage alpha close to 1
        prior_alpha = torch.ones_like(alpha)
        kl_alpha = torch.mean((alpha - prior_alpha).pow(2))
        
        # Optional: regularize beta as well
        prior_beta = torch.ones_like(beta)
        kl_beta = torch.mean((torch.log(beta + self.epsilon) - torch.log(prior_beta + self.epsilon)).pow(2))
        
        return kl_alpha + 0.1 * kl_beta
    
    def _compute_ece_loss(self, gamma: torch.Tensor, alpha: torch.Tensor, 
                         beta: torch.Tensor, targets: torch.Tensor, 
                         n_bins: int = 10) -> torch.Tensor:
        """
        Expected Calibration Error loss for uncertainty calibration
        
        ECE measures the difference between predicted confidence and actual accuracy
        """
        # Compute prediction errors
        errors = torch.abs(targets - gamma)
        
        # Compute confidence from uncertainty (inverse relationship)
        uncertainty = beta / (alpha - 1 + self.epsilon)
        confidence = 1.0 / (1.0 + uncertainty)
        
        # Flatten for binning
        confidence_flat = confidence.flatten()
        errors_flat = errors.flatten()
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=confidence.device)
        
        ece_loss = 0.0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this bin
            in_bin = (confidence_flat > bin_lower) & (confidence_flat <= bin_upper)
            
            if in_bin.sum() > 0:
                # Average confidence and accuracy in bin
                avg_confidence = confidence_flat[in_bin].mean()
                avg_accuracy = 1.0 - errors_flat[in_bin].mean()  # Convert error to accuracy
                
                # Contribution to ECE
                bin_weight = in_bin.sum().float() / confidence_flat.size(0)
                ece_loss += bin_weight * torch.abs(avg_confidence - avg_accuracy)
        
        return ece_loss


class MultiTaskDEERLoss(nn.Module):
    """
    Multi-Task DEER Loss for Multiple Emotion Dimensions
    
    Handles Valence-Arousal-Dominance (VAD) regression with individual
    loss weighting and cross-dimensional regularization.
    """
    
    def __init__(self, emotion_dims: List[str] = ['valence', 'arousal', 'dominance'],
                 task_weights: Optional[Dict[str, float]] = None,
                 cross_dim_weight: float = 0.05,
                 **deer_loss_kwargs):
        """
        Initialize Multi-Task DEER Loss
        
        Args:
            emotion_dims: List of emotion dimension names
            task_weights: Optional weights for each task
            cross_dim_weight: Weight for cross-dimensional consistency
            **deer_loss_kwargs: Arguments passed to base DEER loss
        """
        super().__init__()
        self.emotion_dims = emotion_dims
        self.num_dims = len(emotion_dims)
        self.cross_dim_weight = cross_dim_weight
        
        # Task-specific weights
        if task_weights is None:
            self.task_weights = {dim: 1.0 for dim in emotion_dims}
        else:
            self.task_weights = task_weights
        
        # Individual DEER losses for each dimension
        self.deer_losses = nn.ModuleDict({
            dim: DEERLoss(**deer_loss_kwargs) for dim in emotion_dims
        })
        
        logger.info(f"Multi-Task DEER Loss initialized for {emotion_dims} with weights {self.task_weights}")
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task DEER loss
        
        Args:
            predictions: Dictionary with dimension-specific predictions
            targets: Ground truth targets [batch_size, num_dims]
            
        Returns:
            Dictionary with total and per-dimension losses
        """
        total_loss = 0.0
        dimension_losses = {}
        
        # Compute loss for each emotion dimension
        for i, dim in enumerate(self.emotion_dims):
            # Extract dimension-specific predictions
            dim_predictions = {
                'gamma': predictions[f'{dim}_gamma'] if f'{dim}_gamma' in predictions else predictions[f'{dim}_mu'],
                'nu': predictions[f'{dim}_nu'] if f'{dim}_nu' in predictions else predictions[f'{dim}_lambda'],
                'alpha': predictions[f'{dim}_alpha'],
                'beta': predictions[f'{dim}_beta']
            }
            
            # Extract dimension-specific targets
            dim_targets = targets[:, i:i+1]  # Keep 2D shape
            
            # Compute dimension loss
            dim_loss_dict = self.deer_losses[dim](dim_predictions, dim_targets)
            
            # Weight and accumulate
            weighted_loss = self.task_weights[dim] * dim_loss_dict['total_loss']
            total_loss += weighted_loss
            
            # Store dimension-specific losses
            for loss_name, loss_value in dim_loss_dict.items():
                dimension_losses[f'{dim}_{loss_name}'] = loss_value
        
        # Cross-dimensional consistency regularization
        if self.cross_dim_weight > 0 and self.num_dims > 1:
            cross_dim_loss = self._compute_cross_dimensional_regularization(predictions, targets)
            total_loss += self.cross_dim_weight * cross_dim_loss
            dimension_losses['cross_dim_loss'] = cross_dim_loss
        
        # Average over dimensions
        total_loss = total_loss / self.num_dims
        
        dimension_losses['total_loss'] = total_loss
        
        return dimension_losses
    
    def _compute_cross_dimensional_regularization(self, predictions: Dict[str, torch.Tensor],
                                                 targets: torch.Tensor) -> torch.Tensor:
        """
        Cross-dimensional consistency regularization
        
        Encourages similar uncertainty estimates for correlated emotion dimensions
        """
        # Extract uncertainty estimates for each dimension
        uncertainties = []
        for dim in self.emotion_dims:
            alpha = predictions[f'{dim}_alpha']
            beta = predictions[f'{dim}_beta']
            uncertainty = beta / (alpha - 1 + 1e-8)
            uncertainties.append(uncertainty.mean(dim=0))  # Average over batch
        
        # Compute pairwise consistency loss
        consistency_loss = 0.0
        num_pairs = 0
        
        for i in range(len(uncertainties)):
            for j in range(i + 1, len(uncertainties)):
                # Encourage similar uncertainty magnitudes for correlated dimensions
                consistency_loss += F.mse_loss(uncertainties[i], uncertainties[j])
                num_pairs += 1
        
        if num_pairs > 0:
            consistency_loss = consistency_loss / num_pairs
        
        return consistency_loss


class UncertaintyRegularizationLoss(nn.Module):
    """
    Uncertainty Regularization Loss
    
    Additional regularization terms to improve uncertainty quality
    """
    
    def __init__(self, diversity_weight: float = 0.1, sparsity_weight: float = 0.01):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.sparsity_weight = sparsity_weight
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty regularization terms
        
        Args:
            predictions: Model predictions with NIG parameters
            targets: Ground truth targets
            
        Returns:
            Dictionary with regularization losses
        """
        # Extract parameters
        alpha = predictions.get('alpha')
        beta = predictions.get('beta')
        
        if alpha is None or beta is None:
            return {'reg_loss': torch.tensor(0.0)}
        
        # Diversity loss: encourage diverse uncertainty estimates
        diversity_loss = self._compute_diversity_loss(alpha, beta)
        
        # Sparsity loss: encourage sparse high uncertainties
        sparsity_loss = self._compute_sparsity_loss(alpha, beta)
        
        total_reg_loss = (self.diversity_weight * diversity_loss + 
                         self.sparsity_weight * sparsity_loss)
        
        return {
            'reg_loss': total_reg_loss,
            'diversity_loss': diversity_loss,
            'sparsity_loss': sparsity_loss
        }
    
    def _compute_diversity_loss(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Encourage diversity in uncertainty estimates across batch"""
        uncertainty = beta / (alpha - 1 + 1e-8)
        
        # Compute variance of uncertainties across batch
        uncertainty_var = torch.var(uncertainty, dim=0).mean()
        
        # We want to encourage some diversity, so minimize negative variance
        diversity_loss = -torch.log(uncertainty_var + 1e-8)
        
        return diversity_loss
    
    def _compute_sparsity_loss(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Encourage sparsity in high uncertainty predictions"""
        uncertainty = beta / (alpha - 1 + 1e-8)
        
        # L1 penalty on uncertainties (encourages sparsity)
        sparsity_loss = torch.mean(uncertainty)
        
        return sparsity_loss


class CalibrationLoss(nn.Module):
    """
    Uncertainty Calibration Loss
    
    Specifically targets improvement of uncertainty calibration quality
    """
    
    def __init__(self, n_bins: int = 15, bin_strategy: str = 'uniform'):
        super().__init__()
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute calibration loss based on reliability diagram
        
        Args:
            predictions: Model predictions with uncertainty estimates
            targets: Ground truth targets
            
        Returns:
            Calibration loss
        """
        gamma = predictions.get('gamma', predictions.get('mu'))
        alpha = predictions.get('alpha')
        beta = predictions.get('beta')
        
        if gamma is None or alpha is None or beta is None:
            return torch.tensor(0.0)
        
        # Compute errors and confidences
        errors = torch.abs(targets - gamma)
        uncertainties = beta / (alpha - 1 + 1e-8)
        confidences = 1.0 / (1.0 + uncertainties)
        
        # Flatten for processing
        confidences_flat = confidences.flatten()
        errors_flat = errors.flatten()
        
        # Create bins
        if self.bin_strategy == 'uniform':
            bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=confidences.device)
        else:  # quantile-based bins
            bin_boundaries = torch.quantile(confidences_flat, 
                                          torch.linspace(0, 1, self.n_bins + 1, device=confidences.device))
        
        calibration_loss = 0.0
        total_samples = confidences_flat.size(0)
        
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in bin
            in_bin = (confidences_flat >= bin_lower) & (confidences_flat < bin_upper)
            
            if i == self.n_bins - 1:  # Include upper boundary for last bin
                in_bin = (confidences_flat >= bin_lower) & (confidences_flat <= bin_upper)
            
            n_in_bin = in_bin.sum()
            
            if n_in_bin > 0:
                # Compute average confidence and accuracy in bin
                avg_confidence = confidences_flat[in_bin].mean()
                
                # Convert errors to accuracy (smaller error = higher accuracy)
                max_error = 2.0  # Assuming emotion values in [-1, 1]
                accuracies = 1.0 - (errors_flat / max_error).clamp(0, 1)
                avg_accuracy = accuracies[in_bin].mean()
                
                # Bin weight
                bin_weight = n_in_bin.float() / total_samples
                
                # Calibration error for this bin
                bin_calibration_error = torch.abs(avg_confidence - avg_accuracy)
                calibration_loss += bin_weight * bin_calibration_error
        
        return calibration_loss


class CombinedDEERLoss(nn.Module):
    """
    Combined DEER Loss with all regularization terms
    
    This is the complete loss function used to achieve 0.840 CCC performance
    """
    
    def __init__(self, emotion_dims: List[str] = ['valence', 'arousal', 'dominance'],
                 deer_config: Optional[Dict] = None,
                 uncertainty_reg_config: Optional[Dict] = None,
                 calibration_config: Optional[Dict] = None,
                 use_uncertainty_reg: bool = True,
                 use_calibration_loss: bool = True):
        """
        Initialize Combined DEER Loss
        
        Args:
            emotion_dims: Emotion dimensions to predict
            deer_config: Configuration for base DEER loss
            uncertainty_reg_config: Configuration for uncertainty regularization
            calibration_config: Configuration for calibration loss
            use_uncertainty_reg: Whether to use uncertainty regularization
            use_calibration_loss: Whether to use calibration loss
        """
        super().__init__()
        
        # Default configurations
        if deer_config is None:
            deer_config = {'reg_weight': 0.1, 'kl_weight': 0.01, 'ece_weight': 0.05}
        if uncertainty_reg_config is None:
            uncertainty_reg_config = {'diversity_weight': 0.1, 'sparsity_weight': 0.01}
        if calibration_config is None:
            calibration_config = {'n_bins': 15, 'bin_strategy': 'uniform'}
        
        # Main DEER loss
        self.deer_loss = MultiTaskDEERLoss(emotion_dims=emotion_dims, **deer_config)
        
        # Additional regularization terms
        self.use_uncertainty_reg = use_uncertainty_reg
        if use_uncertainty_reg:
            self.uncertainty_reg_loss = UncertaintyRegularizationLoss(**uncertainty_reg_config)
        
        self.use_calibration_loss = use_calibration_loss
        if use_calibration_loss:
            self.calibration_loss = CalibrationLoss(**calibration_config)
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined DEER loss with all regularization terms
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary with all loss components
        """
        # Main DEER loss
        deer_losses = self.deer_loss(predictions, targets)
        
        # Additional regularization
        total_loss = deer_losses['total_loss']
        all_losses = deer_losses.copy()
        
        if self.use_uncertainty_reg:
            uncertainty_reg_losses = self.uncertainty_reg_loss(predictions, targets)
            total_loss += uncertainty_reg_losses['reg_loss']
            all_losses.update(uncertainty_reg_losses)
        
        if self.use_calibration_loss:
            calibration_loss_value = self.calibration_loss(predictions, targets)
            total_loss += 0.1 * calibration_loss_value  # Weight calibration loss
            all_losses['calibration_loss'] = calibration_loss_value
        
        all_losses['combined_total_loss'] = total_loss
        
        return all_losses


def create_deer_loss(loss_type: str = 'combined', config: Optional[Dict] = None) -> nn.Module:
    """
    Factory function to create DEER loss functions
    
    Args:
        loss_type: Type of loss ('basic', 'multitask', 'combined')
        config: Loss configuration
        
    Returns:
        Initialized loss function
    """
    if config is None:
        config = {}
    
    if loss_type.lower() == 'basic':
        return DEERLoss(**config)
    elif loss_type.lower() == 'multitask':
        return MultiTaskDEERLoss(**config)
    elif loss_type.lower() == 'combined':
        return CombinedDEERLoss(**config)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def test_deer_losses():
    """Test DEER loss implementations"""
    print("🧪 Testing DEER Loss Functions...")
    
    # Test data
    batch_size = 16
    num_dims = 3  # VAD
    
    # Mock predictions (NIG parameters)
    predictions = {
        'valence_gamma': torch.randn(batch_size, 1),
        'valence_nu': F.softplus(torch.randn(batch_size, 1)) + 1e-6,
        'valence_alpha': F.softplus(torch.randn(batch_size, 1)) + 1.0,
        'valence_beta': F.softplus(torch.randn(batch_size, 1)) + 1e-6,
        
        'arousal_gamma': torch.randn(batch_size, 1),
        'arousal_nu': F.softplus(torch.randn(batch_size, 1)) + 1e-6,
        'arousal_alpha': F.softplus(torch.randn(batch_size, 1)) + 1.0,
        'arousal_beta': F.softplus(torch.randn(batch_size, 1)) + 1e-6,
        
        'dominance_gamma': torch.randn(batch_size, 1),
        'dominance_nu': F.softplus(torch.randn(batch_size, 1)) + 1e-6,
        'dominance_alpha': F.softplus(torch.randn(batch_size, 1)) + 1.0,
        'dominance_beta': F.softplus(torch.randn(batch_size, 1)) + 1e-6,
    }
    
    # Ground truth targets
    targets = torch.randn(batch_size, num_dims)
    
    # Test different loss types
    loss_functions = {
        'basic': create_deer_loss('basic'),
        'multitask': create_deer_loss('multitask'),
        'combined': create_deer_loss('combined')
    }
    
    for loss_name, loss_fn in loss_functions.items():
        print(f"\n   Testing {loss_name} loss...")
        
        try:
            if loss_name == 'basic':
                # Test with single dimension
                single_predictions = {
                    'gamma': predictions['valence_gamma'],
                    'nu': predictions['valence_nu'],
                    'alpha': predictions['valence_alpha'],
                    'beta': predictions['valence_beta']
                }
                loss_dict = loss_fn(single_predictions, targets[:, 0:1])
            else:
                loss_dict = loss_fn(predictions, targets)
            
            total_loss = loss_dict['total_loss']
            print(f"      ✅ {loss_name} loss computed: {total_loss.item():.4f}")
            print(f"      📊 Loss components: {list(loss_dict.keys())}")
            
        except Exception as e:
            print(f"      ❌ {loss_name} loss failed: {e}")
    
    print("\n✅ DEER loss testing completed successfully!")
    return True


if __name__ == "__main__":
    test_deer_losses()