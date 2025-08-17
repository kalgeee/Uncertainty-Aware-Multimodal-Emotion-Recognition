"""
Deep Evidential Emotion Regression (DEER) for Multimodal Recognition

This module implements the core DEER methodology extended to multimodal settings,
achieving uncertainty quantification through Normal-Inverse-Gamma distributions
over Gaussian likelihood functions.

Mathematical Framework:
    μ = Neural Network Output (mean prediction)
    ν = Evidence parameter (determines precision)  
    α = Evidence parameter (shape parameter)
    β = Evidence parameter (rate parameter)
    
    NIG(μ, λ, α, β) where λ = ν
    Total Uncertainty = Aleatoric + Epistemic
    
References:
    Wu, J., et al. (2023). Deep Evidential Emotion Regression. ACL 2023.
    Amini, A., et al. (2020). Deep Evidential Regression. NeurIPS 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import math


class DEERLayer(nn.Module):
    """Deep Evidential Emotion Regression Layer"""
    
    def __init__(self, input_dim: int, output_dim: int = 1, 
                 hidden_dim: int = 256, dropout: float = 0.3):
        """
        Args:
            input_dim: Dimension of input features
            output_dim: Number of emotion dimensions (1 for single dimension)
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super(DEERLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Evidence network architecture
        self.evidence_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4 * output_dim)  # μ, ν, α, β for each dimension
        )
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.evidence_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing NIG parameters
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Dictionary containing:
                'mu': Mean predictions [batch_size, output_dim]
                'nu': Evidence parameter ν [batch_size, output_dim]  
                'alpha': Evidence parameter α [batch_size, output_dim]
                'beta': Evidence parameter β [batch_size, output_dim]
                'uncertainty': Total uncertainty [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Compute evidence parameters
        evidence = self.evidence_net(x)
        evidence = evidence.view(batch_size, self.output_dim, 4)
        
        # Extract NIG parameters
        mu = evidence[:, :, 0]  # Mean
        nu = F.softplus(evidence[:, :, 1]) + 1e-6  # Precision (> 0)
        alpha = F.softplus(evidence[:, :, 2]) + 1.0  # Shape (> 1)
        beta = F.softplus(evidence[:, :, 3]) + 1e-6   # Rate (> 0)
        
        # Compute uncertainties
        aleatoric = beta / (alpha - 1)  # Data uncertainty
        epistemic = beta / (nu * (alpha - 1))  # Model uncertainty
        total_uncertainty = aleatoric + epistemic
        
        return {
            'mu': mu,
            'nu': nu, 
            'alpha': alpha,
            'beta': beta,
            'aleatoric_uncertainty': aleatoric,
            'epistemic_uncertainty': epistemic,
            'uncertainty': total_uncertainty
        }


class DEERLoss(nn.Module):
    """DEER Loss Function with Evidence Regularization"""
    
    def __init__(self, evidence_weight: float = 1.0, 
                 kl_weight: float = 1.0):
        """
        Args:
            evidence_weight: Weight for evidence regularization
            kl_weight: Weight for KL divergence term
        """
        super(DEERLoss, self).__init__()
        self.evidence_weight = evidence_weight
        self.kl_weight = kl_weight
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute DEER loss with evidence regularization
        
        Args:
            predictions: Dictionary from DEER layer forward pass
            targets: Ground truth values [batch_size, output_dim]
            
        Returns:
            Dictionary containing loss components
        """
        mu = predictions['mu']
        nu = predictions['nu']
        alpha = predictions['alpha']
        beta = predictions['beta']
        
        # Ensure targets have correct shape
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
            
        # Compute squared error
        squared_error = (targets - mu) ** 2
        
        # Evidence lower bound (ELBO) - NIG likelihood
        nig_nll = (
            0.5 * torch.log(np.pi / nu)
            - alpha * torch.log(2 * beta)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
            + (alpha + 0.5) * torch.log(
                beta + nu * squared_error / 2
            )
        )
        
        # Evidence regularization term
        evidence_reg = (
            nu * squared_error + 2 * beta * (1 + nu)
        ) / (2 * nu * (1 + nu))
        
        # KL divergence regularization (prevents overconfidence)
        kl_div = self._compute_kl_regularization(nu, alpha, beta)
        
        # Total loss
        total_loss = (
            torch.mean(nig_nll) + 
            self.evidence_weight * torch.mean(evidence_reg) +
            self.kl_weight * torch.mean(kl_div)
        )
        
        return {
            'total_loss': total_loss,
            'nll_loss': torch.mean(nig_nll),
            'evidence_reg': torch.mean(evidence_reg),
            'kl_reg': torch.mean(kl_div),
            'mse': torch.mean(squared_error)
        }
    
    def _compute_kl_regularization(self, nu: torch.Tensor, 
                                  alpha: torch.Tensor, 
                                  beta: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence regularization term"""
        # KL divergence from NIG to uniform prior
        kl = (
            0.5 * (nu - 1) 
            + alpha * torch.log(beta)
            - torch.lgamma(alpha)
            + torch.lgamma(alpha + 0.5)
            - 0.5 * torch.log(2 * np.pi * beta)
        )
        return torch.clamp(kl, min=0)


class MultiDimensionalDEER(nn.Module):
    """Multi-dimensional DEER for valence, arousal, dominance"""
    
    def __init__(self, input_dim: int, emotion_dims: int = 3,
                 hidden_dim: int = 256, dropout: float = 0.3):
        """
        Args:
            input_dim: Input feature dimension
            emotion_dims: Number of emotion dimensions (3 for VAD)
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super(MultiDimensionalDEER, self).__init__()
        
        self.emotion_dims = emotion_dims
        
        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate DEER heads for each emotion dimension
        self.deer_heads = nn.ModuleList([
            DEERLayer(hidden_dim, output_dim=1, hidden_dim=hidden_dim//2, dropout=dropout)
            for _ in range(emotion_dims)
        ])
        
        # Dimension names for interpretability
        self.dimension_names = ['valence', 'arousal', 'dominance'][:emotion_dims]
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-dimensional emotion prediction
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with predictions for each emotion dimension
        """
        batch_size = x.size(0)
        
        # Shared feature processing
        features = self.feature_processor(x)
        
        # Process each emotion dimension
        predictions = {}
        for i, (deer_head, dim_name) in enumerate(zip(self.deer_heads, self.dimension_names)):
            dim_pred = deer_head(features)
            
            # Add dimension-specific keys
            for key, value in dim_pred.items():
                predictions[f"{dim_name}_{key}"] = value
        
        # Aggregate predictions
        predictions['mu_all'] = torch.cat([
            predictions[f"{dim}_mu"] for dim in self.dimension_names
        ], dim=1)
        
        predictions['uncertainty_all'] = torch.cat([
            predictions[f"{dim}_uncertainty"] for dim in self.dimension_names  
        ], dim=1)
        
        return predictions


class HierarchicalDEERFusion(nn.Module):
    """Hierarchical multimodal fusion with DEER uncertainty propagation"""
    
    def __init__(self, audio_dim: int = 84, video_dim: int = 256, 
                 text_dim: int = 768, fusion_dim: int = 512,
                 emotion_dims: int = 3, dropout: float = 0.3):
        """
        Args:
            audio_dim: Audio feature dimension
            video_dim: Video feature dimension  
            text_dim: Text feature dimension
            fusion_dim: Fusion layer dimension
            emotion_dims: Number of emotion dimensions
            dropout: Dropout probability
        """
        super(HierarchicalDEERFusion, self).__init__()
        
        # Modality encoders
        self.audio_encoder = nn.Linear(audio_dim, 256)
        self.video_encoder = nn.Linear(video_dim, 256)  
        self.text_encoder = nn.Linear(text_dim, 256)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(256, num_heads=8)
        
        # Stage 1: Audio-Visual fusion
        self.av_fusion = nn.Sequential(
            nn.Linear(512, fusion_dim),  # 256 + 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Stage 2: Trimodal fusion
        self.trimodal_fusion = nn.Sequential(
            nn.Linear(fusion_dim + 256, fusion_dim),  # AV + Text
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Final DEER prediction
        self.deer_predictor = MultiDimensionalDEER(
            fusion_dim, emotion_dims, hidden_dim=256, dropout=dropout
        )
        
    def forward(self, audio_features: torch.Tensor, 
                video_features: torch.Tensor,
                text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Hierarchical fusion with uncertainty propagation
        
        Args:
            audio_features: Audio features [batch_size, audio_dim]
            video_features: Video features [batch_size, video_dim]
            text_features: Text features [batch_size, text_dim]
            
        Returns:
            DEER predictions with uncertainty estimates
        """
        # Encode modalities
        audio_encoded = F.relu(self.audio_encoder(audio_features))
        video_encoded = F.relu(self.video_encoder(video_features))
        text_encoded = F.relu(self.text_encoder(text_features))
        
        # Cross-modal attention
        audio_attended, video_attended = self.cross_attention(
            audio_encoded, video_encoded, text_encoded
        )
        
        # Stage 1: Audio-Visual fusion
        av_concat = torch.cat([audio_attended, video_attended], dim=1)
        av_fused = self.av_fusion(av_concat)
        
        # Stage 2: Trimodal fusion
        trimodal_concat = torch.cat([av_fused, text_encoded], dim=1)
        trimodal_fused = self.trimodal_fusion(trimodal_concat)
        
        # Final DEER prediction
        predictions = self.deer_predictor(trimodal_fused)
        
        return predictions


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism with uncertainty weighting"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super(CrossModalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Attention projections
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Uncertainty-based weighting
        self.uncertainty_gate = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),  # Weights for audio and video
            nn.Softmax(dim=1)
        )
        
    def forward(self, audio: torch.Tensor, video: torch.Tensor, 
                text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal attention with text as context
        
        Args:
            audio: Audio features [batch_size, feature_dim]
            video: Video features [batch_size, feature_dim] 
            text: Text features [batch_size, feature_dim]
            
        Returns:
            Attended audio and video features
        """
        batch_size = audio.size(0)
        
        # Compute attention weights using text as query
        q = self.query_proj(text).view(batch_size, self.num_heads, self.head_dim)
        k_audio = self.key_proj(audio).view(batch_size, self.num_heads, self.head_dim)
        k_video = self.key_proj(video).view(batch_size, self.num_heads, self.head_dim)
        v_audio = self.value_proj(audio).view(batch_size, self.num_heads, self.head_dim)
        v_video = self.value_proj(video).view(batch_size, self.num_heads, self.head_dim)
        
        # Attention scores
        scores_audio = torch.sum(q * k_audio, dim=2) / math.sqrt(self.head_dim)
        scores_video = torch.sum(q * k_video, dim=2) / math.sqrt(self.head_dim)
        
        # Attention weights
        attn_audio = F.softmax(scores_audio, dim=1)
        attn_video = F.softmax(scores_video, dim=1)
        
        # Apply attention
        attended_audio = torch.sum(
            attn_audio.unsqueeze(2) * v_audio, dim=1
        )
        attended_video = torch.sum(
            attn_video.unsqueeze(2) * v_video, dim=1
        )
        
        # Uncertainty-based modality weighting
        modality_context = torch.cat([audio, video, text], dim=1)
        modality_weights = self.uncertainty_gate(modality_context)
        
        # Apply modality weights
        weighted_audio = attended_audio * modality_weights[:, 0:1]
        weighted_video = attended_video * modality_weights[:, 1:2]
        
        return weighted_audio, weighted_video


def test_deer_implementation():
    """Test DEER implementation with sample data"""
    print("Testing DEER Implementation...")
    
    # Test parameters
    batch_size = 16
    audio_dim, video_dim, text_dim = 84, 256, 768
    
    # Create sample data
    audio_features = torch.randn(batch_size, audio_dim)
    video_features = torch.randn(batch_size, video_dim)
    text_features = torch.randn(batch_size, text_dim)
    targets = torch.randn(batch_size, 3)  # VAD targets
    
    # Initialize model
    model = HierarchicalDEERFusion(
        audio_dim=audio_dim,
        video_dim=video_dim, 
        text_dim=text_dim,
        fusion_dim=512,
        emotion_dims=3
    )
    
    # Forward pass
    predictions = model(audio_features, video_features, text_features)
    
    # Test loss computation
    loss_fn = DEERLoss(evidence_weight=1.0, kl_weight=0.1)
    
    # Compute loss for each dimension
    total_losses = []
    for dim in ['valence', 'arousal', 'dominance']:
        dim_predictions = {
            'mu': predictions[f'{dim}_mu'],
            'nu': predictions[f'{dim}_nu'],
            'alpha': predictions[f'{dim}_alpha'],
            'beta': predictions[f'{dim}_beta']
        }
        
        dim_targets = targets[:, ['valence', 'arousal', 'dominance'].index(dim)]
        loss_dict = loss_fn(dim_predictions, dim_targets)
        total_losses.append(loss_dict['total_loss'])
    
    total_loss = sum(total_losses)
    
    print(f" Model forward pass successful")
    print(f" Predictions shape: {predictions['mu_all'].shape}")
    print(f" Uncertainty shape: {predictions['uncertainty_all'].shape}")
    print(f" Total loss: {total_loss.item():.4f}")
    print(f" Sample predictions: μ={predictions['mu_all'][0].detach().numpy()}")
    print(f" Sample uncertainties: σ={torch.sqrt(predictions['uncertainty_all'][0]).detach().numpy()}")
    
    return model, predictions, total_loss


if __name__ == "__main__":
    # Run implementation test
    test_deer_implementation()
    print("\n DEER implementation test completed successfully!")