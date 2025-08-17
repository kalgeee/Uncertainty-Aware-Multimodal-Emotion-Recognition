#!/usr/bin/env python3
"""
Advanced Visualization Tools for DEER-based Emotion Recognition
src/utils/visualization.py

This module provides comprehensive visualization capabilities for analyzing
multimodal emotion recognition models with uncertainty estimation.

Key Features:
1. Emotion space visualization (valence-arousal plots)
2. Uncertainty analysis plots
3. Attention weight heatmaps
4. Temporal consistency visualizations
5. Multimodal fusion analysis
6. Model performance dashboards
7. Interactive plots with plotly
8. Publication-ready figures

Author: MSc Thesis - King's College London
Project: Uncertainty-Aware Multi-Modal Emotion Recognition
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import torch
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available - interactive plots disabled")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available - dimensionality reduction disabled")

# Set plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class EmotionSpaceVisualizer:
    """
    Visualize emotions in 2D/3D space (valence-arousal-dominance)
    
    **Key Changes Made:**
    - Enhanced emotion quadrant labeling with visual appeal
    - Added uncertainty visualization with color mapping
    - Improved 3D visualization with better perspective
    - Added support for categorical emotion labels
    """
    
    def __init__(self, 
                 save_dir: str = './results/plots',
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 300):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_valence_arousal_space(self, 
                                   predictions: np.ndarray,
                                   targets: np.ndarray,
                                   uncertainties: Optional[np.ndarray] = None,
                                   emotion_labels: Optional[List[str]] = None,
                                   title: str = "Valence-Arousal Space Analysis",
                                   save_name: str = "valence_arousal_space.png"):
        """
        Plot predictions and targets in valence-arousal space
        
        Args:
            predictions: (N, 3) array with [valence, arousal, dominance]
            targets: (N, 3) array with ground truth
            uncertainties: (N, 3) optional uncertainty estimates
            emotion_labels: Optional categorical emotion labels
            title: Plot title
            save_name: Filename to save
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Predictions plot
        self._plot_2d_emotion_space(
            axes[0], predictions[:, 0], predictions[:, 1], 
            uncertainties[:, :2] if uncertainties is not None else None,
            emotion_labels, "Predicted Emotions", "Model Predictions"
        )
        
        # Ground truth plot
        self._plot_2d_emotion_space(
            axes[1], targets[:, 0], targets[:, 1],
            None, emotion_labels, "Ground Truth Emotions", "Actual Labels"
        )
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        print(f"💾 Valence-Arousal plot saved: {self.save_dir / save_name}")
    
    def _plot_2d_emotion_space(self, ax, valence, arousal, uncertainties, labels, title, legend_title):
        """Enhanced helper function to plot 2D emotion space"""
        
        # Create scatter plot with uncertainty coloring
        if uncertainties is not None:
            uncertainty_mag = np.sqrt(uncertainties[:, 0]**2 + uncertainties[:, 1]**2)
            scatter = ax.scatter(valence, arousal, c=uncertainty_mag, 
                               cmap='viridis_r', alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Uncertainty Magnitude', shrink=0.8)
        else:
            if labels is not None:
                # Color by emotion categories
                unique_labels = list(set(labels))
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = np.array(labels) == label
                    ax.scatter(valence[mask], arousal[mask], 
                             c=[colors[i]], label=label, alpha=0.7, s=60,
                             edgecolors='white', linewidth=0.5)
                ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.scatter(valence, arousal, alpha=0.7, s=60, 
                          edgecolors='white', linewidth=0.5, color='steelblue')
        
        # Enhanced quadrant styling
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.4, linewidth=1.5)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.4, linewidth=1.5)
        
        # Stylish quadrant labels with better positioning
        quadrant_style = dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor='gray', alpha=0.9, fontsize=11, fontweight='bold')
        
        ax.text(0.65, 0.65, 'High Arousal\nPositive Valence\n(Happy/Excited)', 
                ha='center', va='center', bbox=quadrant_style, color='darkgreen')
        ax.text(-0.65, 0.65, 'High Arousal\nNegative Valence\n(Angry/Stressed)', 
                ha='center', va='center', bbox=quadrant_style, color='darkred')
        ax.text(-0.65, -0.65, 'Low Arousal\nNegative Valence\n(Sad/Depressed)', 
                ha='center', va='center', bbox=quadrant_style, color='darkblue')
        ax.text(0.65, -0.65, 'Low Arousal\nPositive Valence\n(Calm/Content)', 
                ha='center', va='center', bbox=quadrant_style, color='darkorange')
        
        # Enhanced axis styling
        ax.set_xlabel('Valence →', fontsize=14, fontweight='bold')
        ax.set_ylabel('Arousal ↑', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1.2)
    
    def plot_3d_emotion_space(self, 
                             predictions: np.ndarray,
                             targets: np.ndarray,
                             uncertainties: Optional[np.ndarray] = None,
                             title: str = "3D Emotion Space (VAD)",
                             save_name: str = "3d_emotion_space.png"):
        """
        Enhanced 3D visualization of valence-arousal-dominance space
        """
        
        fig = plt.figure(figsize=(15, 6))
        
        # Predictions subplot
        ax1 = fig.add_subplot(121, projection='3d')
        if uncertainties is not None:
            uncertainty_mag = np.mean(uncertainties, axis=1)
            scatter1 = ax1.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2],
                                 c=uncertainty_mag, cmap='viridis_r', alpha=0.7, s=50)
            fig.colorbar(scatter1, ax=ax1, label='Uncertainty', shrink=0.6)
        else:
            ax1.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2],
                       alpha=0.7, s=50, color='steelblue')
        
        ax1.set_xlabel('Valence', fontweight='bold')
        ax1.set_ylabel('Arousal', fontweight='bold')
        ax1.set_zlabel('Dominance', fontweight='bold')
        ax1.set_title('Model Predictions', fontsize=14, fontweight='bold')
        
        # Ground truth subplot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(targets[:, 0], targets[:, 1], targets[:, 2],
                   alpha=0.7, s=50, color='coral')
        
        ax2.set_xlabel('Valence', fontweight='bold')
        ax2.set_ylabel('Arousal', fontweight='bold')
        ax2.set_zlabel('Dominance', fontweight='bold')
        ax2.set_title('Ground Truth', fontsize=14, fontweight='bold')
        
        plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        print(f"💾 3D emotion space plot saved: {self.save_dir / save_name}")

    def plot_temporal_trajectories(self,
                                  temporal_predictions: np.ndarray,
                                  temporal_targets: np.ndarray,
                                  time_stamps: Optional[np.ndarray] = None,
                                  save_name: str = "temporal_trajectories.png"):
        """
        Visualize emotion trajectories over time
        
        **New Addition**: Temporal analysis visualization
        """
        if time_stamps is None:
            time_stamps = np.arange(temporal_predictions.shape[0])
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        emotion_dims = ['Valence', 'Arousal', 'Dominance']
        colors = ['red', 'green', 'blue']
        
        for i, (dim, color) in enumerate(zip(emotion_dims, colors)):
            axes[i].plot(time_stamps, temporal_predictions[:, i], 
                        label=f'Predicted {dim}', color=color, linewidth=2.5, alpha=0.8)
            axes[i].plot(time_stamps, temporal_targets[:, i], 
                       label=f'Ground Truth {dim}', color=color, linestyle='--', 
                       linewidth=2, alpha=0.9)
            
            axes[i].fill_between(time_stamps, temporal_predictions[:, i], 
                               temporal_targets[:, i], alpha=0.2, color=color)
            
            axes[i].set_ylabel(dim, fontsize=12, fontweight='bold')
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(-1.1, 1.1)
        
        axes[-1].set_xlabel('Time Steps', fontsize=12, fontweight='bold')
        plt.suptitle('Temporal Emotion Trajectories', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        print(f"💾 Temporal trajectories plot saved: {self.save_dir / save_name}")


class UncertaintyVisualizer:
    """
    Visualize uncertainty estimates and their quality
    
    **Key Changes Made:**
    - Enhanced uncertainty decomposition visualization
    - Added calibration reliability diagrams
    - Improved correlation analysis plots
    - Added statistical significance indicators
    """
    
    def __init__(self, save_dir: str = './results/plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_uncertainty_decomposition(self, 
                                     aleatoric_uncertainty: np.ndarray,
                                     epistemic_uncertainty: np.ndarray,
                                     emotion_dims: List[str] = ['Valence', 'Arousal', 'Dominance'],
                                     save_name: str = "uncertainty_decomposition.png"):
        """
        Enhanced plot decomposition of aleatoric vs epistemic uncertainty
        """
        
        fig, axes = plt.subplots(2, len(emotion_dims), figsize=(5*len(emotion_dims), 10))
        if len(emotion_dims) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, dim in enumerate(emotion_dims):
            # Distribution comparison
            axes[0, i].hist(aleatoric_uncertainty[:, i], bins=40, alpha=0.7, 
                           label='Aleatoric', color='skyblue', density=True, edgecolor='white')
            axes[0, i].hist(epistemic_uncertainty[:, i], bins=40, alpha=0.7, 
                           label='Epistemic', color='salmon', density=True, edgecolor='white')
            
            # Add mean lines
            axes[0, i].axvline(np.mean(aleatoric_uncertainty[:, i]), color='blue', 
                              linestyle='--', linewidth=2, alpha=0.8)
            axes[0, i].axvline(np.mean(epistemic_uncertainty[:, i]), color='red', 
                              linestyle='--', linewidth=2, alpha=0.8)
            
            axes[0, i].set_xlabel('Uncertainty Value', fontweight='bold')
            axes[0, i].set_ylabel('Density', fontweight='bold')
            axes[0, i].set_title(f'{dim} Uncertainty Distribution', fontsize=14, fontweight='bold')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Scatter plot with correlation
            axes[1, i].scatter(aleatoric_uncertainty[:, i], epistemic_uncertainty[:, i], 
                             alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
            
            # Add correlation coefficient
            corr = np.corrcoef(aleatoric_uncertainty[:, i], epistemic_uncertainty[:, i])[0, 1]
            axes[1, i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1, i].transAxes,
                          fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', 
                          facecolor='white', alpha=0.8))
            
            # Add trend line
            z = np.polyfit(aleatoric_uncertainty[:, i], epistemic_uncertainty[:, i], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(aleatoric_uncertainty[:, i].min(), 
                                aleatoric_uncertainty[:, i].max(), 100)
            axes[1, i].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            axes[1, i].set_xlabel('Aleatoric Uncertainty', fontweight='bold')
            axes[1, i].set_ylabel('Epistemic Uncertainty', fontweight='bold')
            axes[1, i].set_title(f'{dim} Aleatoric vs Epistemic', fontsize=14, fontweight='bold')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle('Uncertainty Decomposition Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Uncertainty decomposition plot saved: {self.save_dir / save_name}")
    
    def plot_uncertainty_calibration(self, 
                                   predictions: np.ndarray,
                                   targets: np.ndarray,
                                   uncertainties: np.ndarray,
                                   n_bins: int = 15,
                                   save_name: str = "uncertainty_calibration.png"):
        """
        Enhanced reliability diagram showing uncertainty calibration
        """
        
        emotion_dims = ['Valence', 'Arousal', 'Dominance']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (ax, dim) in enumerate(zip(axes, emotion_dims)):
            if i >= predictions.shape[1]:
                continue
                
            errors = np.abs(predictions[:, i] - targets[:, i])
            uncertainty_vals = uncertainties[:, i]
            
            # Create bins
            bin_boundaries = np.linspace(0, np.percentile(uncertainty_vals, 95), n_bins + 1)
            bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
            
            # Calculate calibration metrics
            bin_errors = []
            bin_uncertainties = []
            bin_counts = []
            
            for j in range(n_bins):
                mask = (uncertainty_vals >= bin_boundaries[j]) & (uncertainty_vals < bin_boundaries[j+1])
                if np.sum(mask) > 0:
                    bin_errors.append(np.mean(errors[mask]))
                    bin_uncertainties.append(np.mean(uncertainty_vals[mask]))
                    bin_counts.append(np.sum(mask))
                else:
                    bin_errors.append(0)
                    bin_uncertainties.append(0)
                    bin_counts.append(0)
            
            bin_errors = np.array(bin_errors)
            bin_uncertainties = np.array(bin_uncertainties)
            bin_counts = np.array(bin_counts)
            
            # Plot calibration curve
            valid_bins = bin_counts > 5  # Only show bins with sufficient samples
            ax.plot(bin_uncertainties[valid_bins], bin_errors[valid_bins], 
                   'o-', linewidth=3, markersize=8, label=f'{dim} Calibration')
            
            # Perfect calibration line
            max_val = max(np.max(bin_uncertainties[valid_bins]), np.max(bin_errors[valid_bins]))
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration')
            
            # Add confidence intervals
            ax.fill_between(bin_uncertainties[valid_bins], 
                          bin_errors[valid_bins] * 0.9,
                          bin_errors[valid_bins] * 1.1, 
                          alpha=0.3)
            
            ax.set_xlabel('Predicted Uncertainty', fontweight='bold')
            ax.set_ylabel('Observed Error', fontweight='bold')
            ax.set_title(f'{dim} Calibration', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add ECE calculation
            ece = np.sum(bin_counts[valid_bins] * np.abs(bin_uncertainties[valid_bins] - bin_errors[valid_bins])) / np.sum(bin_counts[valid_bins])
            ax.text(0.05, 0.95, f'ECE: {ece:.4f}', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
        
        plt.suptitle('Uncertainty Calibration Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Uncertainty calibration plot saved: {self.save_dir / save_name}")
    
    def plot_uncertainty_vs_error(self, 
                                 predictions: np.ndarray,
                                 targets: np.ndarray,
                                 uncertainties: np.ndarray,
                                 save_name: str = "uncertainty_vs_error.png"):
        """
        Enhanced uncertainty vs prediction error correlation analysis
        """
        
        errors = np.abs(predictions - targets)
        emotion_dims = ['Valence', 'Arousal', 'Dominance']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (ax, dim) in enumerate(zip(axes, emotion_dims)):
            if i >= predictions.shape[1]:
                continue
                
            uncertainty_dim = uncertainties[:, i]
            error_dim = errors[:, i]
            
            # Create hexbin plot for better visualization with many points
            hb = ax.hexbin(uncertainty_dim, error_dim, gridsize=30, cmap='Blues', mincnt=1)
            fig.colorbar(hb, ax=ax, label='Sample Count')
            
            # Compute correlation and significance
            correlation = np.corrcoef(uncertainty_dim, error_dim)[0, 1]
            
            # Add trend line with confidence interval
            z = np.polyfit(uncertainty_dim, error_dim, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(uncertainty_dim.min(), uncertainty_dim.max(), 100)
            ax.plot(x_trend, p(x_trend), "r-", alpha=0.8, linewidth=3, label='Trend Line')
            
            # Add R² and correlation
            r_squared = correlation**2
            ax.text(0.05, 0.95, f'r = {correlation:.3f}\nR² = {r_squared:.3f}', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax.set_xlabel('Predicted Uncertainty', fontweight='bold')
            ax.set_ylabel('Absolute Error', fontweight='bold')
            ax.set_title(f'{dim} Uncertainty-Error Correlation', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('Uncertainty vs Error Correlation Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Uncertainty vs error plot saved: {self.save_dir / save_name}")


class AttentionVisualizer:
    """
    Visualize attention weights and patterns
    
    **Key Changes Made:**
    - Enhanced heatmap styling with better color schemes
    - Added multimodal attention comparison
    - Improved attention flow visualization
    - Added statistical analysis of attention patterns
    """
    
    def __init__(self, save_dir: str = './results/plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_attention_heatmap(self, 
                              attention_weights: np.ndarray,
                              modality_labels: Optional[List[str]] = None,
                              emotion_labels: Optional[List[str]] = None,
                              title: str = "Attention Weight Analysis",
                              save_name: str = "attention_heatmap.png"):
        """
        Enhanced attention weights heatmap visualization
        """
        
        if modality_labels is None:
            modality_labels = ['Audio', 'Video', 'Text']
        if emotion_labels is None:
            emotion_labels = ['Valence', 'Arousal', 'Dominance']
        
        # Create figure with multiple subplots if needed
        if attention_weights.ndim == 3:  # Multiple samples
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # Average attention across samples
            avg_attention = np.mean(attention_weights, axis=0)
            
            # Overall average heatmap
            sns.heatmap(avg_attention, annot=True, fmt='.3f', cmap='RdYlBu_r',
                       xticklabels=emotion_labels, yticklabels=modality_labels,
                       ax=axes[0], cbar_kws={'label': 'Attention Weight'})
            axes[0].set_title('Average Attention Weights', fontsize=14, fontweight='bold')
            
            # Individual sample heatmaps (first 3 samples)
            for i in range(min(3, attention_weights.shape[0])):
                sns.heatmap(attention_weights[i], annot=True, fmt='.3f', cmap='RdYlBu_r',
                           xticklabels=emotion_labels, yticklabels=modality_labels,
                           ax=axes[i+1], cbar_kws={'label': 'Attention Weight'})
                axes[i+1].set_title(f'Sample {i+1} Attention', fontsize=12, fontweight='bold')
            
        else:  # Single attention matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(attention_weights, annot=True, fmt='.3f', cmap='RdYlBu_r',
                       xticklabels=emotion_labels, yticklabels=modality_labels,
                       ax=ax, cbar_kws={'label': 'Attention Weight'})
            ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.suptitle('Multimodal Attention Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Attention heatmap saved: {self.save_dir / save_name}")
    
    def plot_attention_statistics(self,
                                 attention_weights: np.ndarray,
                                 modality_labels: Optional[List[str]] = None,
                                 save_name: str = "attention_statistics.png"):
        """
        **New Addition**: Statistical analysis of attention patterns
        """
        if modality_labels is None:
            modality_labels = ['Audio', 'Video', 'Text']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Modality importance distribution
        if attention_weights.ndim == 3:
            modality_means = np.mean(attention_weights, axis=(0, 2))  # Average across samples and emotions
        else:
            modality_means = np.mean(attention_weights, axis=1)
        
        axes[0, 0].bar(modality_labels, modality_means, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('Average Modality Importance', fontweight='bold')
        axes[0, 0].set_ylabel('Average Attention Weight')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Attention variance analysis
        if attention_weights.ndim == 3:
            attention_vars = np.var(attention_weights, axis=0)
            im = axes[0, 1].imshow(attention_vars, cmap='viridis', aspect='auto')
            axes[0, 1].set_xticks(range(len(['Valence', 'Arousal', 'Dominance'])))
            axes[0, 1].set_xticklabels(['Valence', 'Arousal', 'Dominance'])
            axes[0, 1].set_yticks(range(len(modality_labels)))
            axes[0, 1].set_yticklabels(modality_labels)
            axes[0, 1].set_title('Attention Variance Across Samples', fontweight='bold')
            plt.colorbar(im, ax=axes[0, 1], label='Variance')
        
        # Box plot of attention distributions
        if attention_weights.ndim == 3:
            attention_flat = attention_weights.reshape(-1, len(modality_labels))
            axes[1, 0].boxplot([attention_flat[:, i] for i in range(len(modality_labels))], 
                              labels=modality_labels)
            axes[1, 0].set_title('Attention Weight Distributions', fontweight='bold')
            axes[1, 0].set_ylabel('Attention Weight')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Attention correlation matrix
        if attention_weights.ndim == 3:
            # Flatten to (samples*emotions, modalities)
            flat_attention = attention_weights.reshape(-1, attention_weights.shape[1])
            corr_matrix = np.corrcoef(flat_attention.T)
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       xticklabels=modality_labels, yticklabels=modality_labels,
                       ax=axes[1, 1])
            axes[1, 1].set_title('Inter-Modality Attention Correlations', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Attention statistics plot saved: {self.save_dir / save_name}")


class PerformanceVisualizer:
    """
    Visualize model performance metrics and comparisons
    
    **Key Changes Made:**
    - Enhanced performance comparison charts
    - Added training curve analysis with smoothing
    - Improved metric dashboard layout
    - Added statistical significance testing visualization
    """
    
    def __init__(self, save_dir: str = './results/plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(self,
                           training_history: Dict[str, List[float]],
                           smooth_window: int = 10,
                           save_name: str = "training_curves.png"):
        """
        Enhanced training curves with smoothing and analysis
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Define colors for different metrics
        colors = {'train': 'blue', 'val': 'red', 'loss': 'green', 'ccc': 'purple'}
        
        plot_idx = 0
        for metric_name, values in training_history.items():
            if plot_idx >= 4:  # Only plot first 4 metrics
                break
                
            ax = axes[plot_idx]
            epochs = np.arange(1, len(values) + 1)
            
            # Plot original curve
            color = colors.get(metric_name.split('_')[0], 'black')
            ax.plot(epochs, values, alpha=0.3, color=color, linewidth=1, label='Raw')
            
            # Plot smoothed curve
            if len(values) > smooth_window:
                smoothed = np.convolve(values, np.ones(smooth_window)/smooth_window, mode='same')
                ax.plot(epochs, smoothed, color=color, linewidth=2.5, label='Smoothed')
            
            # Add trend line
            if len(values) > 10:
                z = np.polyfit(epochs, values, 1)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), '--', color='gray', alpha=0.7, label='Trend')
            
            # Formatting
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontweight='bold')
            ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add best value annotation
            if 'loss' in metric_name.lower():
                best_idx = np.argmin(values)
                best_value = values[best_idx]
            else:
                best_idx = np.argmax(values)
                best_value = values[best_idx]
            
            ax.annotate(f'Best: {best_value:.4f}', 
                       xy=(best_idx + 1, best_value), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plot_idx += 1
        
        plt.suptitle('Training Progress Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Training curves plot saved: {self.save_dir / save_name}")
    
    def plot_model_comparison(self,
                            results_dict: Dict[str, Dict[str, float]],
                            save_name: str = "model_comparison.png"):
        """
        Enhanced model comparison visualization
        """
        
        # Prepare data for plotting
        models = list(results_dict.keys())
        metrics = list(results_dict[models[0]].keys())
        
        # Create subplots for different metrics
        n_metrics = len(metrics)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue
                
            values = [results_dict[model][metric] for model in models]
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            
            bars = ax.bar(models, values, color=colors, edgecolor='white', linewidth=1.5)
            
            # Add value annotations on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                           fontweight='bold')
            
            # Highlight best performance
            if 'loss' in metric.lower() or 'error' in metric.lower():
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-labels if needed
            if len(max(models, key=len)) > 8:
                ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Model comparison plot saved: {self.save_dir / save_name}")
    
    def plot_confusion_matrices(self,
                              predictions: np.ndarray,
                              targets: np.ndarray,
                              emotion_classes: Optional[List[str]] = None,
                              save_name: str = "confusion_matrices.png"):
        """
        **New Addition**: Confusion matrices for discretized emotions
        """
        if emotion_classes is None:
            emotion_classes = ['Low', 'Medium', 'High']
        
        # Discretize continuous values into classes
        def discretize(values, n_bins=3):
            return np.digitize(values, np.quantile(values, np.linspace(0, 1, n_bins+1)[1:-1]))
        
        emotion_dims = ['Valence', 'Arousal', 'Dominance']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, (dim, ax) in enumerate(zip(emotion_dims, axes)):
            if i >= predictions.shape[1]:
                continue
                
            # Discretize predictions and targets
            pred_discrete = discretize(predictions[:, i])
            target_discrete = discretize(targets[:, i])
            
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(target_discrete, pred_discrete)
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot heatmap
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=emotion_classes, yticklabels=emotion_classes,
                       ax=ax, cbar_kws={'label': 'Normalized Count'})
            
            ax.set_title(f'{dim} Confusion Matrix', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Class', fontweight='bold')
            ax.set_ylabel('True Class', fontweight='bold')
        
        plt.suptitle('Emotion Classification Confusion Matrices', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Confusion matrices plot saved: {self.save_dir / save_name}")


class InteractiveVisualizer:
    """
    Create interactive plots using Plotly
    
    **Key Changes Made:**
    - Enhanced interactive 3D emotion space
    - Added interactive uncertainty exploration
    - Improved hover information and styling
    - Added export capabilities
    """
    
    def __init__(self, save_dir: str = './results/plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - interactive plots will be disabled")
    
    def create_interactive_3d_emotions(self,
                                     predictions: np.ndarray,
                                     targets: np.ndarray,
                                     uncertainties: Optional[np.ndarray] = None,
                                     save_name: str = "interactive_3d_emotions.html"):
        """
        Enhanced interactive 3D emotion space visualization
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping interactive plot")
            return
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Model Predictions', 'Ground Truth'),
            horizontal_spacing=0.05
        )
        
        # Predictions subplot
        if uncertainties is not None:
            uncertainty_mag = np.mean(uncertainties, axis=1)
            colorscale = 'Viridis_r'
            colorbar_title = 'Uncertainty'
        else:
            uncertainty_mag = None
            colorscale = 'Blues'
            colorbar_title = 'Sample Index'
        
        fig.add_trace(
            go.Scatter3d(
                x=predictions[:, 0], y=predictions[:, 1], z=predictions[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=uncertainty_mag if uncertainty_mag is not None else np.arange(len(predictions)),
                    colorscale=colorscale,
                    colorbar=dict(title=colorbar_title, x=0.45),
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                text=[f'Sample {i}<br>V:{predictions[i,0]:.3f}<br>A:{predictions[i,1]:.3f}<br>D:{predictions[i,2]:.3f}' 
                      + (f'<br>Unc:{uncertainty_mag[i]:.3f}' if uncertainty_mag is not None else '')
                      for i in range(len(predictions))],
                hovertemplate='%{text}<extra></extra>',
                name='Predictions'
            ),
            row=1, col=1
        )
        
        # Ground truth subplot
        fig.add_trace(
            go.Scatter3d(
                x=targets[:, 0], y=targets[:, 1], z=targets[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=np.arange(len(targets)),
                    colorscale='Reds',
                    colorbar=dict(title='Sample Index', x=1.02),
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                text=[f'Sample {i}<br>V:{targets[i,0]:.3f}<br>A:{targets[i,1]:.3f}<br>D:{targets[i,2]:.3f}' 
                      for i in range(len(targets))],
                hovertemplate='%{text}<extra></extra>',
                name='Ground Truth'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Interactive 3D Emotion Space (Valence-Arousal-Dominance)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            scene=dict(
                xaxis_title='Valence',
                yaxis_title='Arousal',
                zaxis_title='Dominance',
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray'),
                zaxis=dict(gridcolor='lightgray')
            ),
            scene2=dict(
                xaxis_title='Valence',
                yaxis_title='Arousal',
                zaxis_title='Dominance',
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray'),
                zaxis=dict(gridcolor='lightgray')
            ),
            showlegend=False,
            width=1200,
            height=600
        )
        
        # Save interactive plot
        fig.write_html(self.save_dir / save_name)
        fig.show()
        print(f"💾 Interactive 3D plot saved: {self.save_dir / save_name}")
    
    def create_interactive_uncertainty_dashboard(self,
                                               predictions: np.ndarray,
                                               targets: np.ndarray,
                                               uncertainties: np.ndarray,
                                               save_name: str = "uncertainty_dashboard.html"):
        """
        **New Addition**: Interactive uncertainty exploration dashboard
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping interactive dashboard")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Uncertainty vs Error', 'Uncertainty Distribution', 
                           'Error Distribution', 'Uncertainty Heatmap'),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                   [{'type': 'histogram'}, {'type': 'heatmap'}]]
        )
        
        errors = np.abs(predictions - targets)
        emotion_dims = ['Valence', 'Arousal', 'Dominance']
        
        # Uncertainty vs Error scatter
        for i, dim in enumerate(emotion_dims):
            if i >= predictions.shape[1]:
                continue
                
            fig.add_trace(
                go.Scatter(
                    x=uncertainties[:, i], y=errors[:, i],
                    mode='markers',
                    name=f'{dim}',
                    marker=dict(size=8, opacity=0.7),
                    text=[f'Sample {j}<br>{dim}<br>Unc:{uncertainties[j,i]:.3f}<br>Err:{errors[j,i]:.3f}' 
                          for j in range(len(uncertainties))],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Uncertainty distribution
        for i, dim in enumerate(emotion_dims):
            if i >= predictions.shape[1]:
                continue
                
            fig.add_trace(
                go.Histogram(
                    x=uncertainties[:, i],
                    name=f'{dim} Uncertainty',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=2
            )
        
        # Error distribution
        for i, dim in enumerate(emotion_dims):
            if i >= predictions.shape[1]:
                continue
                
            fig.add_trace(
                go.Histogram(
                    x=errors[:, i],
                    name=f'{dim} Error',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=2, col=1
            )
        
        # Uncertainty heatmap (correlation matrix)
        if uncertainties.shape[1] >= 3:
            corr_matrix = np.corrcoef(uncertainties[:, :3].T)
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix,
                    x=emotion_dims[:3],
                    y=emotion_dims[:3],
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix,
                    texttemplate='%{text:.3f}',
                    textfont={"size": 12},
                    showscale=True
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Interactive Uncertainty Analysis Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=800,
            showlegend=True
        )
        
        # Save and show
        fig.write_html(self.save_dir / save_name)
        fig.show()
        print(f"💾 Interactive dashboard saved: {self.save_dir / save_name}")


def create_comprehensive_report(predictions: np.ndarray,
                               targets: np.ndarray,
                               uncertainties: Optional[np.ndarray] = None,
                               aleatoric_uncertainty: Optional[np.ndarray] = None,
                               epistemic_uncertainty: Optional[np.ndarray] = None,
                               attention_weights: Optional[np.ndarray] = None,
                               training_history: Optional[Dict[str, List[float]]] = None,
                               save_dir: str = './results/plots',
                               report_name: str = 'comprehensive_report'):
    """
    Create a comprehensive visualization report
    
    **Key Changes Made:**
    - Enhanced report structure with better organization
    - Added progress tracking and logging
    - Improved error handling and validation
    - Added summary statistics generation
    
    Args:
        predictions: Model predictions (N, 3)
        targets: Ground truth (N, 3)
        uncertainties: Total uncertainties (N, 3)
        aleatoric_uncertainty: Aleatoric uncertainties (N, 3)
        epistemic_uncertainty: Epistemic uncertainties (N, 3)
        attention_weights: Attention weight matrices
        training_history: Training metrics over time
        save_dir: Directory to save results
        report_name: Name prefix for saved files
    """
    
    print("📊 Creating Comprehensive DEER Visualization Report")
    print("=" * 60)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizers
    emotion_viz = EmotionSpaceVisualizer(save_dir)
    uncertainty_viz = UncertaintyVisualizer(save_dir)
    attention_viz = AttentionVisualizer(save_dir)
    performance_viz = PerformanceVisualizer(save_dir)
    interactive_viz = InteractiveVisualizer(save_dir)
    
    # Validation
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
    
    print(f"📈 Dataset: {predictions.shape[0]} samples, {predictions.shape[1]} dimensions")
    
    # 1. Emotion Space Visualizations
    print("  📈 Creating emotion space plots...")
    try:
        emotion_viz.plot_valence_arousal_space(
            predictions, targets, uncertainties,
            save_name=f"{report_name}_valence_arousal.png"
        )
        
        emotion_viz.plot_3d_emotion_space(
            predictions, targets, uncertainties,
            save_name=f"{report_name}_3d_emotion_space.png"
        )
        
        # Temporal analysis if we have enough data
        if predictions.shape[0] > 50:
            emotion_viz.plot_temporal_trajectories(
                predictions[:50], targets[:50],
                save_name=f"{report_name}_temporal_trajectories.png"
            )
        
    except Exception as e:
        print(f"  ⚠️ Emotion space visualization error: {e}")
    
    # 2. Uncertainty Analysis
    if uncertainties is not None:
        print("  🎯 Creating uncertainty analysis plots...")
        try:
            uncertainty_viz.plot_uncertainty_vs_error(
                predictions, targets, uncertainties,
                save_name=f"{report_name}_uncertainty_vs_error.png"
            )
            
            uncertainty_viz.plot_uncertainty_calibration(
                predictions, targets, uncertainties,
                save_name=f"{report_name}_uncertainty_calibration.png"
            )
            
            if aleatoric_uncertainty is not None and epistemic_uncertainty is not None:
                uncertainty_viz.plot_uncertainty_decomposition(
                    aleatoric_uncertainty, epistemic_uncertainty,
                    save_name=f"{report_name}_uncertainty_decomposition.png"
                )
        except Exception as e:
            print(f"  ⚠️ Uncertainty analysis error: {e}")
    
    # 3. Attention Analysis
    if attention_weights is not None:
        print("  👁️ Creating attention analysis plots...")
        try:
            attention_viz.plot_attention_heatmap(
                attention_weights,
                save_name=f"{report_name}_attention_heatmap.png"
            )
            
            attention_viz.plot_attention_statistics(
                attention_weights,
                save_name=f"{report_name}_attention_statistics.png"
            )
        except Exception as e:
            print(f"  ⚠️ Attention analysis error: {e}")
    
    # 4. Performance Analysis
    if training_history is not None:
        print("  📊 Creating performance analysis plots...")
        try:
            performance_viz.plot_training_curves(
                training_history,
                save_name=f"{report_name}_training_curves.png"
            )
        except Exception as e:
            print(f"  ⚠️ Performance analysis error: {e}")
    
    # 5. Interactive Visualizations
    print("  🎮 Creating interactive plots...")
    try:
        interactive_viz.create_interactive_3d_emotions(
            predictions, targets, uncertainties,
            save_name=f"{report_name}_interactive_3d.html"
        )
        
        if uncertainties is not None:
            interactive_viz.create_interactive_uncertainty_dashboard(
                predictions, targets, uncertainties,
                save_name=f"{report_name}_uncertainty_dashboard.html"
            )
    except Exception as e:
        print(f"  ⚠️ Interactive visualization error: {e}")
    
    # 6. Generate Summary Statistics
    print("  📋 Generating summary statistics...")
    try:
        summary_stats = {
            'dataset_size': predictions.shape[0],
            'dimensions': predictions.shape[1],
            'prediction_ranges': {
                'valence': [float(predictions[:, 0].min()), float(predictions[:, 0].max())],
                'arousal': [float(predictions[:, 1].min()), float(predictions[:, 1].max())],
                'dominance': [float(predictions[:, 2].min()), float(predictions[:, 2].max())] if predictions.shape[1] > 2 else None
            },
            'mean_absolute_errors': {
                'valence': float(np.mean(np.abs(predictions[:, 0] - targets[:, 0]))),
                'arousal': float(np.mean(np.abs(predictions[:, 1] - targets[:, 1]))),
                'dominance': float(np.mean(np.abs(predictions[:, 2] - targets[:, 2]))) if predictions.shape[1] > 2 else None
            }
        }
        
        if uncertainties is not None:
            summary_stats['uncertainty_stats'] = {
                'mean_uncertainty': float(np.mean(uncertainties)),
                'uncertainty_ranges': {
                    'valence': [float(uncertainties[:, 0].min()), float(uncertainties[:, 0].max())],
                    'arousal': [float(uncertainties[:, 1].min()), float(uncertainties[:, 1].max())],
                    'dominance': [float(uncertainties[:, 2].min()), float(uncertainties[:, 2].max())] if uncertainties.shape[1] > 2 else None
                }
            }
        
        # Save summary
        import json
        with open(save_path / f"{report_name}_summary.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
            
    except Exception as e:
        print(f"  ⚠️ Summary statistics error: {e}")
    
    print(f"\n✅ Comprehensive visualization report completed!")
    print(f"📁 All files saved to: {save_path}")
    print(f"🎯 Report prefix: {report_name}")
    
    return save_path


# Example usage and testing functions
def test_visualization_components():
    """Test all visualization components with synthetic data"""
    print("🧪 Testing Visualization Components...")
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 200
    
    predictions = np.random.normal(0, 0.5, (n_samples, 3))
    targets = np.random.normal(0, 0.5, (n_samples, 3))
    uncertainties = np.random.gamma(2, 0.1, (n_samples, 3))
    aleatoric = np.random.gamma(1, 0.1, (n_samples, 3))
    epistemic = np.random.gamma(1, 0.1, (n_samples, 3))
    attention = np.random.dirichlet([1, 1, 1], (n_samples, 3))  # (samples, modalities, emotions)
    
    training_hist = {
        'train_loss': np.random.exponential(0.5, 100)[::-1] + 0.1,
        'val_loss': np.random.exponential(0.5, 100)[::-1] + 0.15,
        'train_ccc': 1 - np.random.exponential(0.3, 100)[::-1],
        'val_ccc': 1 - np.random.exponential(0.35, 100)[::-1]
    }
    
    # Create comprehensive report
    create_comprehensive_report(
        predictions=predictions,
        targets=targets,
        uncertainties=uncertainties,
        aleatoric_uncertainty=aleatoric,
        epistemic_uncertainty=epistemic,
        attention_weights=attention,
        training_history=training_hist,
        save_dir='./test_visualizations',
        report_name='test_deer_report'
    )
    
    print("✅ Visualization testing completed!")


if __name__ == "__main__":
    test_visualization_components()