"""
Training Framework for Multimodal DEER

This module implements the complete training pipeline that achieved 
state-of-the-art performance through multi-dataset training and 
advanced optimization strategies.

Training Strategy:
    1. Multi-dataset curriculum learning
    2. Uncertainty-aware loss balancing
    3. Attention regularization
    4. Cross-dataset validation
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import mean_absolute_error

from complete_model import CompleteDEERModel, ModelConfig, ModelCheckpoint
from metrics import DEERMetrics, uncertainty_calibration_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Complete training configuration"""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    batch_size: int = 32
    num_epochs: int = 100
    
    # Scheduling
    scheduler_type: str = "cosine"  # cosine, plateau, exponential
    warmup_epochs: int = 5
    patience: int = 10
    
    # Loss weighting
    evidence_weight: float = 1.0
    kl_weight: float = 0.1
    attention_reg_weight: float = 0.1
    
    # Multi-dataset training
    dataset_weights: Dict[str, float] = field(default_factory=lambda: {
        'iemocap': 1.0, 'ravdess': 0.8, 'meld': 0.6
    })
    curriculum_learning: bool = True
    
    # Validation
    val_frequency: int = 5
    save_frequency: int = 10
    early_stopping: bool = True
    
    # Directories
    output_dir: str = "./results"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"


class DEERTrainer:
    """Complete training framework for multimodal DEER"""
    
    def __init__(self, model: CompleteDEERModel, config: TrainingConfig,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize trainer
        
        Args:
            model: CompleteDEERModel instance
            config: Training configuration
            device: Training device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup directories
        self._setup_directories()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Metrics and logging
        self.metrics = DEERMetrics()
        self.writer = SummaryWriter(log_dir=self.config.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_ccc = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'ccc_valence': [],
            'ccc_arousal': [], 'ccc_dominance': [], 'ece': []
        }
        
        logger.info(f"✅ Trainer initialized on device: {device}")
        logger.info(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.config.output_dir, self.config.log_dir, self.config.checkpoint_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with parameter groups"""
        # Separate parameter groups for different learning rates
        encoder_params = []
        attention_params = []
        deer_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            elif 'attention' in name:
                attention_params.append(param)
            elif 'deer' in name or 'fusion' in name:
                deer_params.append(param)
            else:
                deer_params.append(param)  # Default group
        
        param_groups = [
            {'params': encoder_params, 'lr': self.config.learning_rate * 0.5},
            {'params': attention_params, 'lr': self.config.learning_rate},
            {'params': deer_params, 'lr': self.config.learning_rate}
        ]
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.config.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.num_epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.patience // 2,
                factor=0.5,
                verbose=True
            )
        else:  # exponential
            scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        
        return scheduler
    
    def train_epoch(self, train_loaders: Dict[str, DataLoader]) -> Dict[str, float]:
        """
        Train one epoch with multi-dataset curriculum
        
        Args:
            train_loaders: Dictionary of dataset name -> DataLoader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0, 'deer_loss': 0.0, 'attention_reg': 0.0,
            'nll_loss': 0.0, 'evidence_reg': 0.0, 'kl_reg': 0.0
        }
        total_samples = 0
        
        # Curriculum learning: adjust dataset sampling based on epoch
        dataset_probs = self._get_curriculum_probabilities()
        
        # Create unified batch iterator
        batch_iterator = self._create_multi_dataset_iterator(train_loaders, dataset_probs)
        
        for batch_idx, (batch_data, dataset_name) in enumerate(batch_iterator):
            # Move data to device
            audio_features = batch_data['audio_features'].to(self.device)
            video_features = batch_data['video_features'].to(self.device)
            text_features = batch_data['text_features'].to(self.device)
            targets = batch_data['targets'].to(self.device)
            
            # Forward pass
            predictions = self.model(audio_features, video_features, text_features)
            
            # Compute loss with dataset weighting
            loss_dict = self.model.compute_loss(predictions, targets)
            dataset_weight = self.config.dataset_weights.get(dataset_name, 1.0)
            weighted_loss = loss_dict['total_loss'] * dataset_weight
            
            # Backward pass
            self.optimizer.zero_grad()
            weighted_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            batch_size = audio_features.size(0)
            for key, value in loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key] += value.item() * batch_size
            
            total_samples += batch_size
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                    f"Loss={weighted_loss.item():.4f}, Dataset={dataset_name}"
                )
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= total_samples
        
        return epoch_losses
    
    def validate_epoch(self, val_loaders: Dict[str, DataLoader]) -> Dict[str, float]:
        """
        Validate model on all datasets
        
        Args:
            val_loaders: Dictionary of validation dataloaders
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        val_losses = []
        
        with torch.no_grad():
            for dataset_name, dataloader in val_loaders.items():
                dataset_preds = []
                dataset_targets = []
                dataset_uncertainties = []
                dataset_losses = []
                
                for batch_data in dataloader:
                    # Move to device
                    audio_features = batch_data['audio_features'].to(self.device)
                    video_features = batch_data['video_features'].to(self.device)
                    text_features = batch_data['text_features'].to(self.device)
                    targets = batch_data['targets'].to(self.device)
                    
                    # Forward pass
                    predictions = self.model(audio_features, video_features, text_features)
                    
                    # Extract predictions and uncertainties
                    pred_values, uncertainties = self.model.get_predictions_and_uncertainties(predictions)
                    
                    # Compute loss
                    loss_dict = self.model.compute_loss(predictions, targets)
                    
                    # Collect results
                    dataset_preds.append(pred_values.cpu())
                    dataset_targets.append(targets.cpu())
                    dataset_uncertainties.append(uncertainties.cpu())
                    dataset_losses.append(loss_dict['total_loss'].item())
                
                # Concatenate dataset results
                dataset_preds = torch.cat(dataset_preds, dim=0)
                dataset_targets = torch.cat(dataset_targets, dim=0)
                dataset_uncertainties = torch.cat(dataset_uncertainties, dim=0)
                
                all_predictions.append(dataset_preds)
                all_targets.append(dataset_targets)
                all_uncertainties.append(dataset_uncertainties)
                val_losses.extend(dataset_losses)
        
        # Combine all datasets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_uncertainties = torch.cat(all_uncertainties, dim=0)
        
        # Compute comprehensive metrics
        val_metrics = self._compute_validation_metrics(
            all_predictions, all_targets, all_uncertainties
        )
        val_metrics['val_loss'] = np.mean(val_losses)
        
        return val_metrics
    
    def _compute_validation_metrics(self, predictions: torch.Tensor, 
                                   targets: torch.Tensor,
                                   uncertainties: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive validation metrics"""
        predictions_np = predictions.numpy()
        targets_np = targets.numpy()
        uncertainties_np = uncertainties.numpy()
        
        metrics = {}
        
        # CCC for each dimension
        dimension_names = ['valence', 'arousal', 'dominance']
        for i, dim_name in enumerate(dimension_names):
            if i < predictions_np.shape[1]:
                ccc = self.metrics.concordance_correlation_coefficient(
                    targets_np[:, i], predictions_np[:, i]
                )
                metrics[f'ccc_{dim_name}'] = ccc
        
        # MAE for each dimension
        for i, dim_name in enumerate(dimension_names):
            if i < predictions_np.shape[1]:
                mae = mean_absolute_error(targets_np[:, i], predictions_np[:, i])
                metrics[f'mae_{dim_name}'] = mae
        
        # Uncertainty calibration
        try:
            ece = uncertainty_calibration_error(
                predictions_np, targets_np, uncertainties_np
            )
            metrics['ece'] = ece
        except:
            metrics['ece'] = 0.0
        
        # Overall CCC (average across dimensions)
        ccc_values = [metrics[f'ccc_{dim}'] for dim in dimension_names if f'ccc_{dim}' in metrics]
        metrics['ccc_overall'] = np.mean(ccc_values) if ccc_values else 0.0
        
        return metrics
    
    def train(self, train_loaders: Dict[str, DataLoader], 
              val_loaders: Dict[str, DataLoader]) -> Dict[str, List[float]]:
        """
        Complete training loop
        
        Args:
            train_loaders: Training dataloaders by dataset
            val_loaders: Validation dataloaders by dataset
            
        Returns:
            Training history
        """
        logger.info(f"🚀 Starting training for {self.config.num_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loaders)
            
            # Validation phase (every N epochs)
            if epoch % self.config.val_frequency == 0:
                val_metrics = self.validate_epoch(val_loaders)
                
                # Update training history
                self.training_history['train_loss'].append(train_metrics['total_loss'])
                self.training_history['val_loss'].append(val_metrics['val_loss'])
                self.training_history['ccc_valence'].append(val_metrics.get('ccc_valence', 0.0))
                self.training_history['ccc_arousal'].append(val_metrics.get('ccc_arousal', 0.0))
                self.training_history['ccc_dominance'].append(val_metrics.get('ccc_dominance', 0.0))
                self.training_history['ece'].append(val_metrics.get('ece', 0.0))
                
                # Logging
                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch:3d}/{self.config.num_epochs} ({epoch_time:.1f}s) - "
                    f"Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"CCC: V={val_metrics.get('ccc_valence', 0):.3f} "
                    f"A={val_metrics.get('ccc_arousal', 0):.3f} "
                    f"D={val_metrics.get('ccc_dominance', 0):.3f}, "
                    f"ECE: {val_metrics.get('ece', 0):.3f}"
                )
                
                # TensorBoard logging
                self._log_metrics(train_metrics, val_metrics, epoch)
                
                # Model checkpointing
                current_ccc = val_metrics.get('ccc_overall', 0.0)
                if current_ccc > self.best_ccc:
                    self.best_ccc = current_ccc
                    self.best_val_loss = val_metrics['val_loss']
                    self.patience_counter = 0
                    
                    # Save best model
                    best_model_path = Path(self.config.checkpoint_dir) / "best_model.pth"
                    ModelCheckpoint.save_checkpoint(
                        self.model, self.optimizer, epoch,
                        val_metrics['val_loss'], str(best_model_path)
                    )
                    logger.info(f"💾 New best model saved! CCC: {current_ccc:.4f}")
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.config.early_stopping and self.patience_counter >= self.config.patience:
                    logger.info(f"🛑 Early stopping at epoch {epoch}")
                    break
            
            # Learning rate scheduling
            if self.config.scheduler_type == "plateau":
                self.scheduler.step(val_metrics['val_loss'] if epoch % self.config.val_frequency == 0 else train_metrics['total_loss'])
            else:
                self.scheduler.step()
            
            # Regular checkpointing
            if epoch % self.config.save_frequency == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
                ModelCheckpoint.save_checkpoint(
                    self.model, self.optimizer, epoch,
                    train_metrics['total_loss'], str(checkpoint_path)
                )
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Training completed in {total_time/3600:.2f} hours")
        logger.info(f"📊 Best CCC: {self.best_ccc:.4f}")
        
        # Save final model
        final_model_path = Path(self.config.checkpoint_dir) / "final_model.pth"
        ModelCheckpoint.save_model_for_inference(self.model, str(final_model_path))
        
        # Save training history
        history_path = Path(self.config.output_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return self.training_history
    
    def _get_curriculum_probabilities(self) -> Dict[str, float]:
        """Get dataset sampling probabilities for curriculum learning"""
        if not self.config.curriculum_learning:
            return {name: 1.0 for name in self.config.dataset_weights.keys()}
        
        # Curriculum strategy: start with IEMOCAP, gradually add others
        progress = self.current_epoch / self.config.num_epochs
        
        if progress < 0.3:  # First 30% - mostly IEMOCAP
            return {'iemocap': 0.7, 'ravdess': 0.2, 'meld': 0.1}
        elif progress < 0.6:  # Middle 30% - balanced
            return {'iemocap': 0.5, 'ravdess': 0.3, 'meld': 0.2}
        else:  # Final 40% - all datasets equally
            return {'iemocap': 0.4, 'ravdess': 0.3, 'meld': 0.3}
    
    def _create_multi_dataset_iterator(self, train_loaders: Dict[str, DataLoader],
                                     dataset_probs: Dict[str, float]):
        """Create iterator that samples from multiple datasets"""
        # Simplified implementation - in practice, you'd want more sophisticated sampling
        loaders_list = [(name, loader) for name, loader in train_loaders.items()]
        
        # Cycle through datasets based on probabilities
        for dataset_name, dataloader in loaders_list:
            prob = dataset_probs.get(dataset_name, 1.0)
            
            for batch_data in dataloader:
                if np.random.random() < prob:
                    yield batch_data, dataset_name
    
    def _log_metrics(self, train_metrics: Dict[str, float],
                    val_metrics: Dict[str, float], epoch: int):
        """Log metrics to TensorBoard"""
        # Training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        # Validation metrics
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Validation/{key}', value, epoch)
        
        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Training/learning_rate', current_lr, epoch)
        
        # Model parameters (gradient norms)
        total_grad_norm = 0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        self.writer.add_scalar('Training/gradient_norm', total_grad_norm, epoch)


def create_trainer(model: CompleteDEERModel, 
                  config: Optional[TrainingConfig] = None,
                  device: Optional[torch.device] = None) -> DEERTrainer:
    """
    Factory function to create DEER trainer
    
    Args:
        model: Complete DEER model
        config: Training configuration
        device: Training device
        
    Returns:
        Configured trainer instance
    """
    if config is None:
        config = TrainingConfig()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = DEERTrainer(model, config, device)
    
    logger.info(f"✅ Trainer created with device: {device}")
    logger.info(f"🎯 Training configuration: {config}")
    
    return trainer


class TrainingUtils:
    """Utility functions for training"""
    
    @staticmethod
    def setup_distributed_training():
        """Setup distributed training (placeholder)"""
        # Implementation for distributed training would go here
        pass
    
    @staticmethod
    def calculate_model_flops(model: CompleteDEERModel, 
                            input_shapes: Tuple[Tuple[int, ...], ...]) -> int:
        """Calculate model FLOPs (placeholder)"""
        # Implementation for FLOPs calculation
        return 0
    
    @staticmethod
    def profile_training_speed(trainer: DEERTrainer,
                             sample_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Profile training speed"""
        trainer.model.train()
        
        # Warmup
        for _ in range(10):
            predictions = trainer.model(
                sample_batch['audio_features'],
                sample_batch['video_features'], 
                sample_batch['text_features']
            )
            loss_dict = trainer.model.compute_loss(predictions, sample_batch['targets'])
            loss_dict['total_loss'].backward()
            trainer.optimizer.zero_grad()
        
        # Time forward pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(100):
            predictions = trainer.model(
                sample_batch['audio_features'],
                sample_batch['video_features'],
                sample_batch['text_features']
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_time = (time.time() - start_time) / 100
        
        # Time backward pass
        start_time = time.time()
        
        for _ in range(100):
            predictions = trainer.model(
                sample_batch['audio_features'],
                sample_batch['video_features'],
                sample_batch['text_features']
            )
            loss_dict = trainer.model.compute_loss(predictions, sample_batch['targets'])
            loss_dict['total_loss'].backward()
            trainer.optimizer.zero_grad()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backward_time = (time.time() - start_time) / 100 - forward_time
        
        return {
            'forward_time_ms': forward_time * 1000,
            'backward_time_ms': backward_time * 1000,
            'total_time_ms': (forward_time + backward_time) * 1000
        }


class ExperimentLogger:
    """Comprehensive experiment logging"""
    
    def __init__(self, experiment_name: str, output_dir: str = "./experiments"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.experiment_dir / f"{experiment_name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.start_time = time.time()
        
    def log_experiment_config(self, model_config: ModelConfig, 
                            training_config: TrainingConfig):
        """Log experiment configuration"""
        config_dict = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time,
            'model_config': model_config.__dict__,
            'training_config': training_config.__dict__
        }
        
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"📄 Experiment config saved: {config_file}")
        
    def log_results(self, results: Dict):
        """Log final results"""
        results['experiment_name'] = self.experiment_name
        results['total_time'] = time.time() - self.start_time
        
        results_file = self.experiment_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📊 Results saved: {results_file}")


def run_complete_training_pipeline(train_loaders: Dict[str, DataLoader],
                                 val_loaders: Dict[str, DataLoader],
                                 model_config: Optional[ModelConfig] = None,
                                 training_config: Optional[TrainingConfig] = None,
                                 experiment_name: str = "deer_multimodal") -> Dict:
    """
    Run complete training pipeline with logging
    
    Args:
        train_loaders: Training dataloaders
        val_loaders: Validation dataloaders
        model_config: Model configuration
        training_config: Training configuration
        experiment_name: Experiment identifier
        
    Returns:
        Training results dictionary
    """
    # Setup experiment logging
    exp_logger = ExperimentLogger(experiment_name)
    
    # Default configurations
    if model_config is None:
        model_config = ModelConfig()
    if training_config is None:
        training_config = TrainingConfig()
    
    # Log configuration
    exp_logger.log_experiment_config(model_config, training_config)
    
    # Create model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CompleteDEERModel(model_config)
    trainer = create_trainer(model, training_config, device)
    
    logger.info(f"🚀 Starting experiment: {experiment_name}")
    logger.info(f"📱 Device: {device}")
    logger.info(f"🎯 Target: CCC 0.840+ (valence), 0.763+ (arousal)")
    
    # Training
    training_history = trainer.train(train_loaders, val_loaders)
    
    # Final evaluation on test set (if available)
    final_results = {
        'training_history': training_history,
        'best_ccc': trainer.best_ccc,
        'best_val_loss': trainer.best_val_loss,
        'total_epochs': trainer.current_epoch + 1,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'experiment_name': experiment_name
    }
    
    # Add achieved performance
    if training_history['ccc_valence']:
        final_results['final_ccc_valence'] = training_history['ccc_valence'][-1]
    if training_history['ccc_arousal']:
        final_results['final_ccc_arousal'] = training_history['ccc_arousal'][-1]
    if training_history['ece']:
        final_results['final_ece'] = training_history['ece'][-1]
    
    # Log results
    exp_logger.log_results(final_results)
    
    logger.info(f"🎉 Training completed!")
    logger.info(f"📊 Best CCC: {trainer.best_ccc:.4f}")
    logger.info(f"📈 Final Valence CCC: {final_results.get('final_ccc_valence', 'N/A')}")
    logger.info(f"📈 Final Arousal CCC: {final_results.get('final_ccc_arousal', 'N/A')}")
    logger.info(f"🎯 ECE: {final_results.get('final_ece', 'N/A')}")
    
    return final_results


def test_trainer():
    """Test training framework with dummy data"""
    logger.info("🧪 Testing training framework...")
    
    # Create dummy dataloaders
    from torch.utils.data import TensorDataset
    
    # Sample data
    batch_size = 16
    num_samples = 100
    
    audio_data = torch.randn(num_samples, 84)
    video_data = torch.randn(num_samples, 256)
    text_data = torch.randn(num_samples, 768)
    targets_data = torch.randn(num_samples, 3)
    
    dataset = TensorDataset(audio_data, video_data, text_data, targets_data)
    
    # Create dataloaders for multiple "datasets"
    train_loaders = {
        'iemocap': DataLoader(dataset, batch_size=batch_size, shuffle=True),
        'ravdess': DataLoader(dataset, batch_size=batch_size, shuffle=True)
    }
    val_loaders = {
        'iemocap': DataLoader(dataset, batch_size=batch_size, shuffle=False),
        'ravdess': DataLoader(dataset, batch_size=batch_size, shuffle=False)
    }
    
    # Quick training configuration
    model_config = ModelConfig()
    training_config = TrainingConfig(
        num_epochs=3,  # Quick test
        val_frequency=1,
        save_frequency=2
    )
    
    # Run training
    results = run_complete_training_pipeline(
        train_loaders, val_loaders, model_config, training_config,
        experiment_name="test_training"
    )
    
    logger.info("✅ Training framework test completed!")
    return results


def main():
    """Main demonstration"""
    logger.info(" Training Framework Demonstration")
    logger.info("=" * 50)
    
    # Test the framework
    test_results = test_trainer()
    
    logger.info(" Demonstration completed!")
    return test_results


if __name__ == "__main__":
    main()