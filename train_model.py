#!/usr/bin/env python3
"""
Leukemia cVAE Model Training Script
==================================

This script provides a complete training pipeline for the conditional VAE model
used for leukemia gene expression data augmentation.

Usage:
    python train_model.py [--config config.yaml]
"""

import os
import sys
import argparse
import warnings
import json
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from sklearn.ensemble import RandomForestClassifier

# Import our modules
from cvae_model import ConditionalVAE
from cvae_trainer import CVAETrainer
from data_preprocessing import LeukemiaDataPreprocessor

warnings.filterwarnings('ignore')

class ModelTrainingPipeline:
    """Complete training pipeline for leukemia cVAE model"""
    
    def __init__(self, config=None):
        """Initialize training pipeline
        
        Args:
            config (dict): Training configuration parameters
        """
        # Default configuration
        self.config = {
            'data': {
                'input_file': 'datasets/raw_data.csv',
                'test_size': 0.15,
                'val_size': 0.12,
                'random_state': 42
            },
            'model': {
                'input_dim': 200,
                'condition_dim': 5,
                'latent_dim': 32,
                'encoder_hidden_dims': [128, 64],
                'decoder_hidden_dims': [64, 128]
            },
            'training': {
                'batch_size': 8,
                'learning_rate': 1e-4,
                'epochs': 100,
                'patience': 20,
                'beta_schedule': 'linear',
                'beta_start': 0.1,
                'beta_end': 1.0,
                'device': 'auto'
            },
            'output': {
                'model_dir': 'models',
                'logs_dir': 'training_logs',
                'save_best': True,
                'save_checkpoints': True
            }
        }
        
        # Update with provided config
        if config:
            self._update_config(config)
        
        # Setup directories
        self._setup_directories()
        
        # Initialize components
        self.device = self._get_device()
        self.preprocessor = None
        self.model = None
        self.trainer = None
        
    def _update_config(self, new_config):
        """Recursively update configuration"""
        def update_dict(base_dict, new_dict):
            for key, value in new_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_dict(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        update_dict(self.config, new_config)
    
    def _setup_directories(self):
        """Create necessary directories"""
        Path(self.config['output']['model_dir']).mkdir(exist_ok=True)
        Path(self.config['output']['logs_dir']).mkdir(exist_ok=True)
        Path('datasets').mkdir(exist_ok=True)
    
    def _get_device(self):
        """Get training device"""
        if self.config['training']['device'] == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(self.config['training']['device'])
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("🔄 Loading and preprocessing data...")
        
        # Check if data file exists
        data_file = self.config['data']['input_file']
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load data
        df = pd.read_csv(data_file)
        print(f"✅ Loaded data: {df.shape[0]} samples, {df.shape[1]-1} features")
        
        # Separate features and labels
        if 'type' in df.columns:
            # Extract gene column names (excluding 'samples' and 'type')
            gene_columns = [col for col in df.columns if col not in ['samples', 'type']]
            X = df[gene_columns].values
            y = df['type'].values
        else:
            raise ValueError("No 'type' column found in the dataset")
        
        # Check data quality
        print(f"📊 Data summary:")
        print(f"   - Features: {X.shape[1]}")
        print(f"   - Samples: {X.shape[0]}")
        print(f"   - Classes: {len(np.unique(y))}")
        print(f"   - Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"   - Gene columns: {len(gene_columns)} genes")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config['data']['val_size'] / (1 - self.config['data']['test_size']),
            random_state=self.config['data']['random_state'],
            stratify=y_temp
        )
        
        print(f"📊 Data splits:")
        print(f"   - Training: {X_train.shape[0]} samples")
        print(f"   - Validation: {X_val.shape[0]} samples") 
        print(f"   - Test: {X_test.shape[0]} samples")
        
        # Preprocessing
        print("🔄 Applying preprocessing...")
        
        # Label encoding
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)
        
        # ===== 新增：特徵選擇 =====
        n_selected = 500  # 可調整
        print(f"🔍 進行特徵選擇（隨機森林，選前 {n_selected} 個基因）...")
        rf = RandomForestClassifier(n_estimators=100, random_state=self.config['data']['random_state'])
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:n_selected]
        selected_genes = [gene_columns[i] for i in indices]
        # 儲存特徵選擇結果
        with open('datasets/selected_genes.txt', 'w', encoding='utf-8') as f:
            for g in selected_genes:
                f.write(g + '\n')
        print(f"✅ 已選出 {n_selected} 個基因特徵，已存至 datasets/selected_genes.txt")
        # 只保留選中的特徵
        X_train = X_train[:, indices]
        X_val = X_val[:, indices]
        X_test = X_test[:, indices]
        gene_columns = selected_genes
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        # 儲存 scaler
        with open('datasets/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        # 儲存 label_encoder
        with open('datasets/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        # Save metadata including gene names (特徵選擇後)
        metadata = {
            'class_names': label_encoder.classes_.tolist(),
            'gene_names': gene_columns,  # 這時候是 200 維
            'n_features': X_train.shape[1],
            'n_classes': len(label_encoder.classes_),
            'feature_range': [float(X_train.min()), float(X_train.max())],
            'training_samples': X_train.shape[0],
            'preprocessing_date': datetime.now().isoformat()
        }
        # 儲存 metadata
        with open('datasets/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        # input_dim 只根據 n_selected 設定
        X_train_final, X_val_final, X_test_final = X_train_scaled, X_val_scaled, X_test_scaled
        feature_columns = gene_columns
        self.config['model']['input_dim'] = n_selected
        # 儲存降維後的raw_data_reduced.csv
        all_X = np.vstack([X_train_final, X_val_final, X_test_final])
        all_y_encoded = np.concatenate([y_train_encoded, y_val_encoded, y_test_encoded])
        all_samples = np.arange(1000, 1000 + all_X.shape[0])
        df_reduced = pd.DataFrame(all_X, columns=feature_columns)
        df_reduced.insert(0, 'type', label_encoder.inverse_transform(all_y_encoded))
        df_reduced.insert(0, 'samples', all_samples)
        df_reduced.to_csv('datasets/raw_data_reduced.csv', index=False)
        print(f"✅ Saved reduced data to datasets/raw_data_reduced.csv, shape: {df_reduced.shape}")
        return {
            'X_train': X_train_final,
            'X_val': X_val_final,
            'X_test': X_test_final,
            'y_train': y_train_encoded,
            'y_val': y_val_encoded,
            'y_test': y_test_encoded,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'metadata': metadata
        }
    
    def create_model(self):
        """Create the cVAE model"""
        print("🔄 Creating cVAE model...")
        
        model_kwargs = dict(self.config['model'])
        if 'lambda_class' not in model_kwargs:
            model_kwargs['lambda_class'] = 1.0
        model = ConditionalVAE(**model_kwargs).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created:")
        print(f"   - Device: {self.device}")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        return model
    
    def train_model(self, data):
        """Train the cVAE model"""
        print("🔄 Starting model training...")
        
        # Create model
        self.model = self.create_model()
        
        # Create trainer
        self.trainer = CVAETrainer(
            model=self.model,
            device=self.device,
            learning_rate=self.config['training']['learning_rate'],
            beta_schedule=self.config['training']['beta_schedule'],
            beta_start=self.config['training']['beta_start'],
            beta_end=self.config['training']['beta_end']
        )
        
        # Prepare training data
        train_data = {
            'features': data['X_train'],
            'labels': data['y_train']
        }
        
        val_data = {
            'features': data['X_val'],
            'labels': data['y_val']
        }
        
        # Training parameters
        training_params = {
            'epochs': self.config['training']['epochs'],
            'batch_size': self.config['training']['batch_size'],
            'early_stopping_patience': self.config['training']['patience'],
            'save_dir': self.config['output']['model_dir']
        }
        
        # Start training
        training_history = self.trainer.train(
            train_data=train_data,
            val_data=val_data,
            **training_params
        )
        
        # Save final model
        model_path = os.path.join(self.config['output']['model_dir'], 'leukemia_cvae_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metadata': data['metadata'],
            'training_history': training_history
        }, model_path)
        
        print(f"✅ Training completed!")
        print(f"   - Model saved: {model_path}")
        print(f"   - Training logs: {self.config['output']['logs_dir']}")
        
        return training_history
    
    def evaluate_model(self, data):
        """Evaluate the trained model"""
        print("🔄 Evaluating model...")
        
        if self.model is None:
            raise RuntimeError("Model not trained yet. Please run train_model first.")
        
        # Test data evaluation
        test_features = torch.FloatTensor(data['X_test']).to(self.device)
        test_labels = torch.LongTensor(data['y_test']).to(self.device)
        
        # Create condition tensor
        from cvae_model import create_condition_tensor
        test_conditions = create_condition_tensor(
            data['y_test'], 
            self.config['model']['condition_dim'], 
            self.device
        )
        
        # Evaluate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(test_features, test_conditions)
            loss_dict = self.model.compute_loss(test_features, test_conditions)
        
        print(f"✅ Model evaluation:")
        print(f"   - Test reconstruction loss: {loss_dict['recon_loss'].item():.4f}")
        print(f"   - Test KL divergence: {loss_dict['kl_loss'].item():.4f}")
        print(f"   - Test total loss: {loss_dict['total_loss'].item():.4f}")
        
        return loss_dict
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("🚀 Starting complete training pipeline...")
        print("=" * 60)
        
        try:
            # Step 1: Load and preprocess data
            data = self.load_and_preprocess_data()
            
            # Step 2: Train model
            training_history = self.train_model(data)
            
            # Step 3: Evaluate model
            evaluation_results = self.evaluate_model(data)
            
            print("=" * 60)
            print("🎉 Training pipeline completed successfully!")
            
            # Summary
            print(f"\n📋 Training Summary:")
            print(f"   - Final training loss: {training_history['train_loss'][-1]:.4f}")
            print(f"   - Final validation loss: {training_history['val_loss'][-1]:.4f}")
            print(f"   - Best validation loss: {min(training_history['val_loss']):.4f}")
            print(f"   - Training epochs: {len(training_history['train_loss'])}")
            print(f"   - Model saved in: {self.config['output']['model_dir']}")
            
            return {
                'training_history': training_history,
                'evaluation_results': evaluation_results,
                'data_info': data['metadata']
            }
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {e}")
            raise

def load_config_from_file(config_file):
    """Load configuration from YAML or JSON file"""
    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            print("⚠️  PyYAML not installed. Please install it to use YAML config files.")
            return None
    elif config_file.endswith('.json'):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("⚠️  Unsupported config file format. Use .yaml, .yml, or .json")
        return None

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Leukemia cVAE Model')
    parser.add_argument('--config', type=str, help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent space dimension')
    parser.add_argument('--device', type=str, default='auto', help='Training device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config_from_file(args.config)
        if config is None:
            print("❌ Failed to load configuration file")
            return
    
    # Override with command line arguments
    if args.epochs != 100:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size != 8:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.learning_rate != 1e-4:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
    if args.latent_dim != 256:
        config.setdefault('model', {})['latent_dim'] = args.latent_dim
    if args.device != 'auto':
        config.setdefault('training', {})['device'] = args.device
    
    # Initialize and run training pipeline
    pipeline = ModelTrainingPipeline(config)
    
    try:
        results = pipeline.run_full_pipeline()
        print("\n🎯 Training completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Use model_usage_guide.py to generate synthetic samples")
        print("   2. Evaluate the quality of generated samples")
        print("   3. Use augmented data for downstream tasks")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check if datasets/raw_data.csv exists")
        print("   2. Ensure you have enough memory (recommended: 8GB+)")
        print("   3. Try reducing batch size if out of memory")
        print("   4. Check virtual environment activation")

if __name__ == "__main__":
    main() 