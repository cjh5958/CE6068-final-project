"""
Leukemia cVAE Model Usage Guide
==============================

This script demonstrates how to load and use the trained cVAE model 
for generating new synthetic leukemia gene expression samples.
"""

import os
import pickle
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to path to import custom modules
import sys
sys.path.append('..')
from cvae_model import ConditionalVAE

class LeukemiaCVAEGenerator:
    """Leukemia cVAE Data Generator
    
    Simple interface for generating synthetic leukemia gene expression data
    using pre-trained conditional VAE model.
    """
    
    def __init__(self):
        # Update paths to new directory structure
        self.model_path = os.path.join('models', 'leukemia_cvae_model.pth')
        self.scaler_path = os.path.join('datasets', 'scaler.pkl')
        self.label_encoder_path = os.path.join('datasets', 'label_encoder.pkl')
        self.metadata_path = os.path.join('datasets', 'metadata.pkl')
        self.pca_path = 'datasets/pca.pkl'
        
        self.device = torch.device('cpu')
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.class_names = ['AML', 'Bone_Marrow', 'Bone_Marrow_CD34', 'PB', 'PBSC_CD34']
        self.gene_names = None
        
        self._load_components()
    
    def _load_components(self):
        """Load model and preprocessing components"""
        try:
            # Load PCA
            if os.path.exists(self.pca_path):
                with open(self.pca_path, 'rb') as f:
                    self.pca = pickle.load(f)
                print(f"✅ PCA loaded successfully (n_components={self.pca.n_components_}, original_dim={self.pca.components_.shape[1]})")
                input_dim = self.pca.n_components_
            else:
                self.pca = None
                print("⚠️  PCA not found, cannot restore to original dimension")
                input_dim = 32  # fallback
            # 直接用固定 hidden_dims 或從模型檔案 config 讀取
            encoder_hidden_dims = [64, 32]
            decoder_hidden_dims = [32, 64]
            # Load model
            if os.path.exists(self.model_path):
                from cvae_model import ConditionalVAE
                checkpoint = torch.load(self.model_path, 
                                      map_location=self.device, 
                                      weights_only=False)
                self.model = ConditionalVAE(
                    input_dim=input_dim,
                    condition_dim=5,
                    latent_dim=16,
                    encoder_hidden_dims=encoder_hidden_dims,
                    decoder_hidden_dims=decoder_hidden_dims
                )
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                print("✅ Model loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load scaler
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✅ Scaler loaded successfully")
            else:
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            
            # Load label encoder
            if os.path.exists(self.label_encoder_path):
                with open(self.label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print("✅ Label encoder loaded successfully")
            else:
                print("⚠️  Label encoder not found, using default mapping")
            
            # Load metadata (including gene names)
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                if 'gene_names' in self.metadata:
                    self.gene_names = self.metadata['gene_names']
                    print(f"✅ Metadata loaded successfully ({len(self.gene_names)} gene names)")
                else:
                    print("⚠️  Gene names not found in metadata, using default names")
                    self.gene_names = [f'gene_{i}' for i in range(22283)]
            else:
                print("⚠️  Metadata not found, using default gene names")
                self.gene_names = [f'gene_{i}' for i in range(22283)]
                
        except Exception as e:
            print(f"❌ Error loading components: {e}")
            raise
    
    def generate_samples(self, leukemia_type, n_samples=10):
        """Generate synthetic samples for specified leukemia type
        
        Args:
            leukemia_type (str): Type of leukemia to generate
            n_samples (int): Number of samples to generate
            
        Returns:
            dict: Generated data with metadata
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model components not properly loaded")
        
        if leukemia_type not in self.class_names:
            raise ValueError(f"Invalid leukemia type. Choose from: {self.class_names}")
        
        # Get class label
        class_idx = self.class_names.index(leukemia_type)
        
        # Create one-hot encoded condition
        condition = torch.zeros(1, 5)  # 5 classes
        condition[0, class_idx] = 1.0
        
        # Generate samples using model's generate method
        with torch.no_grad():
            generated_samples = self.model.generate(condition, n_samples=n_samples)
            
        # 直接輸出 32 維特徵，不做 scaler.inverse_transform
        generated_samples = generated_samples.cpu().numpy()
        return {
            'features': generated_samples,
            'labels': np.full(n_samples, class_idx),
            'class_names': [leukemia_type] * n_samples,
            'n_samples': n_samples,
            'leukemia_type': leukemia_type,
            'generation_time': datetime.now().isoformat()
        }
    
    def generate_balanced_dataset(self, samples_per_class=50):
        """Generate balanced dataset with equal samples per class
        
        Args:
            samples_per_class (int): Number of samples per leukemia type
            
        Returns:
            dict: Balanced dataset with metadata
        """
        all_features = []
        all_labels = []
        all_class_names = []
        
        for i, class_name in enumerate(self.class_names):
            data = self.generate_samples(class_name, samples_per_class)
            all_features.append(data['features'])
            all_labels.extend([i] * samples_per_class)
            all_class_names.extend([class_name] * samples_per_class)
        
        all_features = np.vstack(all_features)
        
        return {
            'features': all_features,
            'labels': np.array(all_labels),
            'class_names': all_class_names,
            'total_samples': len(all_features),
            'samples_per_class': samples_per_class,
            'generation_time': datetime.now().isoformat()
        }
    
    def save_data(self, data, output_prefix='gen_data', save_format='both'):
        """Save generated data to files
        
        Args:
            data (dict): Generated data
            output_prefix (str): Prefix for output files
            save_format (str): 'csv', 'pkl', or 'both'
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {}
        
        # Prepare output directory
        output_dir = 'datasets'
        os.makedirs(output_dir, exist_ok=True)
        
        if save_format in ['csv', 'both']:
            # Create DataFrame with PCA features
            df = pd.DataFrame(data['features'])
            # 欄位名稱：PCA_0, PCA_1, ...
            pca_columns = [f'PCA_{i}' for i in range(df.shape[1])]
            df.columns = pca_columns
            # Add samples column (starting from next available number)
            start_sample_id = 1000
            sample_ids = list(range(start_sample_id, start_sample_id + len(df)))
            df.insert(0, 'type', data['class_names'])
            df.insert(0, 'samples', sample_ids)
            # Save CSV
            csv_path = os.path.join(output_dir, f'gen_data.csv')
            df.to_csv(csv_path, index=False)
            results['csv_file'] = csv_path
            print(f"✅ CSV saved: {csv_path}")
            print(f"   Format: samples, type, {len(pca_columns)} PCA columns")
        
        if save_format in ['pkl', 'both']:
            pkl_path = os.path.join(output_dir, f'gen_data_complete.pkl')
            complete_data = {
                **data,
                'metadata': {
                    'total_pca_dim': data['features'].shape[1],
                    'pca_columns': [f'PCA_{i}' for i in range(data['features'].shape[1])],
                    'available_classes': self.class_names,
                    'save_time': datetime.now().isoformat()
                }
            }
            
            with open(pkl_path, 'wb') as f:
                pickle.dump(complete_data, f)
            results['pkl_file'] = pkl_path
            print(f"✅ PKL saved: {pkl_path}")
        
        return results
    
    def quick_generate_and_save(self, leukemia_type, n_samples=10, 
                               output_prefix=None, save_format='both'):
        """One-click generation and saving"""
        if output_prefix is None:
            output_prefix = 'gen_data'
        
        print(f"🧬 Generating {n_samples} {leukemia_type} samples...")
        data = self.generate_samples(leukemia_type, n_samples)
        
        print(f"💾 Saving data...")
        files = self.save_data(data, output_prefix, save_format)
        
        print(f"✅ Generation complete!")
        print(f"   - Samples: {data['n_samples']}")
        print(f"   - Type: {data['leukemia_type']}")
        print(f"   - Files: {list(files.values())}")
        
        return data
    
    def quick_generate_balanced_and_save(self, samples_per_class=50, 
                                        output_prefix='gen_data',
                                        save_format='both'):
        """One-click balanced dataset generation and saving"""
        total_samples = samples_per_class * 5
        print(f"🧬 Generating balanced dataset ({samples_per_class} samples per type)...")
        print(f"   Total: {total_samples} samples across 5 leukemia types")
        
        data = self.generate_balanced_dataset(samples_per_class)
        
        print(f"💾 Saving data...")
        files = self.save_data(data, output_prefix, save_format)
        
        print(f"✅ Generation complete!")
        print(f"   - Total samples: {data['total_samples']}")
        print(f"   - Per class: {data['samples_per_class']}")
        print(f"   - Files: {list(files.values())}")
        
        return data

# Simple example usage
if __name__ == "__main__":
    print("🧬 Leukemia cVAE Data Generator")
    print("=" * 50)
    
    try:
        # Initialize generator
        generator = LeukemiaCVAEGenerator()
        
        # Example 1: Generate AML samples
        print("\n📊 Example 1: Generate 10 AML samples")
        aml_data = generator.quick_generate_and_save(
            leukemia_type='AML',
            n_samples=10,
            save_format='csv'
        )
        
        # Example 2: Generate balanced dataset
        print("\n📊 Example 2: Generate balanced dataset")
        balanced_data = generator.quick_generate_balanced_and_save(
            samples_per_class=20,
            save_format='both'
        )
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure 'models/leukemia_cvae_model.pth' exists")
        print("2. Ensure 'datasets/scaler.pkl' exists")
        print("3. Check that virtual environment is activated")
