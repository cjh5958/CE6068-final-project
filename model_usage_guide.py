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
            # Load metadata (取得 gene_names)
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                if 'gene_names' in self.metadata:
                    self.gene_names = self.metadata['gene_names']
                    print(f"✅ Metadata loaded successfully ({len(self.gene_names)} gene names)")
                else:
                    raise ValueError('gene_names not found in metadata')
            else:
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            # Load model
            if os.path.exists(self.model_path):
                from cvae_model import ConditionalVAE
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model = ConditionalVAE(
                    input_dim=len(self.gene_names),
                    condition_dim=5,
                    latent_dim=32,
                    encoder_hidden_dims=[128, 64],
                    decoder_hidden_dims=[64, 128]
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
            
        # Convert to numpy and denormalize
        generated_samples = generated_samples.cpu().numpy()
        generated_samples = self.scaler.inverse_transform(generated_samples)
        
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
    
    def save_data(self, data, output_prefix='leukemia_data', save_format='both'):
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
            # Create DataFrame with gene features
            df = pd.DataFrame(data['features'])
            df.columns = self.gene_names
            # Add samples column (starting from next available number)
            start_sample_id = 1000
            sample_ids = list(range(start_sample_id, start_sample_id + len(df)))
            df.insert(0, 'samples', sample_ids)
            df.insert(1, 'type', data['class_names'])
            column_order = ['samples', 'type'] + self.gene_names
            df = df[column_order]
            csv_path = os.path.join(output_dir, f'{output_prefix}_{timestamp}.csv')
            df.to_csv(csv_path, index=False)
            results['csv_file'] = csv_path
            print(f"✅ CSV saved: {csv_path}")
            print(f"   Format: samples, type, {len(self.gene_names)} gene columns")
        
        if save_format in ['pkl', 'both']:
            # Save PKL with complete metadata
            pkl_path = os.path.join(output_dir, f'{output_prefix}_complete_{timestamp}.pkl')
            complete_data = {
                **data,
                'metadata': {
                    'total_genes': data['features'].shape[1],
                    'gene_names': self.gene_names,
                    'available_classes': self.class_names,
                    'save_time': datetime.now().isoformat()
                }
            }
            
            with open(pkl_path, 'wb') as f:
                pickle.dump(complete_data, f)
            results['pkl_file'] = pkl_path
            print(f"✅ PKL saved: {pkl_path}")
        
        return results
    
    def augment_from_raw_samples(self, n_augments=1, balanced=False, samples_per_class=10):
        """
        針對原始樣本進行 latent space 擴增
        Args:
            n_augments: 每個原始樣本擴增幾倍
            balanced: 是否 balanced 模式
            samples_per_class: balanced 模式下每類要產生幾個樣本
        Returns:
            dict: augmented data
        """
        # 讀取降維後原始資料
        df = pd.read_csv('datasets/raw_data_reduced.csv')
        features = df[self.gene_names].values
        labels = df['type'].values
        # 轉換 label 為 index
        if self.label_encoder is not None:
            label_indices = self.label_encoder.transform(labels)
        else:
            label_indices = np.array([self.class_names.index(t) for t in labels])
        # 標準化
        features_norm = self.scaler.transform(features)
        device = self.device
        all_aug_features = []
        all_aug_labels = []
        all_aug_class_names = []
        if balanced:
            # 每類各取原始樣本，隨機抽樣
            for class_idx, class_name in enumerate(self.class_names):
                class_mask = (label_indices == class_idx)
                class_features = features_norm[class_mask]
                n_orig = class_features.shape[0]
                n_needed = samples_per_class
                # 若原始樣本不足，重複抽
                idxs = np.random.choice(n_orig, n_needed, replace=(n_needed > n_orig))
                sel_features = class_features[idxs]
                sel_labels = np.full(n_needed, class_idx)
                # encode
                cond = torch.zeros(n_needed, len(self.class_names), device=device)
                cond[range(n_needed), class_idx] = 1.0
                x_tensor = torch.tensor(sel_features, dtype=torch.float32, device=device)
                with torch.no_grad():
                    mu, logvar = self.model.encode(x_tensor, cond)
                    # latent 擴增
                    z_aug = mu + torch.randn_like(mu) * 0.2
                    # decode
                    x_aug = self.model.decode(z_aug, cond)
                    x_aug = x_aug.cpu().numpy()
                    x_aug = self.scaler.inverse_transform(x_aug)
                all_aug_features.append(x_aug)
                all_aug_labels.extend(sel_labels)
                all_aug_class_names.extend([class_name]*n_needed)
            all_aug_features = np.vstack(all_aug_features)
            return {
                'features': all_aug_features,
                'labels': np.array(all_aug_labels),
                'class_names': all_aug_class_names,
                'total_samples': len(all_aug_labels),
                'samples_per_class': samples_per_class,
                'generation_time': datetime.now().isoformat()
            }
        else:
            # 非 balanced 模式，針對所有原始樣本擴增 n_augments 倍
            n_orig = features_norm.shape[0]
            aug_features = []
            aug_labels = []
            aug_class_names = []
            for i in range(n_orig):
                x = features_norm[i:i+1]
                label = label_indices[i]
                class_name = labels[i]
                cond = torch.zeros(1, len(self.class_names), device=device)
                cond[0, label] = 1.0
                x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
                with torch.no_grad():
                    mu, logvar = self.model.encode(x_tensor, cond)
                    for _ in range(n_augments):
                        z_aug = mu + torch.randn_like(mu) * 0.2
                        x_aug = self.model.decode(z_aug, cond)
                        x_aug = x_aug.cpu().numpy()
                        x_aug = self.scaler.inverse_transform(x_aug)
                        aug_features.append(x_aug[0])
                        aug_labels.append(label)
                        aug_class_names.append(class_name)
            aug_features = np.array(aug_features)
            return {
                'features': aug_features,
                'labels': np.array(aug_labels),
                'class_names': aug_class_names,
                'n_samples': len(aug_labels),
                'generation_time': datetime.now().isoformat()
            }

    def quick_generate_and_save(self, leukemia_type, n_samples=10, 
                               output_prefix=None, save_format='both'):
        """針對原始樣本進行 latent 擴增，僅產生指定類別"""
        if output_prefix is None:
            output_prefix = f'{leukemia_type.lower()}_samples'
        # 只取該類型的原始樣本，並擴增
        print(f"🧬 Augmenting {n_samples} {leukemia_type} samples from raw data...")
        data = self.augment_from_raw_samples(n_augments=n_samples, balanced=False)
        # 篩選指定類別
        class_idx = self.class_names.index(leukemia_type)
        mask = (data['labels'] == class_idx)
        data['features'] = data['features'][mask]
        data['labels'] = data['labels'][mask]
        data['class_names'] = [leukemia_type]*np.sum(mask)
        data['n_samples'] = np.sum(mask)
        print(f"💾 Saving data...")
        files = self.save_data(data, output_prefix, save_format)
        print(f"✅ Augmentation complete!")
        print(f"   - Samples: {data['n_samples']}")
        print(f"   - Type: {leukemia_type}")
        print(f"   - Files: {list(files.values())}")
        return data

    def quick_generate_balanced_and_save(self, samples_per_class=50, 
                                        output_prefix='balanced_leukemia_dataset',
                                        save_format='both'):
        """針對原始樣本進行 latent 擴增，balanced 模式"""
        print(f"🧬 Augmenting balanced dataset ({samples_per_class} samples per type) from raw data...")
        data = self.augment_from_raw_samples(n_augments=1, balanced=True, samples_per_class=samples_per_class)
        print(f"💾 Saving data...")
        files = self.save_data(data, output_prefix, save_format)
        print(f"✅ Augmentation complete!")
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
