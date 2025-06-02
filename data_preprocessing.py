import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class LeukemiaDataPreprocessor:
    """
    Leukemia gene expression data preprocessor for cVAE training
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.gene_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Store metadata
        self.gene_columns = None
        self.label_mapping = None
        self.n_classes = None
        self.n_genes = None
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load the cleaned leukemia dataset
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Identify gene columns (all except 'samples' and 'type')
        self.gene_columns = [col for col in df.columns if col not in ['samples', 'type']]
        self.n_genes = len(self.gene_columns)
        
        print(f"Dataset loaded: {df.shape[0]} samples, {self.n_genes} genes")
        print(f"Sample distribution:\n{df['type'].value_counts()}")
        
        return df
    
    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        """
        Encode categorical labels to integers
        """
        if not hasattr(self.label_encoder, 'classes_'):
            # First time - fit the encoder
            encoded_labels = self.label_encoder.fit_transform(labels)
            self.label_mapping = dict(zip(self.label_encoder.classes_, 
                                        self.label_encoder.transform(self.label_encoder.classes_)))
            self.n_classes = len(self.label_encoder.classes_)
            
            print(f"Label encoding mapping:")
            for original, encoded in self.label_mapping.items():
                print(f"  {original} -> {encoded}")
        else:
            # Already fitted - just transform
            encoded_labels = self.label_encoder.transform(labels)
            
        return encoded_labels
    
    def split_data(self, df: pd.DataFrame, 
                   train_ratio: float = 0.8, 
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets with stratification
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # First split: train vs temp (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - train_ratio),
            random_state=self.random_state,
            stratify=df['type']
        )
        
        # Second split: val vs test
        if val_ratio > 0 and test_ratio > 0:
            val_size_in_temp = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - val_size_in_temp),
                random_state=self.random_state,
                stratify=temp_df['type']
            )
        elif val_ratio > 0:
            val_df = temp_df
            test_df = pd.DataFrame()
        else:
            val_df = pd.DataFrame()
            test_df = temp_df
        
        print(f"\nData split completed:")
        print(f"Train set: {len(train_df)} samples")
        if not val_df.empty:
            print(f"Validation set: {len(val_df)} samples")
        if not test_df.empty:
            print(f"Test set: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def normalize_features(self, train_data: np.ndarray, 
                          val_data: np.ndarray = None, 
                          test_data: np.ndarray = None) -> Tuple[np.ndarray, ...]:
        """
        Normalize gene expression features using StandardScaler
        Fit on training data only, then transform all sets
        """
        print("Normalizing gene expression features...")
        
        # Fit scaler on training data only
        train_normalized = self.gene_scaler.fit_transform(train_data)
        
        results = [train_normalized]
        
        if val_data is not None:
            val_normalized = self.gene_scaler.transform(val_data)
            results.append(val_normalized)
        
        if test_data is not None:
            test_normalized = self.gene_scaler.transform(test_data)
            results.append(test_normalized)
        
        print(f"Feature normalization completed.")
        print(f"Training data stats - Mean: {train_normalized.mean():.3f}, Std: {train_normalized.std():.3f}")
        
        return tuple(results)
    
    def prepare_data_for_cvae(self, df: pd.DataFrame, 
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.1,
                             test_ratio: float = 0.1) -> Dict[str, Any]:
        """
        Complete data preparation pipeline for cVAE training
        """
        print("=" * 60)
        print("STARTING DATA PREPROCESSING FOR cVAE")
        print("=" * 60)
        
        # Step 1: Split data
        train_df, val_df, test_df = self.split_data(df, train_ratio, val_ratio, test_ratio)
        
        # Step 2: Encode labels
        train_labels_encoded = self.encode_labels(train_df['type'])
        val_labels_encoded = self.encode_labels(val_df['type']) if not val_df.empty else None
        test_labels_encoded = self.encode_labels(test_df['type']) if not test_df.empty else None
        
        # Step 3: Extract and normalize features
        train_features = train_df[self.gene_columns].values
        val_features = val_df[self.gene_columns].values if not val_df.empty else None
        test_features = test_df[self.gene_columns].values if not test_df.empty else None
        
        # Normalize features
        normalized_data = self.normalize_features(train_features, val_features, test_features)
        train_features_norm = normalized_data[0]
        val_features_norm = normalized_data[1] if len(normalized_data) > 1 and val_features is not None else None
        test_features_norm = normalized_data[2] if len(normalized_data) > 2 and test_features is not None else None
        
        # Prepare result dictionary
        result = {
            'train': {
                'features': train_features_norm,
                'labels': train_labels_encoded,
                'original_labels': train_df['type'].values,
                'sample_ids': train_df['samples'].values
            },
            'metadata': {
                'n_samples_train': len(train_df),
                'n_genes': self.n_genes,
                'n_classes': self.n_classes,
                'label_mapping': self.label_mapping,
                'gene_columns': self.gene_columns,
                'class_distribution_train': train_df['type'].value_counts().to_dict()
            }
        }
        
        if val_features_norm is not None:
            result['validation'] = {
                'features': val_features_norm,
                'labels': val_labels_encoded,
                'original_labels': val_df['type'].values,
                'sample_ids': val_df['samples'].values
            }
            result['metadata']['n_samples_val'] = len(val_df)
            result['metadata']['class_distribution_val'] = val_df['type'].value_counts().to_dict()
        
        if test_features_norm is not None:
            result['test'] = {
                'features': test_features_norm,
                'labels': test_labels_encoded,
                'original_labels': test_df['type'].values,
                'sample_ids': test_df['samples'].values
            }
            result['metadata']['n_samples_test'] = len(test_df)
            result['metadata']['class_distribution_test'] = test_df['type'].value_counts().to_dict()
        
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING COMPLETED")
        print("=" * 60)
        
        return result
    
    def save_preprocessor(self, save_dir: str):
        """
        Save the preprocessor components for later use
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save scaler
        with open(os.path.join(save_dir, 'gene_scaler.pkl'), 'wb') as f:
            pickle.dump(self.gene_scaler, f)
        
        # Save label encoder
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save metadata
        metadata = {
            'gene_columns': self.gene_columns,
            'label_mapping': self.label_mapping,
            'n_classes': self.n_classes,
            'n_genes': self.n_genes
        }
        
        with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Preprocessor saved to {save_dir}")
    
    def load_preprocessor(self, save_dir: str):
        """
        Load previously saved preprocessor components
        """
        # Load scaler
        with open(os.path.join(save_dir, 'gene_scaler.pkl'), 'rb') as f:
            self.gene_scaler = pickle.load(f)
        
        # Load label encoder
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load metadata
        with open(os.path.join(save_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            self.gene_columns = metadata['gene_columns']
            self.label_mapping = metadata['label_mapping']
            self.n_classes = metadata['n_classes']
            self.n_genes = metadata['n_genes']
        
        print(f"Preprocessor loaded from {save_dir}")

def main():
    """
    Main function to demonstrate data preprocessing
    """
    # Initialize preprocessor
    preprocessor = LeukemiaDataPreprocessor(random_state=42)
    
    # Load data
    df = preprocessor.load_data('datasets/raw_data.csv')
    
    # Prepare data for cVAE training
    processed_data = preprocessor.prepare_data_for_cvae(
        df, 
        train_ratio=0.8, 
        val_ratio=0.1, 
        test_ratio=0.1
    )
    
    # Save preprocessor for later use
    preprocessor.save_preprocessor('datasets')
    
    # Display summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Training samples: {processed_data['metadata']['n_samples_train']}")
    if 'validation' in processed_data:
        print(f"Validation samples: {processed_data['metadata']['n_samples_val']}")
    if 'test' in processed_data:
        print(f"Test samples: {processed_data['metadata']['n_samples_test']}")
    print(f"Number of genes: {processed_data['metadata']['n_genes']}")
    print(f"Number of classes: {processed_data['metadata']['n_classes']}")
    
    print(f"\nClass distribution in training set:")
    for class_name, count in processed_data['metadata']['class_distribution_train'].items():
        print(f"  {class_name}: {count} samples")
    
    print(f"\nLabel encoding:")
    for original, encoded in processed_data['metadata']['label_mapping'].items():
        print(f"  {original} -> {encoded}")
    
    return processed_data

if __name__ == "__main__":
    processed_data = main() 