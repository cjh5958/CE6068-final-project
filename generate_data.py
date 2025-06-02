#!/usr/bin/env python3
"""
Leukemia cVAE Data Generation Script
===================================

This script provides a simple interface for generating synthetic leukemia 
gene expression data using pre-trained conditional VAE models.

Usage:
    python generate_data.py --type AML --samples 100
    python generate_data.py --balanced --samples-per-class 50
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Import our model usage guide
from model_usage_guide import LeukemiaCVAEGenerator

def create_distribution_plots(data, output_dir, filename_prefix):
    """
    Create comprehensive distribution analysis plots for generated data
    
    Args:
        data (dict): Generated data containing features, labels, class_names
        output_dir (str): Directory to save plots
        filename_prefix (str): Prefix for plot filenames
    """
    print("🎨 Creating distribution analysis plots...")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    features = data['features']
    labels = data['labels']
    class_names = np.array(data['class_names'])
    unique_classes = np.unique(class_names)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Class Distribution Pie Chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    class_counts = pd.Series(class_names).value_counts()
    colors = sns.color_palette("husl", len(class_counts))
    
    wedges, texts, autotexts = ax.pie(class_counts.values, 
                                     labels=class_counts.index, 
                                     autopct='%1.1f%%',
                                     colors=colors,
                                     startangle=90)
    
    ax.set_title('Generated Data - Class Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{filename_prefix}_class_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Statistics Summary
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature mean distribution
    feature_means = np.mean(features, axis=1)
    axes[0, 0].hist(feature_means, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Sample Means', fontweight='bold')
    axes[0, 0].set_xlabel('Mean Expression Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Feature variance distribution
    feature_vars = np.var(features, axis=1)
    axes[0, 1].hist(feature_vars, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Distribution of Sample Variances', fontweight='bold')
    axes[0, 1].set_xlabel('Variance')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gene expression distribution (top 1000 genes)
    top_genes = features[:, :1000].flatten()
    axes[1, 0].hist(top_genes, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Gene Expression Distribution (Top 1000 Genes)', fontweight='bold')
    axes[1, 0].set_xlabel('Expression Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Class-wise mean comparison
    class_means = []
    for class_name in unique_classes:
        class_mask = class_names == class_name
        class_features = features[class_mask]
        class_means.append(np.mean(class_features))
    
    axes[1, 1].bar(unique_classes, class_means, color=colors[:len(unique_classes)], alpha=0.7)
    axes[1, 1].set_title('Average Expression by Class', fontweight='bold')
    axes[1, 1].set_xlabel('Leukemia Type')
    axes[1, 1].set_ylabel('Mean Expression')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Generated Data - Statistical Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{filename_prefix}_statistics_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. PCA Analysis
    print("   - Computing PCA...")
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for i, class_name in enumerate(unique_classes):
        class_mask = class_names == class_name
        ax.scatter(pca_features[class_mask, 0], 
                  pca_features[class_mask, 1], 
                  label=class_name, 
                  alpha=0.7, 
                  s=50)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
    ax.set_title('PCA Analysis of Generated Data', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{filename_prefix}_pca_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. t-SNE Analysis (if not too many samples)
    if features.shape[0] <= 1000:  # Only run t-SNE for reasonable sample sizes
        print("   - Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]//4))
        tsne_features = tsne.fit_transform(features)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for i, class_name in enumerate(unique_classes):
            class_mask = class_names == class_name
            ax.scatter(tsne_features[class_mask, 0], 
                      tsne_features[class_mask, 1], 
                      label=class_name, 
                      alpha=0.7, 
                      s=50)
        
        ax.set_xlabel('t-SNE Component 1', fontweight='bold')
        ax.set_ylabel('t-SNE Component 2', fontweight='bold')
        ax.set_title('t-SNE Analysis of Generated Data', fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{filename_prefix}_tsne_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Correlation Heatmap (top 50 genes)
    print("   - Creating correlation heatmap...")
    top_50_features = features[:, :50]
    correlation_matrix = np.corrcoef(top_50_features.T)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                cmap='coolwarm', 
                center=0, 
                square=True,
                ax=ax,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    ax.set_title('Gene Expression Correlation Matrix (Top 50 Genes)', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Gene Index', fontweight='bold')
    ax.set_ylabel('Gene Index', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{filename_prefix}_correlation_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Distribution plots saved in: {plots_dir}/")
    return plots_dir

def main():
    """Main generation function"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic leukemia gene expression data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 AML samples
  python generate_data.py --type AML --samples 100
  
  # Generate balanced dataset (50 samples per type)
  python generate_data.py --balanced --samples-per-class 50
  
  # Generate custom samples with specific output format
  python generate_data.py --type PB --samples 200 --format csv --output my_pb_data
  
Available leukemia types:
  AML, Bone_Marrow, Bone_Marrow_CD34, PB, PBSC_CD34
        """
    )
    
    # Generation mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--type', type=str, 
                           choices=['AML', 'Bone_Marrow', 'Bone_Marrow_CD34', 'PB', 'PBSC_CD34'],
                           help='Type of leukemia to generate')
    mode_group.add_argument('--balanced', action='store_true',
                           help='Generate balanced dataset with all types')
    
    # Sample quantity
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples to generate (for single type)')
    parser.add_argument('--samples-per-class', type=int, default=50,
                       help='Number of samples per class (for balanced dataset)')
    
    # Output options
    parser.add_argument('--format', type=str, choices=['csv', 'pkl', 'both'], default='both',
                       help='Output format (default: both)')
    parser.add_argument('--output', type=str, default='generated_data',
                       help='Custom output filename prefix (default: generated_data)')
    
    # Generation options
    parser.add_argument('--output-dir', type=str, default='datasets',
                       help='Output directory (default: datasets)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating distribution plots')
    
    args = parser.parse_args()
    
    # Check if required files exist
    required_files = [
        'models/leukemia_cvae_model.pth',
        'datasets/scaler.pkl'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Required files not found:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 Please run train_model.py first to create the model")
        return 1
    
    try:
        # Initialize generator
        print("🔄 Initializing cVAE generator...")
        generator = LeukemiaCVAEGenerator()
        
        # Create output directory if needed
        Path(args.output_dir).mkdir(exist_ok=True)
        
        # Generate data based on mode
        if args.balanced:
            # Balanced dataset generation
            print(f"🧬 Generating balanced dataset ({args.samples_per_class} samples per type)...")
            
            output_prefix = args.output
            
            data = generator.quick_generate_balanced_and_save(
                samples_per_class=args.samples_per_class,
                output_prefix=output_prefix,
                save_format=args.format
            )
            
            print(f"\n✅ Generated balanced dataset:")
            print(f"   - Total samples: {data['total_samples']}")
            print(f"   - Per class: {data['samples_per_class']}")
            print(f"   - Classes: {len(generator.class_names)}")
            
        else:
            # Single type generation
            print(f"🧬 Generating {args.samples} {args.type} samples...")
            
            output_prefix = f"{args.output}_{args.type.lower()}"
            
            data = generator.quick_generate_and_save(
                leukemia_type=args.type,
                n_samples=args.samples,
                output_prefix=output_prefix,
                save_format=args.format
            )
            
            print(f"\n✅ Generated {args.type} samples:")
            print(f"   - Samples: {data['n_samples']}")
            print(f"   - Type: {data['leukemia_type']}")
        
        # Generate distribution plots if requested
        if not args.no_plots:
            try:
                plots_dir = create_distribution_plots(data, args.output_dir, args.output)
                print(f"\n📊 Distribution analysis plots created in: {plots_dir}")
            except Exception as e:
                print(f"\n⚠️  Warning: Could not create plots - {e}")
                print("Data generation was successful, but plot creation failed.")
        
        print(f"\n📁 Files saved in: {args.output_dir}/")
        print(f"🎉 Generation completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check if models/leukemia_cvae_model.pth exists")
        print("   2. Check if datasets/scaler.pkl exists")
        print("   3. Ensure virtual environment is activated")
        print("   4. Try running train_model.py if model files are missing")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 