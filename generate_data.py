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

# Import our model usage guide
from model_usage_guide import LeukemiaCVAEGenerator

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
    parser.add_argument('--output', type=str, default=None,
                       help='Custom output filename prefix')
    
    # Generation options
    parser.add_argument('--output-dir', type=str, default='datasets',
                       help='Output directory (default: datasets)')
    
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
            
            output_prefix = args.output or 'balanced_leukemia_dataset'
            
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
            
            output_prefix = args.output or f'{args.type.lower()}_samples'
            
            data = generator.quick_generate_and_save(
                leukemia_type=args.type,
                n_samples=args.samples,
                output_prefix=output_prefix,
                save_format=args.format
            )
            
            print(f"\n✅ Generated {args.type} samples:")
            print(f"   - Samples: {data['n_samples']}")
            print(f"   - Type: {data['leukemia_type']}")
        
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