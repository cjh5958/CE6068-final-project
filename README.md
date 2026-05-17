# Leukemia Gene Expression Data Augmentation with Conditional VAE

This project explores how a Conditional Variational Autoencoder (cVAE) can be used to augment a small, high-dimensional leukemia gene expression dataset. The original dataset contains only 64 samples, but each sample has 22,283 gene expression features and belongs to one of five biological classes.

The main goal is to learn a class-conditioned latent representation and generate synthetic samples that preserve class-specific gene expression patterns. The generated data can then be used for downstream classifier experiments or exploratory analysis.

## Project Highlights

- Trains a PyTorch Conditional VAE on leukemia gene expression profiles.
- Conditions generation on five sample types:
  - `AML`
  - `Bone_Marrow`
  - `Bone_Marrow_CD34`
  - `PB`
  - `PBSC_CD34`
- Uses standardized preprocessing and label encoding.
- Supports balanced synthetic data generation.
- Provides dataset visualization tools using PCA, t-SNE, UMAP, correlation analysis, and summary statistics.
- Includes downstream classifier experiments with Logistic Regression, KNN, Decision Tree, MLP, and Random Forest.

## Dataset

The raw dataset is expected at:

```text
datasets/raw_data.csv
```

Expected CSV format:

```text
samples,type,<gene_1>,<gene_2>,...,<gene_n>
```

Current raw dataset summary:

| Property | Value |
| --- | ---: |
| Samples | 64 |
| Gene features | 22,283 |
| Total columns | 22,285 |
| Classes | 5 |

Class distribution:

| Class | Samples |
| --- | ---: |
| AML | 26 |
| Bone_Marrow | 10 |
| Bone_Marrow_CD34 | 8 |
| PB | 10 |
| PBSC_CD34 | 10 |

The repository may also contain generated datasets such as:

```text
datasets/generated_data.csv
```

In the current generated dataset, there are 250 synthetic samples: 50 samples for each of the five classes.

## Method Overview

The workflow is:

1. Load leukemia gene expression data.
2. Encode the class label with `LabelEncoder`.
3. Standardize gene expression features with `StandardScaler`.
4. Train a Conditional VAE using the class label as a one-hot condition.
5. Sample or perturb latent vectors for each class.
6. Decode the latent vectors into synthetic gene expression profiles.
7. Save generated data as CSV and/or PKL.
8. Evaluate generated data through visualization and downstream classifiers.

The cVAE objective combines:

- Reconstruction loss: mean squared error between original and reconstructed gene expression vectors.
- KL divergence: regularizes the latent distribution.
- Optional classification loss: encourages the latent representation to preserve class information.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- train_model.py              # End-to-end cVAE training pipeline
|-- cvae_model.py               # Encoder, decoder, and ConditionalVAE modules
|-- cvae_trainer.py             # Training loop, beta scheduling, checkpointing
|-- data_preprocessing.py       # Data loading, splitting, scaling, label encoding
|-- generate_data.py            # CLI for synthetic data generation
|-- model_usage_guide.py        # Python API for loading the model and generating samples
|-- analyze_dataset.py          # PCA, t-SNE, UMAP, statistics, and correlation analysis
|-- models/
|   |-- base_model.py           # Downstream classifier experiment with augmented data
|   |-- raw_data_model.py       # Downstream classifier experiment with raw data only
|-- datasets/
|   |-- raw_data.csv            # Raw leukemia gene expression dataset
|   |-- generated_data.csv      # Example generated dataset, if present locally
```

Model checkpoints and preprocessing artifacts are generated during training and are intentionally suitable for local storage:

```text
models/leukemia_cvae_model.pth
models/best_cvae_model.pth
models/final_cvae_model.pth
datasets/scaler.pkl
datasets/label_encoder.pkl
datasets/metadata.pkl
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Train the cVAE with the default configuration:

```bash
python train_model.py
```

Useful options:

```bash
python train_model.py --epochs 100 --batch-size 8 --learning-rate 1e-4 --latent-dim 32
```

Use GPU if available:

```bash
python train_model.py --device cuda
```

The training pipeline saves the model and preprocessing artifacts needed for generation.

## Generate Synthetic Data

Generate synthetic samples for one class:

```bash
python generate_data.py --type AML --samples 100
```

Generate a balanced dataset:

```bash
python generate_data.py --balanced --samples-per-class 50
```

Save only CSV output and skip plots:

```bash
python generate_data.py --balanced --samples-per-class 50 --format csv --no-plots
```

Available classes:

```text
AML, Bone_Marrow, Bone_Marrow_CD34, PB, PBSC_CD34
```

## Analyze a Dataset

Run the visualization and statistics pipeline:

```bash
python analyze_dataset.py datasets/generated_data.csv
```

For faster analysis on larger files:

```bash
python analyze_dataset.py datasets/generated_data.csv --skip-tsne --skip-correlation
```

Analysis outputs are saved under:

```text
datasets/plots/
```

Depending on the options, the analyzer can produce:

- PCA plots
- t-SNE plots
- UMAP plots
- Correlation heatmaps
- Statistical summaries
- Dimensionality-reduction comparisons

## Downstream Classifier Experiments

Train classifiers on the raw reduced data:

```bash
python models/raw_data_model.py
```

Train classifiers using raw data plus generated data:

```bash
python models/base_model.py
```

The experiment scripts evaluate:

- Logistic Regression
- KNN
- Decision Tree
- MLP
- Random Forest

Result logs are written to files such as:

```text
rawdata_model_results_42.txt
model_results_42.txt
conbine_data_model_results.txt
```

## Current Experimental Results

The repository includes several saved result files. The most reliable way to interpret them is as exploratory downstream validation rather than a final benchmark, because the dataset is very small and the train/test split contains only 32 test samples in the recorded experiments.

Example results from `model_results_42.txt`, using raw data combined with generated data:

| Classifier | Accuracy |
| --- | ---: |
| Logistic Regression | 0.9688 |
| KNN | 1.0000 |
| Decision Tree | 0.8438 |
| MLP | 1.0000 |
| Random Forest | 1.0000 |

Example results from `rawdata_model_results_42.txt`, using raw data only:

| Classifier | Accuracy |
| --- | ---: |
| Logistic Regression | 1.0000 |
| KNN | 1.0000 |
| Decision Tree | 0.7500 |
| MLP | 1.0000 |
| Random Forest | 1.0000 |

These results show that the classes are highly separable in this dataset. The cVAE component is most useful as a generative modeling and augmentation study, not simply as a way to inflate classifier accuracy.

## Python API Example

```python
from model_usage_guide import LeukemiaCVAEGenerator

generator = LeukemiaCVAEGenerator()

aml_data = generator.quick_generate_and_save(
    leukemia_type="AML",
    n_samples=100,
    save_format="csv"
)

balanced_data = generator.quick_generate_balanced_and_save(
    samples_per_class=50,
    save_format="both"
)
```

## Notes and Limitations

- The raw dataset is small: only 64 samples.
- Gene expression data is high-dimensional, so overfitting is a serious risk.
- Some current scripts assume generated artifacts already exist, such as `models/leukemia_cvae_model.pth`, `datasets/scaler.pkl`, and `datasets/metadata.pkl`.
- The current downstream classifier results should be treated as sanity checks, not clinical validation.
- The project is for machine learning experimentation and should not be used for medical decision-making.

## Suggested Future Improvements

- Add cross-validation and repeated random seeds for more stable evaluation.
- Compare cVAE augmentation against simpler baselines such as SMOTE or Gaussian noise.
- Add quantitative distribution similarity metrics between raw and generated data.
- Package preprocessing, training, generation, and evaluation into a single reproducible CLI.
- Add unit tests for model shape checks, preprocessing artifacts, and generation outputs.
- Document the biological source and preprocessing history of the raw dataset.

