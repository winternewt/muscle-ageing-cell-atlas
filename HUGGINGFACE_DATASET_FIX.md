# 🔧 HuggingFace Dataset Loading Fix

## The Problem

Your dataset currently loads a validation report instead of the actual data when users call `load_dataset("longevity-db/skeletal-muscle-atlas")`. This happens because HuggingFace doesn't know how to load your parquet files as a structured dataset.

## The Solution

Upload the **dataset loading script** (`skeletal_muscle_atlas.py`) to your HuggingFace repository. This script tells HuggingFace how to properly load your parquet files.

## Quick Fix Steps

1. **Upload the loading script** using your existing upload script:
   ```bash
   python3 scripts/04_upload_to_huggingface.py longevity-db/skeletal-muscle-atlas
   ```
   
   The script now includes `skeletal_muscle_atlas.py` which will fix the loading issue.

2. **Test the fix**:
   ```bash
   python3 test_dataset_config.py
   ```

## What Users Will Be Able to Do After the Fix

### Load Different Data Types
```python
from datasets import load_dataset

# Load sample metadata (age, cell types, etc.)
metadata = load_dataset("longevity-db/skeletal-muscle-atlas", name="sample_metadata")

# Load gene expression matrix
expression = load_dataset("longevity-db/skeletal-muscle-atlas", name="expression")

# Load UMAP embeddings for visualization
umap = load_dataset("longevity-db/skeletal-muscle-atlas", name="projection_umap")

# Load all data types
all_data = load_dataset("longevity-db/skeletal-muscle-atlas", name="all")
```

### Available Configurations
- `sample_metadata`: Cell-level metadata (age, cell type, sex, etc.)
- `feature_metadata`: Gene-level metadata (symbols, IDs, etc.)
- `expression`: Gene expression matrix (183,161 cells × 29,400 genes)
- `projection_pca`: PCA embeddings (50 components)
- `projection_umap`: UMAP embeddings (2D visualization)
- `projection_tsne`: t-SNE embeddings (2D visualization)
- `projection_scvi`: scVI embeddings (30D latent space)
- `all`: All data combined

## Why This Was Needed

HuggingFace datasets need either:
1. A specific directory structure that HF auto-recognizes, OR
2. A custom loading script (like `skeletal_muscle_atlas.py`)

Since you have multiple related parquet files (expression, metadata, embeddings), a custom loading script is the best approach to provide a clean, structured interface for users.

## Files Created

- ✅ `skeletal_muscle_atlas.py` - Main dataset loading script
- ✅ `test_dataset_config.py` - Test script to validate the fix
- ✅ Updated `scripts/04_upload_to_huggingface.py` - Now includes the loading script

## Testing

After uploading, users should be able to:
```python
# This should work properly now (instead of loading validation report)
dataset = load_dataset("longevity-db/skeletal-muscle-atlas", name="sample_metadata")
print(dataset)
# Should show: Dataset with cell metadata, not validation report
``` 