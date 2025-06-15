# 🧬 Human Skeletal Muscle Aging Atlas - Processing Code Repository

> **Processing pipeline to convert the Human Skeletal Muscle Aging Atlas into HuggingFace-compatible dataset format**

This repository contains the complete processing pipeline to transform the original H5AD files from the [Human Skeletal Muscle Aging Atlas study](https://www.nature.com/articles/s43587-024-00613-3) into optimized, analysis-ready parquet files suitable for machine learning and longevity research.

## 📊 Dataset Overview

**Original Study**: [Kedlian et al., Nature Aging 2024](https://www.nature.com/articles/s43587-024-00613-3)  
**Interactive Atlas**: [muscleageingcellatlas.org](https://www.muscleageingcellatlas.org/)  
**Source Code**: [github.com/Teichlab/SKM_ageing_atlas](https://github.com/Teichlab/SKM_ageing_atlas)

- **Input**: `SKM_human_pp_cells2nuclei_2023-06-22.h5ad` (1.9GB)
- **Output**: 8 optimized parquet files (1.6GB total)
- **Transformation**: H5AD → HuggingFace-compatible format
- **Data**: 183,161 cells × 29,400 genes from 17 human donors (age 15-75)

---

## 🚀 Quick Start

### Environment Setup
```bash
# Create and activate virtual environment
uv sync
source .venv/bin/activate

# Verify installation
python3 -c "import scanpy, pandas, numpy, pyarrow; print('✅ All dependencies ready!')"
```

### Download Original Data
```bash
# Download from the original source (requires ~2GB space)
mkdir -p data/
# Place the original H5AD file: SKM_human_pp_cells2nuclei_2023-06-22.h5ad
```

### Run Complete Pipeline
```bash
# Run all processing phases
python3 scripts/01_data_exploration.py    # Phase 1: Exploration
python3 scripts/02_data_processing.py     # Phase 2: Processing  
python3 scripts/validate_processing.py    # Phase 3: Validation
```

---

## 📁 Repository Structure

```
muscleageingcellatlas/
├── data/
│   └── SKM_human_pp_cells2nuclei_2023-06-22.h5ad    # Original H5AD file (1.9GB)
├── processed/                                        # Generated parquet files
│   ├── skeletal_muscle_10x_expression.parquet       # Expression matrix (1.5GB)
│   ├── skeletal_muscle_10x_sample_metadata.parquet  # Cell metadata (4.3MB)
│   ├── skeletal_muscle_10x_feature_metadata.parquet # Gene metadata (1.0MB)
│   ├── skeletal_muscle_10x_projection_X_*.parquet   # Dimensionality reductions
│   ├── skeletal_muscle_10x_unstructured_metadata.json # Processing parameters
│   └── validation_visualizations/                   # QC plots
├── scripts/
│   ├── 01_data_exploration.py                       # Phase 1: Dataset exploration
│   ├── 02_data_processing.py                        # Phase 2: H5AD → Parquet conversion
│   └── validate_processing.py                       # Phase 3: Comprehensive validation
├── README.md                                         # Dataset documentation (HuggingFace)
├── CODE_README.md                                    # This processing documentation
├── ROADMAP.md                                        # Project roadmap and status
└── pyproject.toml                                    # Dependencies and configuration
```

---

## 🔄 Processing Pipeline

### **Phase 1: Data Exploration** 📊
**Script**: `scripts/01_data_exploration.py`  
**Duration**: ~2 minutes  
**Purpose**: Comprehensive dataset structure analysis

```python
# Key functions:
explore_dataset()           # Load and inspect H5AD structure
determine_naming_convention()  # Set file naming (10X method confirmed)
assess_biological_relevance() # Validate aging research value
```

**Outputs**:
- `processed/exploration_results.json` - Dataset overview
- `processed/detailed_structure.json` - Complete structural analysis

**Key Discoveries**:
- ✅ 10X Chromium confirmed (mixed v2/v3 chemistry)
- ✅ 8 age groups spanning 15-75 years
- ✅ 36 major cell types + 82 fine-grained subtypes
- ✅ scVI + UMAP embeddings available
- ⚠️ PCA + t-SNE projections missing (computed in Phase 2)

### **Phase 2: Data Processing** 🔧
**Script**: `scripts/02_data_processing.py`  
**Duration**: ~41 minutes  
**Purpose**: Convert H5AD to optimized parquet format

```python
# Key processing functions:
process_expression_matrix()    # 183K×29K sparse→dense conversion
process_sample_metadata()     # Cell annotations + aging metadata
process_feature_metadata()    # Gene information + expression stats
compute_missing_projections()  # Calculate PCA + t-SNE
process_projections()         # Save all 4 embeddings
process_unstructured_metadata() # Processing parameters
fix_pandas_index_column_bug() # Fix known pandas/PyArrow bug
```

**Data Transformations**:
1. **Expression Matrix**: Sparse CSR → Dense Float32 (memory optimized)
2. **Metadata Enhancement**: Added numeric age column for analysis
3. **Missing Projections**: Computed PCA (50D) and t-SNE (2D)
4. **Bug Fixes**: Resolved pandas `__index_level_0__` parquet bug
5. **Index Consistency**: Synchronized all file indices

**Memory Management**:
- Chunked processing for large matrices (>8GB)
- Sparse matrix handling throughout pipeline
- Automatic memory monitoring and logging

### **Phase 3: Validation** ✅
**Script**: `scripts/validate_processing.py`  
**Duration**: ~20 seconds  
**Purpose**: Comprehensive quality assurance

```python
# Validation categories:
validate_file_existence()     # All 8 files created
validate_expression_matrix()  # Shape, data types, value ranges
validate_sample_metadata()    # Required columns, age groups
validate_feature_metadata()   # Gene coverage, annotation quality
validate_projections()        # All 4 embeddings, reasonable ranges
validate_cross_file_consistency() # Index alignment across files
generate_validation_plots()   # UMAP + t-SNE visualizations
```

**Quality Gates** (40 tests):
- ✅ File structure validation (8/8 files)
- ✅ Dimensional consistency (183,161 cells, 29,400 genes)
- ✅ Data type optimization (float32, categorical encoding)
- ✅ Index synchronization across all files
- ✅ Biological metadata completeness
- ✅ Visualization generation

---

## 🛠️ Technical Implementation

### **Dependencies**
Core libraries with version specifications in `pyproject.toml`:
```toml
[tool.uv.dependencies]
python = ">=3.9"
scanpy = ">=1.9.0"           # Single-cell analysis toolkit
pandas = ">=2.0.0"           # Data manipulation
pyarrow = ">=10.0.0"         # Parquet file handling
numpy = ">=1.21.0"           # Numerical computing
scipy = ">=1.7.0"            # Sparse matrix operations
matplotlib = ">=3.5.0"       # Visualization
seaborn = ">=0.11.0"         # Statistical plotting
```

### **Memory Requirements**
- **Minimum RAM**: 16GB (for basic processing)
- **Recommended RAM**: 32GB (for comfortable processing)
- **Peak Usage**: ~20GB during expression matrix conversion
- **Storage**: 4GB free space (input + output + temp files)

### **Performance Optimization**
1. **Sparse Matrix Handling**: Maintains sparsity until final conversion
2. **Chunked Processing**: Automatic chunking for large matrices
3. **Data Type Optimization**: Float32 instead of Float64 saves 50% memory
4. **Compression**: Snappy compression for fast I/O
5. **Memory Monitoring**: Real-time memory usage tracking

### **Bug Fixes & Patches**
The pipeline includes fixes for known issues:

#### **Pandas Index Column Bug** 🐛
**Issue**: Pandas saves DataFrame index as extra `__index_level_0__` column  
**References**: [pandas#51664](https://github.com/pandas-dev/pandas/issues/51664), [polars#7291](https://github.com/pola-rs/polars/issues/7291)  
**Solution**: PyArrow-based post-processing to remove phantom column

```python
def fix_pandas_index_column_bug(parquet_file: Path) -> bool:
    """Fix known pandas/PyArrow parquet bug"""
    table = pq.read_table(parquet_file)
    if "__index_level_0__" in table.column_names:
        clean_table = table.drop(["__index_level_0__"])
        pq.write_table(clean_table, parquet_file, compression='snappy')
    return True
```

---

## 📊 Processing Results

### **Successfully Generated Files**
```
processed/
├── skeletal_muscle_10x_expression.parquet          ✅ 1.5GB (183K×29K cells×genes)
├── skeletal_muscle_10x_sample_metadata.parquet     ✅ 4.3MB (183K×16 cell annotations)
├── skeletal_muscle_10x_feature_metadata.parquet    ✅ 1.0MB (29K×4 gene annotations)
├── skeletal_muscle_10x_projection_X_scVI.parquet   ✅ 35MB (183K×30 scVI embedding)
├── skeletal_muscle_10x_projection_X_umap.parquet   ✅ 4.1MB (183K×2 UMAP coordinates)
├── skeletal_muscle_10x_projection_X_pca.parquet    ✅ 56MB (183K×50 PCA components)
├── skeletal_muscle_10x_projection_X_tsne.parquet   ✅ 4.1MB (183K×2 t-SNE coordinates)
└── skeletal_muscle_10x_unstructured_metadata.json  ✅ 3.2KB (processing parameters)
```

### **Data Quality Metrics**
- **Processing Success Rate**: 100% (40/40 validation tests passed)
- **Data Integrity**: All files have consistent indices (183,161 cells)
- **Memory Efficiency**: 95.4% sparse → 1.6GB compressed output
- **Biological Validity**: Age groups, cell types, donor IDs all validated

### **Performance Benchmarks**
- **Total Processing Time**: 41 minutes (vs. estimated 3-4 hours)
- **Validation Time**: 20 seconds (comprehensive 40-test suite)
- **Memory Peak**: 20GB (well within 32GB recommendation)
- **I/O Efficiency**: Snappy compression provides optimal speed/size balance

---

## 🔬 Usage Examples

### **Loading Processed Data**
```python
import pandas as pd

# Load core files
expression = pd.read_parquet("processed/skeletal_muscle_10x_expression.parquet")
metadata = pd.read_parquet("processed/skeletal_muscle_10x_sample_metadata.parquet")
features = pd.read_parquet("processed/skeletal_muscle_10x_feature_metadata.parquet")

print(f"Dataset: {expression.shape[0]:,} cells × {expression.shape[1]:,} genes")
print(f"Age range: {metadata['age_numeric'].min()}-{metadata['age_numeric'].max()} years")
print(f"Cell types: {metadata['annotation_level0'].nunique()} major types")
```

### **Age-Related Analysis**
```python
# Compare young vs old muscle
young_cells = metadata[metadata['age_numeric'] <= 30].index
old_cells = metadata[metadata['age_numeric'] >= 60].index

# Gene expression comparison
young_expr = expression.loc[young_cells].mean()
old_expr = expression.loc[old_cells].mean()
age_changes = (old_expr / young_expr).sort_values(ascending=False)

print("Top age-upregulated genes:", age_changes.head(10))
```

### **Visualization**
```python
import matplotlib.pyplot as plt

# Load UMAP coordinates
umap = pd.read_parquet("processed/skeletal_muscle_10x_projection_X_umap.parquet")

# Age-colored UMAP
plt.figure(figsize=(10, 8))
scatter = plt.scatter(umap.iloc[:, 0], umap.iloc[:, 1], 
                     c=metadata['age_numeric'], cmap='viridis', s=0.5)
plt.colorbar(scatter, label='Age (years)')
plt.title('Human Muscle Aging Atlas - UMAP')
plt.show()
```

---

## 🔧 Troubleshooting

### **Common Issues**

#### **Memory Errors**
```bash
# If you encounter OOM errors, reduce chunk size
# Edit scripts/02_data_processing.py line 156:
chunk_size = 5000  # Reduce from default 10000
```

#### **Missing Dependencies**
```bash
# Reinstall environment from scratch
rm -rf .venv/
uv sync
source .venv/bin/activate
```

#### **Data Download Issues**
The original H5AD file is large (1.9GB). Ensure:
- Stable internet connection for download
- At least 4GB free disk space
- Download verification (file should be ~1.9GB)

### **Performance Tuning**

#### **For Limited Memory Systems (<16GB)**
```python
# Modify processing parameters in scripts/02_data_processing.py
CHUNK_SIZE = 5000      # Reduce chunk size
MAX_MEMORY_GB = 4.0    # Lower memory threshold
```

#### **For High-Performance Systems (>32GB)**
```python
# Increase performance in scripts/02_data_processing.py  
CHUNK_SIZE = 20000     # Larger chunks
PARALLEL_JOBS = 4      # Enable parallelization
```

---

## 📈 Roadmap & Development

### **Completed Phases** ✅
- **Phase 1**: Data exploration and method identification
- **Phase 2**: Complete H5AD → Parquet conversion pipeline
- **Phase 3**: Comprehensive validation and quality assurance

### **Future Enhancements** 🚧

**Advanced Single-Cell Analysis Integration**:

#### **Cell Type Annotation Validation**
- **CellTypist Integration**: Implement [CellTypist](https://github.com/Teichlab/celltypist) cross-validation pipeline
- **Automated QC**: Compare manual annotations with automated cell type predictions
- **Annotation Refinement**: Enhance existing cell type labels using pre-trained models

```python
# Planned CellTypist integration
import celltypist

# Cross-validate existing annotations
model = celltypist.models.Model.load('Human_Muscle_Atlas')
predictions = celltypist.annotate(adata, model='Human_Muscle_Atlas')
annotation_concordance = compare_annotations(adata.obs['annotation_level0'], 
                                           predictions.predicted_labels)
```

#### **Gene Network Inference**
- **scPRINT Integration**: Leverage [scPRINT foundation model](https://github.com/cantinilab/scPRINT) for regulatory network inference
- **Age-Comparative Networks**: Infer gene regulatory networks for young vs. old muscle
- **Cell-Type-Specific Networks**: Generate regulatory maps for each muscle cell population

```python
# Planned scPRINT integration  
from scprint import scPrint, GNInfer

# Load pre-trained foundation model
model = scPrint.load_from_checkpoint('path/to/scprint/weights')

# Gene network inference for aging analysis
gn_inferrer = GNInfer(model)
young_networks = gn_inferrer(adata[young_cells])
old_networks = gn_inferrer(adata[old_cells])
aging_network_changes = compare_networks(young_networks, old_networks)
```

**Technical Infrastructure**:
- **GPU Acceleration**: RAPIDS integration for large-scale processing
- **Parallel Processing**: Multi-core utilization for chunked operations
- **Cloud Support**: AWS/GCP batch processing capabilities
- **Interactive Notebooks**: Jupyter examples for common analyses

*Note: Advanced ML integrations are planned future developments beyond current hackathon scope.*

### **Contributing**
We welcome contributions! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b new-feature`)
3. Test with validation pipeline
4. Submit pull request with clear description

---

## 📖 Citation & Attribution

### **Processing Pipeline Citation**
```bibtex
@software{muscle_aging_processing_2024,
  title={Human Skeletal Muscle Aging Atlas Processing Pipeline},
  author={Longevity Genomics Consortium},
  year={2024},
  url={[Repository URL]},
  note={Processing pipeline for Kedlian et al. Nature Aging 2024}
}
```

### **Original Dataset Citation**
```bibtex
@article{kedlian2024human,
  title={Human skeletal muscle aging atlas},
  author={Kedlian, Veronika R and Wang, Yaning and Liu, Tianliang and others},
  journal={Nature Aging},
  volume={4},
  pages={727--744},
  year={2024},
  doi={10.1038/s43587-024-00613-3}
}
```

---

## 🤝 Support & Community

### **Getting Help**
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join community discussions for usage questions
- **Documentation**: Complete roadmap available in `ROADMAP.md`

### **Acknowledgments**
- **Original Authors**: Kedlian, Wang, Liu et al. for the groundbreaking study
- **Teichmann Lab**: For open data sharing and excellent documentation
- **Sanger Institute**: For hosting the interactive atlas
- **HuggingFace**: For providing the dataset hosting platform

---

**🔬 Ready for longevity research • 🧬 Reproducible processing • 💻 Production-optimized • 📊 Fully validated** 