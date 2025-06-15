---
license: mit
tags:
- longevity
- aging
- skeletal-muscle
- human
- single-cell-rna-seq
- muscle-aging
- sarcopenia
- 10x-genomics
- muscle-stem-cells
- regenerative-medicine
pretty_name: "Human Skeletal Muscle Aging Atlas - 10X Chromium"
size_categories:
- 100K<n<1M
language:
- en
---

# 🧬 Human Skeletal Muscle Aging Atlas - 10X Chromium Dataset

> **A comprehensive single-cell RNA-seq atlas of human skeletal muscle aging across the lifespan (15-75 years)**

Skeletal muscle aging is characterized by progressive loss of muscle mass and strength, often leading to sarcopenia and frailty in older adults. This dataset originates from the **Human Skeletal Muscle Aging Atlas** study published in *Nature Aging* (2024), which provides molecular insights into how aging affects different muscle cell types and their interactions.

**Original Study**: [Kedlian et al., Nature Aging 2024](https://www.nature.com/articles/s43587-024-00613-3)  
**Interactive Atlas**: [muscleageingcellatlas.org](https://www.muscleageingcellatlas.org/)  
**Code Repository**: [github.com/Teichlab/SKM_ageing_atlas](https://github.com/Teichlab/SKM_ageing_atlas)

## 🎯 Dataset Overview

This dataset provides unprecedented single-cell resolution insights into human skeletal muscle aging, featuring **183,161 cells** from **17 donors** aged 15-75 years. The data captures the molecular landscape of muscle aging, including changes in muscle fiber types, stem cell populations, immune infiltration, and tissue remodeling processes critical for understanding sarcopenia and age-related muscle dysfunction.

### Key Statistics
- **Cells**: 183,161 high-quality single cells and nuclei
- **Genes**: 29,400 protein-coding genes  
- **Donors**: 17 individuals (age range: 15-75 years)
- **Technology**: 10X Chromium (v2/v3 chemistry)
- **Tissue**: Human skeletal muscle biopsies
- **Cell Types**: 36 major cell populations across muscle, immune, and stromal compartments

## 🤖 **For Machine Learning**

This dataset is ready for ML workflows with standard scikit-learn, PyTorch, or TensorFlow pipelines:

- **Features (X)**: `expression.parquet` → 183,161 × 29,400 gene expression matrix
- **Labels (y)**: `sample_metadata.parquet` → Age groups, cell types, sex, etc.  
- **Embeddings**: Pre-computed PCA, UMAP, scVI, t-SNE projections

### **Common Tasks**
```python
# Age prediction (regression/classification)
X = expression.values              # Features: gene expression  
y = metadata['age_numeric']        # Target: age in years (15-75)

# Cell type classification (36 classes)
y_celltype = metadata['annotation_level0']  

# Using pre-computed embeddings
embeddings = scvi_projection.values  # 30D latent space
```

**Key considerations**: High dimensionality (29,400 genes), 95.4% sparsity, donor batch effects for cross-validation.

## 🧬 Biological Context & Significance

> Skeletal muscle aging affects everyone who lives long enough. Starting around age 30, humans lose 3-8% of muscle mass per decade, accelerating after age 60. This progressive decline—called sarcopenia—leads to frailty, falls, disability, and loss of independence in older adults. Despite affecting over 50 million people globally, the molecular mechanisms driving muscle aging remain poorly understood. Current interventions are limited to exercise and nutrition, with no approved pharmaceuticals specifically targeting age-related muscle loss.

This dataset addresses a critical gap in aging research by providing **the first comprehensive single-cell atlas of human skeletal muscle across the adult lifespan**. Unlike previous studies that analyzed bulk tissue (averaging across millions of cells), single-cell sequencing reveals how individual cell types change with age—uncovering cellular heterogeneity invisible to traditional methods.

### **Why This Matters for ML & Broader Research**

**🔬 Fundamental Biology**: Muscle aging involves complex interactions between stem cells, immune cells, fibroblasts, and muscle fibers. Single-cell data captures these cellular conversations, revealing which cell types drive aging and which are secondary responders.

**💊 Drug Discovery**: Most anti-aging interventions tested in mice fail in humans, partly due to species differences. This human dataset enables ML models to identify drug targets specific to human muscle aging, potentially accelerating therapeutic development.

**🧬 Precision Medicine**: Age-related muscle loss varies dramatically between individuals. ML models trained on this data could predict who is at highest risk for sarcopenia, enabling personalized prevention strategies.

**📊 Methodological Advances**: The dataset showcases state-of-the-art single-cell integration methods (scVI) that ML researchers can adapt for other domains involving batch effects and high-dimensional biological data.

**🌍 Broader Impact**: Muscle aging research intersects with diabetes (muscle insulin resistance), cancer (muscle wasting), neurodegenerative diseases (muscle denervation), and healthy aging. Advances here could benefit multiple fields.

## 🤖 **Machine Learning Use Cases**

This dataset is structured for multiple ML applications common in Hugging Face workflows:

### **🎯 Classification Tasks**

#### **1. Age Prediction (Regression/Classification)**
- **Target Variable**: `age_numeric` (continuous) or `Age_group` (categorical)
- **Features**: Gene expression matrix (29,400 dimensions)
- **Use Case**: Predict biological age from gene expression patterns
- **Challenge**: Large class imbalance (young vs old groups)

```python
# Example ML setup for age prediction
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prepare features (X) and targets (y)
X = expression.values  # 183,161 × 29,400 gene expression matrix
y_age = metadata['Age_group'].values  # 8 age categories
y_numeric = metadata['age_numeric'].values  # Continuous age

# Train/test split maintaining cell type distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y_age, test_size=0.2, stratify=y_age, random_state=42
)
```

#### **2. Cell Type Classification**
- **Target Variables**: `annotation_level0` (36 classes), `annotation_level1` (43 classes), `annotation_level2` (82 classes)
- **Features**: Gene expression matrix
- **Use Case**: Automated cell type annotation for new datasets
- **Applications**: Transfer learning to other muscle datasets

```python  
# Multi-level cell type classification
cell_types_coarse = metadata['annotation_level0']  # 36 major cell types
cell_types_fine = metadata['annotation_level2']    # 82 fine-grained types

# Hierarchical classification possible
print(f"Coarse classes: {cell_types_coarse.nunique()}")
print(f"Fine classes: {cell_types_fine.nunique()}")
```

#### **3. Sex Prediction**
- **Target Variable**: `Sex` (binary: M/F)  
- **Features**: Gene expression matrix
- **Use Case**: Identify sex-specific aging patterns
- **Applications**: Sex-stratified aging biomarkers

#### **4. Protocol Classification**
- **Target Variable**: `batch` (cells vs nuclei)
- **Use Case**: Batch effect detection and correction
- **Applications**: Domain adaptation between protocols

### **🔍 Clustering & Dimensionality Reduction**

#### **Unsupervised Learning Tasks**
- **Pre-computed embeddings**: scVI (30D), PCA (50D), UMAP (2D), t-SNE (2D)
- **Use Cases**: 
  - Cell state discovery
  - Age-associated transcriptional programs
  - Novel cell type identification
  - Trajectory inference

```python
# Use pre-computed embeddings for clustering
scvi_embedding = pd.read_parquet("skeletal_muscle_10x_projection_X_scVI.parquet")
pca_embedding = pd.read_parquet("skeletal_muscle_10x_projection_X_pca.parquet")

# Ready for sklearn clustering
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=10).fit_predict(scvi_embedding.values)
```

### **🧠 Deep Learning Applications**

#### **Foundation Model Training**
- **Task**: Self-supervised pre-training on single-cell data
- **Architecture**: Transformer-based (scBERT, scGPT, Geneformer)
- **Data Format**: Already tokenized as gene expression vectors

#### **Variational Autoencoders (VAE)**
- **Task**: Dimensionality reduction and generative modeling
- **Benchmark**: Compare against pre-computed scVI embeddings
- **Applications**: Data augmentation, missing value imputation

### **📊 Data Distributions**

```python
# Check class distributions for stratified sampling
print("Age group distribution:")
print(metadata['Age_group'].value_counts().sort_index())

print("\nCell type distribution (top 10):")
print(metadata['annotation_level0'].value_counts().head(10))

print("\nSex distribution:")
print(metadata['Sex'].value_counts())
```

**Distribution Notes:**
- **Age**: Older adults (70-75) represent 32.7% of data
- **Cell Types**: Muscle fibers (MF-I: 23%, MF-II: 15.6%) reflect natural tissue composition
- **Rare cell types**: Some immune and stem cell populations are naturally sparse
- **Protocols**: Both single cells and nuclei included

### **📊 Feature Engineering Opportunities**

#### **Gene-Level Features**
- **Highly Variable Genes**: Pre-selected for dimensionality reduction
- **Pathway Scores**: Aggregate gene sets into pathway-level features
- **Gene Ratios**: Type I/II fiber markers, stem cell signatures

#### **Cell-Level Features**  
- **QC Metrics**: `n_genes`, `n_counts` as auxiliary features
- **Derived Features**: Mitochondrial gene %, ribosomal gene %
- **Embedding Features**: Use scVI/PCA as compressed representations

### **🎯 Evaluation Metrics & Benchmarks**

#### **For Classification Tasks**
- **Age Prediction**: MAE, R² for continuous; F1-score for categorical
- **Cell Type**: Macro/micro F1, confusion matrices
- **Imbalanced Classes**: Use weighted metrics, AUROC for rare cell types

#### **For Clustering**
- **Biological Validation**: Adjusted Rand Index against known cell types
- **Silhouette Score**: Cluster quality in embedding space
- **Downstream Task**: Cell type classification accuracy

### **Research Applications**

This dataset enables research across multiple domains:

#### 🔬 **Understanding Muscle Aging**
- **Fiber type switching**: How muscle transitions from power (Type II) to endurance (Type I) with age
- **Stem cell exhaustion**: Why muscle repair fails in older adults
- **Protein quality control**: Cellular mechanisms behind age-related muscle wasting
- **Mitochondrial decline**: Energy production defects in aging muscle

#### 🩺 **Regenerative Medicine** 
- **Stem cell rejuvenation**: Identifying factors that restore youthful muscle stem cell function
- **Tissue engineering**: Understanding cellular requirements for growing muscle in the lab
- **Injury repair**: How aging affects muscle's ability to heal from damage
- **Therapeutic screening**: Testing compounds that promote muscle regeneration

#### 📊 **Biomarker & Diagnostic Development**
- **Aging clocks**: Molecular signatures that predict biological age from muscle biopsies
- **Disease prediction**: Early detection of sarcopenia, frailty, and disability risk
- **Treatment response**: Personalized markers for exercise and drug interventions
- **Health monitoring**: Non-invasive biomarkers of muscle health

#### 💊 **Drug Discovery & Development**
- **Target identification**: Finding druggable pathways specific to human muscle aging
- **Exercise mimetics**: Drugs that replicate the benefits of physical activity
- **Anti-inflammatory therapies**: Reducing chronic inflammation that drives muscle loss
- **Combination therapies**: Optimizing multi-target approaches for muscle preservation

**The Challenge**: By 2050, over 1.6 billion people will be aged 65+, making muscle aging a global health crisis. Current approaches treat symptoms rather than causes.

**The Opportunity**: This dataset provides the molecular roadmap to understand, predict, and potentially reverse muscle aging at the cellular level. For ML researchers, it represents a chance to apply cutting-edge techniques to one of humanity's most pressing biological challenges.

---

## 📁 Repository Contents

This repository contains **12 files (1.6GB total)** organized into three categories:

### **🗂️ Core Dataset Files (1.6GB)**
*Essential files for machine learning and analysis*

- `skeletal_muscle_10x_expression.parquet` (1.5GB) - **Main feature matrix** (183K cells × 29K genes)
- `skeletal_muscle_10x_sample_metadata.parquet` (4.3MB) - **Cell-level annotations** (age, cell type, sex, etc.)
- `skeletal_muscle_10x_feature_metadata.parquet` (1.0MB) - **Gene-level annotations** (symbols, IDs)
- `skeletal_muscle_10x_projection_X_pca.parquet` (57MB) - **PCA embeddings** (50D)
- `skeletal_muscle_10x_projection_X_tsne.parquet` (4.1MB) - **t-SNE visualization** (2D)
- `skeletal_muscle_10x_projection_X_umap.parquet` (4.1MB) - **UMAP visualization** (2D)
- `skeletal_muscle_10x_projection_X_scVI.parquet` (35MB) - **scVI latent space** (30D) *[Bonus embedding]*

### **📊 Metadata Files**
*Additional processing information for transparency*

- `skeletal_muscle_10x_unstructured_metadata.json` (3.2KB) - **Processing parameters** 
- `detailed_structure.json` (4.3KB) - **Detailed data structure information**
- `validation_report.json` (4.4KB) - **Quality validation results**

### **📚 Documentation Files**
- `README.md` - **This comprehensive dataset documentation**
- `LICENSE` - **MIT License for open research use**

---

## 📋 Detailed File Descriptions

### 🧬 **Expression Matrix** 
**File**: `skeletal_muscle_10x_expression.parquet`  
**Size**: 1.5GB | **Shape**: 183,161 cells × 29,400 genes  
**Format**: Sparse matrix stored efficiently in Parquet format

This is the primary feature matrix for ML tasks. Each row is a single cell (sample), each column is a gene (feature).
- **Data Type**: float32 (memory optimized for ML pipelines)
- **Sparsity**: 95.4% (use `scipy.sparse` for efficient processing)
- **Index**: Cell barcodes (consistent across all files)
- **Columns**: Gene symbols (HGNC approved)
- **ML Use**: Direct input to models after train/test splitting

```python
import pandas as pd

# Load expression data
expression = pd.read_parquet("skeletal_muscle_10x_expression.parquet")
print(f"Expression shape: {expression.shape}")
print(f"Data type: {expression.dtypes[0]}")
print(f"Memory usage: {expression.memory_usage(deep=True).sum() / 1e9:.1f} GB")

# Example: Top expressed genes
top_genes = expression.sum().nlargest(10)
print("Top 10 expressed genes:")
print(top_genes)
```

### 👥 **Sample Metadata**
**File**: `skeletal_muscle_10x_sample_metadata.parquet`  
**Size**: 4.3MB | **Shape**: 183,161 cells × 16 columns

Contains target variables and metadata for each cell. Each row corresponds to a cell in the expression matrix.

| Column | **ML Task Type** | Description | Example Values | **ML Application** |
|--------|------------------|-------------|----------------|--------------------|
| `Age_group` | **Classification** | Age stratification | "15-20", "25-30", ..., "70-75" | **Primary aging predictor** (8 classes) |
| `age_numeric` | **Regression** | Numeric age (midpoint) | 17.5, 27.5, ..., 72.5 | **Continuous age prediction** |
| `Sex` | **Binary Classification** | Biological sex | "M", "F" | **Sex prediction from gene expression** |
| `annotation_level0` | **Multi-class Classification** | Major cell types | "MF-I", "MF-II", "MuSC", "FB" | **Cell type annotation** (36 classes) |
| `annotation_level1` | **Multi-class Classification** | Intermediate cell types | 43 subcategories | **Fine-grained cell typing** |
| `annotation_level2` | **Multi-class Classification** | Fine cell types | 82 fine-grained types | **Ultra-fine cell classification** |
| `DonorID` | **Stratification Variable** | Individual donor | "Donor_01", "Donor_02", ... | **Cross-validation grouping** |
| `batch` | **Domain Adaptation** | Protocol type | "cells", "nuclei" | **Batch effect correction** |

#### Age Distribution
```
Age Group Distribution:
├── 15-20: 23,345 cells (12.7%) [Young]
├── 20-25: 398 cells (0.2%)   [Young]
├── 25-30: 30,802 cells (16.8%) [Young] 
├── 35-40: 29,376 cells (16.0%) [Young]
├── 50-55: 4,093 cells (2.2%)  [Old]
├── 55-60: 5,374 cells (2.9%)  [Old]
├── 60-65: 29,976 cells (16.4%) [Old]
└── 70-75: 59,797 cells (32.7%) [Old]
```

#### Cell Type Composition (Top 10)
```
Major Cell Types:
├── MF-I: 42,052 cells (23.0%)      [Type I Myofibers]
├── MF-II: 28,558 cells (15.6%)     [Type II Myofibers]  
├── FB: 27,925 cells (15.2%)        [Fibroblasts]
├── MuSC: 20,215 cells (11.0%)      [Muscle Stem Cells]
├── SMC: 8,049 cells (4.4%)         [Smooth Muscle Cells]
├── T-cell: 7,385 cells (4.0%)      [T Lymphocytes]
├── MF-IIsc(fg): 7,075 cells (3.9%) [Type II Subcutaneous]
├── MF-Isc(fg): 5,205 cells (2.8%)  [Type I Subcutaneous]
├── Macrophage: 4,723 cells (2.6%)  [Macrophages]
└── NK-cell: 3,725 cells (2.0%)     [Natural Killer Cells]
```

```python
# Load and explore sample metadata
metadata = pd.read_parquet("skeletal_muscle_10x_sample_metadata.parquet")

# Age distribution analysis
age_counts = metadata['Age_group'].value_counts().sort_index()
print("Age distribution:")
print(age_counts)

# Cell type analysis
cell_type_counts = metadata['annotation_level0'].value_counts()
print(f"\nTop 10 cell types:")
print(cell_type_counts.head(10))

# Sex distribution
sex_dist = metadata['Sex'].value_counts()
print(f"\nSex distribution: {sex_dist.to_dict()}")
```

### 🧮 **Feature Metadata**
**File**: `skeletal_muscle_10x_feature_metadata.parquet`  
**Size**: 1.0MB | **Shape**: 29,400 genes × 4 columns

Gene annotation with complete coverage:

| Column | Description | Coverage |
|--------|-------------|----------|
| `SYMBOL` | HGNC gene symbol | 100% |
| `ENSEMBL` | Ensembl gene ID | 100% |
| `gene_ids` | Index identifier | 100% |
| `n_cells` | Expression frequency | 100% |

```python
# Load feature metadata
features = pd.read_parquet("skeletal_muscle_10x_feature_metadata.parquet")

# Gene expression frequency
print(f"Genes expressed in >50% cells: {(features['n_cells'] > 91580).sum()}")
print(f"Genes expressed in >10% cells: {(features['n_cells'] > 18316).sum()}")

# Example muscle-specific genes
muscle_genes = features[features['SYMBOL'].str.contains('MYO|ACTN|TTN', na=False)]
print(f"Muscle-related genes found: {len(muscle_genes)}")
```

### 🗺️ **Dimensionality Reductions**

All projection files share the same structure: 183,161 rows × N dimensions, with cell barcodes as index.

#### **scVI Embedding** (Primary Analysis)
**File**: `skeletal_muscle_10x_projection_X_scVI.parquet`  
**Dimensions**: 30D latent space | **Size**: 35MB

- **Method**: Single-cell Variational Inference (scVI)
- **Batch Correction**: Integrated across 10X v2/v3 chemistry
- **Range**: -2.90 to 3.64 (normalized latent space)
- **Use Case**: Primary embedding for downstream analysis

#### **UMAP Projection** (Visualization)
**File**: `skeletal_muscle_10x_projection_X_umap.parquet`  
**Dimensions**: 2D coordinates | **Size**: 4.1MB

- **Method**: Uniform Manifold Approximation and Projection
- **Range**: X: 7.50-19.20, Y: similar range
- **Use Case**: Primary visualization and cluster identification

#### **PCA Projection** (Linear Reduction)
**File**: `skeletal_muscle_10x_projection_X_pca.parquet`  
**Dimensions**: 50 components | **Size**: 56MB

- **Method**: Principal Component Analysis
- **Range**: -30.80 to 17.86 (standardized)
- **Use Case**: Linear dimensionality reduction, variance analysis

#### **t-SNE Projection** (Non-linear Visualization)  
**File**: `skeletal_muscle_10x_projection_X_tsne.parquet`  
**Dimensions**: 2D coordinates | **Size**: 4.1MB

- **Method**: t-Distributed Stochastic Neighbor Embedding
- **Range**: -51.03 to -4.97 (arbitrary t-SNE units)
- **Use Case**: Alternative visualization, local structure emphasis

```python
# Load all projections for integrated analysis
scvi = pd.read_parquet("skeletal_muscle_10x_projection_X_scVI.parquet")
umap = pd.read_parquet("skeletal_muscle_10x_projection_X_umap.parquet") 
pca = pd.read_parquet("skeletal_muscle_10x_projection_X_pca.parquet")
tsne = pd.read_parquet("skeletal_muscle_10x_projection_X_tsne.parquet")

print(f"scVI shape: {scvi.shape}")
print(f"UMAP shape: {umap.shape}")
print(f"PCA shape: {pca.shape}")  
print(f"t-SNE shape: {tsne.shape}")

# Verify index consistency
assert all(scvi.index == umap.index == pca.index == tsne.index)
print("✅ All projection indices are consistent")
```

### ⚙️ **Unstructured Metadata**
**File**: `skeletal_muscle_10x_unstructured_metadata.json`  
**Size**: 3.2KB | **Format**: JSON

Contains processing parameters and technical metadata:
```json
{
  "pca": {
    "params": {"n_comps": 50, "use_highly_variable": true}
  },
  "neighbors": {
    "params": {"n_neighbors": 15, "n_pcs": 50, "method": "umap"}
  },
  "tsne": {
    "params": {"perplexity": 30, "early_exaggeration": 12}
  }
}
```

---

## 🚀 Quick Start Examples

### Basic Data Loading
```python
import pandas as pd
import numpy as np

# Load core files
expression = pd.read_parquet("skeletal_muscle_10x_expression.parquet")
metadata = pd.read_parquet("skeletal_muscle_10x_sample_metadata.parquet")
umap_coords = pd.read_parquet("skeletal_muscle_10x_projection_X_umap.parquet")

print(f"Dataset loaded: {expression.shape[0]:,} cells × {expression.shape[1]:,} genes")
```

### Age-Related Analysis
```python
# Compare young vs old muscle
young_cells = metadata[metadata['age_numeric'] <= 30].index
old_cells = metadata[metadata['age_numeric'] >= 60].index

print(f"Young cohort: {len(young_cells):,} cells")
print(f"Old cohort: {len(old_cells):,} cells")

# Differential expression analysis
young_expr = expression.loc[young_cells].mean()
old_expr = expression.loc[old_cells].mean()

# Calculate fold changes
log2fc = np.log2((old_expr + 1e-9) / (young_expr + 1e-9))
age_genes = log2fc.abs().nlargest(20)

print("Top 20 age-related genes:")
print(age_genes)
```

### Cell Type Analysis
```python
# Analyze muscle stem cells across age
musc_cells = metadata[metadata['annotation_level0'] == 'MuSC']
musc_age_dist = musc_cells.groupby('Age_group').size()

print("Muscle stem cell distribution by age:")
print(musc_age_dist)

# Cell type proportions by age
cell_props = metadata.groupby(['Age_group', 'annotation_level0']).size().unstack(fill_value=0)
cell_props_pct = cell_props.div(cell_props.sum(axis=1), axis=0) * 100

print("Cell type proportions by age group:")
print(cell_props_pct.round(1))
```

### Visualization Example
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create aging UMAP plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Color by age
scatter_data = pd.concat([umap_coords, metadata[['age_numeric', 'annotation_level0']]], axis=1)

axes[0].scatter(scatter_data.iloc[:, 0], scatter_data.iloc[:, 1], 
               c=scatter_data['age_numeric'], cmap='viridis', s=0.1, alpha=0.6)
axes[0].set_title('Muscle Aging UMAP - Colored by Age')
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')

# Plot 2: Color by cell type (top 10 types only)
top_types = metadata['annotation_level0'].value_counts().head(10).index
mask = scatter_data['annotation_level0'].isin(top_types)
filtered_data = scatter_data[mask]

for i, cell_type in enumerate(top_types):
    subset = filtered_data[filtered_data['annotation_level0'] == cell_type]
    axes[1].scatter(subset.iloc[:, 0], subset.iloc[:, 1], 
                   label=cell_type, s=0.1, alpha=0.6)

axes[1].set_title('Muscle Aging UMAP - Colored by Cell Type')
axes[1].set_xlabel('UMAP 1') 
axes[1].set_ylabel('UMAP 2')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
```

### Advanced Analysis: Muscle Fiber Type Switching
```python
# Analyze Type I vs Type II myofiber ratios across age
myofibers = metadata[metadata['annotation_level0'].str.contains('MF-', na=False)]

# Calculate Type I/II ratios by age group
type_counts = myofibers.groupby(['Age_group', 'annotation_level0']).size().unstack(fill_value=0)

# Focus on main fiber types
if 'MF-I' in type_counts.columns and 'MF-II' in type_counts.columns:
    type_counts['Type_I_ratio'] = type_counts['MF-I'] / (type_counts['MF-I'] + type_counts['MF-II'])
    
    print("Type I fiber ratios by age:")
    print(type_counts['Type_I_ratio'].round(3))
    
    # Statistical test for age-related changes
    young_ratio = type_counts.loc[['15-20', '25-30', '35-40'], 'Type_I_ratio'].mean()
    old_ratio = type_counts.loc[['60-65', '70-75'], 'Type_I_ratio'].mean()
    
    print(f"\nYoung Type I ratio: {young_ratio:.3f}")
    print(f"Old Type I ratio: {old_ratio:.3f}")
    print(f"Age-related change: {((old_ratio - young_ratio) / young_ratio * 100):+.1f}%")
```

---

## 📚 **Biological Terms**

### **🧬 Cell Types & Biology**

| **Term** | **ML Interpretation** | **Biological Meaning** |
|----------|----------------------|----------------------|
| **MuSC (Muscle Stem Cells)** | Rare cell class (~11%) | Adult stem cells that repair muscle damage; **key target for anti-aging interventions** |
| **MF-I (Type I Myofibers)** | Largest class (~23%) | Slow-twitch muscle fibers; oxidative, fatigue-resistant; **associated with endurance** |
| **MF-II (Type II Myofibers)** | Major class (~15.6%) | Fast-twitch muscle fibers; glycolytic, powerful; **associated with strength** |
| **FB (Fibroblasts)** | Common class (~15.2%) | Connective tissue cells; **increase with age → fibrosis/scarring** |
| **Sarcopenia** | Primary outcome variable | Age-related muscle loss; **main prediction target** |
| **scVI** | Dimensionality reduction | Single-cell Variational Inference; **state-of-the-art batch correction** |

### **🎯 Key ML-Relevant Biological Concepts**

#### **Myofiber Type Switching**
- **ML Application**: Binary classification (Type I ↔ Type II)
- **Biological Relevance**: Age-related shift from Type II (power) to Type I (endurance)
- **Features**: Gene expression ratios (MYH1, MYH2, MYH7)

#### **Stem Cell Exhaustion**  
- **ML Application**: Regression on stem cell function markers
- **Biological Relevance**: MuSC lose regenerative capacity with age
- **Features**: Cell cycle genes, differentiation markers

#### **Inflammatory Aging (Inflammaging)**
- **ML Application**: Multi-class classification of immune cell states
- **Biological Relevance**: Chronic low-grade inflammation with aging
- **Features**: Cytokine expression, immune cell proportions

#### **Cellular Senescence**
- **ML Application**: Senescence score prediction
- **Biological Relevance**: Cells stop dividing but remain metabolically active
- **Features**: p16, p21, SASP (senescence-associated secretory phenotype) genes

### **📊 Data Structure Translation**

#### **Expression Matrix** = **Feature Matrix (X)**
```python
# Biological: Gene expression per cell
# ML: Features matrix for prediction tasks
expression.shape  # (183,161 cells, 29,400 genes) = (samples, features)
```

#### **Sample Metadata** = **Labels & Covariates (y, metadata)**
```python
# Biological: Cell annotations and donor characteristics  
# ML: Target variables and confounding factors
metadata['Age_group']           # Primary target for aging prediction
metadata['annotation_level0']   # Cell type labels (36 classes)
metadata['Sex']                 # Biological covariate
metadata['DonorID']             # Random effect/batch variable
```

#### **Dimensionality Reductions** = **Learned Representations**
```python
# Biological: Low-dimensional cell state representations
# ML: Pre-computed embeddings for downstream tasks
scvi_embedding   # 30D latent space (like word2vec for cells)
pca_embedding    # 50D linear projection  
umap_coords      # 2D visualization (like t-SNE)
```

### **⚠️ Important ML Considerations**

#### **Important Considerations**
- **Donor Effects**: 17 individuals - use donor ID for stratified cross-validation
- **Sex Differences**: Major confounder in aging studies
- **Batch Effects**: cells vs nuclei protocols
- **Feature Selection**: ~10,000 highly variable genes pre-selected for dimensionality reduction

### **🔬 Biological Validation**

Models should recover known patterns:
- Age prediction: aging genes (CDKN2A, TP53)
- Cell type classification: established cell markers
- Sex prediction: Y-chromosome genes
- Stem cell analysis: regenerative capacity markers  

---

## 🔬 Technical Specifications

### **Data Processing Pipeline**
1. **Quality Control**: Pre-filtered for high-quality cells and genes
2. **Normalization**: Library size normalization applied
3. **Batch Integration**: scVI model trained across 10X chemistry versions
4. **Dimensionality Reduction**: Multiple methods for comprehensive analysis
5. **Data Format**: Optimized Parquet format for fast loading

### **Quality Metrics** (Median values)
- **Genes per cell**: 1,172 (healthy range: 500-5,000) ✅
- **UMIs per cell**: 2,751 (healthy range: 1,000-50,000) ✅
- **Mitochondrial gene %**: Pre-filtered (< 20%) ✅
- **Ribosomal gene %**: Pre-filtered ✅

### **Memory Requirements**
- **Minimum RAM**: 8GB for basic analysis
- **Recommended RAM**: 16GB for full dataset analysis  
- **Expression matrix**: ~6GB in memory (sparse)
- **All projections**: ~2GB additional

### **Compatibility**
- **Python**: pandas, scanpy, anndata, scikit-learn
- **R**: Seurat, SingleCellExperiment, reticulate
- **File Format**: Parquet (cross-platform compatible)

---

## 📚 Dataset Provenance & Methods

This dataset originates from the **Human Skeletal Muscle Aging Atlas** study, downloaded and processed from the original sources listed below. The study characterizes the diversity of cell types in human skeletal muscle across age using complementary single-cell and single-nucleus sequencing technologies.

### **Original Data Sources**
- **Interactive Atlas**: https://www.muscleageingcellatlas.org/
- **Raw Data Repository**: https://github.com/Teichlab/SKM_ageing_atlas
- **Processed Data**: E-MTAB-13874 (ArrayExpress)

### **Sample Collection**
- **Tissue**: Human intercostal muscle biopsies
- **Method**: Percutaneous needle biopsy
- **Subjects**: 17 healthy individuals (age 15-75, 8 young + 9 old)
- **Ethics**: IRB approved, informed consent obtained

### **Sequencing Protocol**
- **Platform**: 10X Chromium Single Cell 3' v2 and v3
- **Chemistry**: Mixed v2/v3 with batch correction applied
- **Sequencing**: Illumina NovaSeq 6000
- **Target Depth**: ~50,000 reads per cell

### **Computational Processing**
- **Alignment**: Cell Ranger (10X Genomics)
- **Reference**: GRCh38-2020-A human genome
- **Integration**: scVI (Single-cell Variational Inference)
- **Quality Control**: Standard scanpy pipeline
- **Batch Correction**: Integrated across chemistry versions

---

## 📖 Citation & Attribution

### **Original Study**
If you use this dataset, please cite the original research:

```bibtex
@article{kedlian2024human,
  title={Human skeletal muscle aging atlas},
  author={Kedlian, Veronika R and Wang, Yaning and Liu, Tianliang and Chen, Xiaoping and Bolt, Liam and Tudor, Catherine and Shen, Zhuojian and Fasouli, Eirini S and Prigmore, Elena and Kleshchevnikov, Vitalii and others},
  journal={Nature Aging},
  volume={4},
  pages={727--744},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s43587-024-00613-3},
  url={https://www.nature.com/articles/s43587-024-00613-3}
}
```

### **Dataset Citation**
```bibtex
@dataset{skeletal_muscle_10x_longevity_2024,
  title={Human Skeletal Muscle Aging Atlas - 10X Chromium Dataset},
  author={Longevity Genomics Consortium},
  year={2024},
  publisher={HuggingFace Hub},
  url={[URL to be updated]}
}
```

### **Future Enhancements** 🚧

This dataset provides a foundation for advanced single-cell analyses using state-of-the-art tools:

#### **Cell Type Annotation Validation**
- **CellTypist Integration**: Cross-validate existing annotations using [CellTypist](https://github.com/Teichlab/celltypist) models
- **Automated Annotation**: Apply pre-trained models for consistent cell type classification
- **Quality Assessment**: Compare manual annotations with automated predictions for data quality control

#### **Gene Network Inference** 
- **scPRINT Integration**: Leverage [scPRINT foundation model](https://github.com/cantinilab/scPRINT) for gene regulatory network inference
- **Age-Specific Networks**: Compare gene regulatory networks between young and old muscle
- **Cell Type Networks**: Infer cell-type-specific regulatory programs in muscle aging

*Note: These enhancements are planned future developments beyond the current dataset release.*

### **Acknowledgments**
- Original data generators and study participants
- 10X Genomics for single-cell technology
- Longevity research community for data standards
- HuggingFace for hosting infrastructure

---

## 🤝 Usage Guidelines & Ethics

### **Data Usage Terms**
- **License**: MIT (permissive use for research and commercial applications)
- **Attribution**: Please cite original study and dataset
- **Responsible Use**: Follow ethical guidelines for human subject data
- **No Re-identification**: Do not attempt to identify individual donors

### **Research Applications**
This dataset is optimized for:
- ✅ Aging mechanism discovery
- ✅ Biomarker identification  
- ✅ Therapeutic target discovery
- ✅ Method development and benchmarking
- ✅ Educational and training purposes

### **Prohibited Uses**
- ❌ Individual identification attempts
- ❌ Discriminatory applications
- ❌ Commercial use without proper attribution
- ❌ Redistribution without proper licensing

---

## 🔧 Troubleshooting & Support

### **Common Issues**

#### Memory Errors
```python
# Load data in chunks for large analyses
chunk_size = 10000
for i in range(0, len(expression), chunk_size):
    chunk = expression.iloc[i:i+chunk_size]
    # Process chunk
```

#### Missing Projections
All projection files are pre-computed and included. If you need custom projections:
```python
import scanpy as sc

# Load into AnnData for scanpy compatibility
adata = sc.AnnData(X=expression.values)
adata.obs = metadata
adata.var_names = expression.columns

# Compute custom projections
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
```

#### Pandas Index Column Bug
If you encounter unexpected columns like `__index_level_0__` when loading the expression matrix, this is a known pandas/PyArrow bug ([pandas #51664](https://github.com/pandas-dev/pandas/issues/51664)). The dataset has been pre-processed to fix this issue, but if you process the raw data yourself:

```python
import pyarrow.parquet as pq

# Fix the phantom index column
table = pq.read_table("your_file.parquet")
if "__index_level_0__" in table.column_names:
    table = table.drop(["__index_level_0__"])
    table.to_pandas().to_parquet("fixed_file.parquet", index=False)
```

### **Performance Optimization**
```python
# Efficient loading with specific columns
metadata_subset = pd.read_parquet(
    "skeletal_muscle_10x_sample_metadata.parquet",
    columns=['Age_group', 'annotation_level0', 'Sex']
)

# Memory-efficient expression analysis
top_genes = ['MYH1', 'MYH2', 'MYH7', 'ACTN3']  # Focus on specific genes
expr_subset = expression[top_genes]
```

### **Data Validation**
```python
# Verify data integrity
assert expression.shape == (183161, 29400), "Expression matrix size mismatch"
assert len(metadata) == len(expression), "Metadata-expression size mismatch"
assert all(metadata.index == expression.index), "Index mismatch"
print("✅ Data integrity verified")
```

---

## 📊 Dataset Statistics Summary

| **Metric** | **Value** | **Details** |
|------------|-----------|-------------|
| **Total Cells** | 183,161 | High-quality single cells + nuclei |
| **Total Genes** | 29,400 | Protein-coding genes (GENCODE) |
| **Age Range** | 15-75 years | 8 age groups, balanced young/old |
| **Donors** | 17 individuals | Biological replicates |
| **Cell Types** | 36 major types | 82 fine-grained subtypes available |
| **Data Size** | 1.6GB total | Optimized for analysis |
| **Sparsity** | 95.4% | Typical single-cell sparsity |
| **Quality Score** | 100% validation | All tests passed |

---

**🔬 Ready for longevity research • 🧬 Human muscle aging • 💻 Analysis-optimized • 📊 Comprehensively documented** 