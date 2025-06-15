# 🏆 Longevity Hackathon - Best Scoring Strategy Guide

> **Based on analysis of the official example:** [tabula-muris-senis-bladder-smartseq2](https://huggingface.co/datasets/longevity-db/tabula-muris-senis-bladder-smartseq2)

---

## 🚀 Step-by-Step Implementation Plan

### **Phase 1: Setup & Data Source Selection**

#### 1.1 Repository Creation
```python
from huggingface_hub import create_repo

# Set your group name
group_name = "longevity-genie"  # Replace with your team name

# Authenticate with write token
token = "your_huggingface_token_here"

# Create repository
repo_id = create_repo(
    repo_id=f"longevity-db/{group_name}",
    token=token,
    private=False,
    repo_type="dataset"
)
print(f"Repository created: {repo_id}")
```

#### 1.2 Data Source Selection Strategy
- [ ] **Target non-cellxgene sources** (2x difficulty multiplier)
- [ ] **Check Karl's list** for 10% bonus datasets
- [ ] **Verify uniqueness** on Discord to avoid duplication penalty
- [ ] **Post your claim** in Discord channel

### **Phase 2: Data Processing Pipeline**

#### 2.1 Required File Structure
```
{group_name}/
├── {tissue}_{method}_expression.parquet          # Main X matrix
├── {tissue}_{method}_sample_metadata.parquet     # Per-sample metadata  
├── {tissue}_{method}_feature_metadata.parquet    # Per-feature metadata
├── {tissue}_{method}_projection_X_pca.parquet     # PCA embeddings
├── {tissue}_{method}_projection_X_tsne.parquet    # t-SNE visualization
├── {tissue}_{method}_projection_X_umap.parquet    # UMAP visualization
├── {tissue}_{method}_unstructured_metadata.json  # Additional metadata
└── README.md                                      # Dataset card
```

#### 2.2 Data Processing Checklist
- [ ] **Expression Matrix** (`expression.parquet`)
  - [ ] Samples as rows, features as columns
  - [ ] Use `float32` for memory efficiency
  - [ ] Maintain original sample IDs as index
  - [ ] Shape example: `(2,432 × 21,069)`

- [ ] **Sample Metadata** (`sample_metadata.parquet`)  
  - [ ] One row per sample/cell
  - [ ] Include age information (`age`, `development_stage`)
  - [ ] Include biological metadata (`sex`, `tissue`, `cell_type`)
  - [ ] Include technical metadata (`n_genes`, `n_counts`)
  - [ ] Use categorical dtypes for efficiency

- [ ] **Feature Metadata** (`feature_metadata.parquet`)
  - [ ] One row per feature/gene
  - [ ] Include feature names and descriptions
  - [ ] Include statistical summaries (`means`, `dispersions`)
  - [ ] Include feature types (`feature_type`, `feature_biotype`)

- [ ] **Dimensionality Reductions** (`projection_*.parquet`)
  - [ ] PCA: First 50 components
  - [ ] t-SNE: 2D for visualization
  - [ ] UMAP: 2D for visualization
  - [ ] Same sample order as expression matrix

- [ ] **Unstructured Metadata** (`unstructured_metadata.json`)
  - [ ] Convert numpy arrays to lists for JSON serialization
  - [ ] Include processing parameters
  - [ ] Include original dataset citations

### **Phase 3: Documentation Excellence**

#### 3.1 YAML Frontmatter (Required)
```yaml
---
license: mit
tags:
- longevity
- aging
- [your-specific-tags]
- [tissue-type]
- [method-type]
pretty_name: "[Tissue] from [Dataset Name] - [Method]"
size_categories:
- [appropriate-size-range]  # e.g., "10K<n<100K"
---
```

#### 3.2 Required README Sections

**🔥 Critical Sections (High Scoring Weight):**

- [ ] **# Dataset Title**
  - [ ] Clear, descriptive title
  - [ ] Biological context in subtitle

- [ ] **## Biological Context & Significance**
  - [ ] Explain relevance to aging research
  - [ ] Quote from original paper (with citation)
  - [ ] Describe what can be learned from this data

- [ ] **## Dataset Overview**
  - [ ] Data modality (RNA-seq, proteomics, etc.)
  - [ ] Organism and tissue type
  - [ ] Age range and sample size
  - [ ] Processing method details

- [ ] **## File Descriptions**
  - [ ] **Individual section for each parquet file**
  - [ ] Shape information (rows × columns)
  - [ ] Column descriptions with biological meaning
  - [ ] **Embedded screenshots** of data samples
  - [ ] Data type specifications

**Example file description:**
```markdown
## `bladder_smartseq2_expression.parquet`

`bladder_smartseq2_expression.parquet` is a 2,432 rows × 21,069 columns dataset. Each row is a single cell's gene expression across 21,069 mouse genes. This is typically the `X` matrix for ML modeling, and would need to be randomly split for test/train/validation sets.

![Data Preview](screenshot_url_here)
```

- [ ] **## Usage Example**
  - [ ] Complete Python code snippet
  - [ ] Show how to load each file type
  - [ ] Basic analysis or visualization example
  - [ ] Include expected output

- [ ] **## Citations & References**
  - [ ] Original paper with DOI
  - [ ] Original data source (GEO, ArrayExpress, etc.)
  - [ ] Your processing code repository
  - [ ] AnnData or other format specifications

- [ ] **## Data Schema**
  - [ ] Complete list of all columns
  - [ ] Units for measurements
  - [ ] Biological interpretation of each field
  - [ ] Missing value handling

- [ ] **## Biological Glossary**
  - [ ] Define domain-specific terms
  - [ ] Explain biological processes
  - [ ] Make accessible to ML practitioners

#### 3.3 Visual Enhancement Checklist
- [ ] **Embedded Images**
  - [ ] Screenshot of data loading
  - [ ] Sample data preview tables
  - [ ] Simple visualization (if applicable)
  - [ ] Upload images to HuggingFace and embed

- [ ] **Professional Formatting**
  - [ ] Proper markdown headers
  - [ ] Code blocks with syntax highlighting
  - [ ] Consistent section structure
  - [ ] Clean, readable layout

### **Phase 4: Quality Assurance**

#### 4.1 Data Validation
- [ ] **File Integrity**
  - [ ] All parquet files loadable with pandas
  - [ ] Index consistency across files
  - [ ] No missing critical columns
  - [ ] Appropriate data types

- [ ] **Biological Validity**
  - [ ] Age ranges make biological sense
  - [ ] Sample metadata is consistent
  - [ ] Feature annotations are accurate
  - [ ] No obvious data corruption

#### 4.2 Documentation Review
- [ ] **Completeness Check**
  - [ ] All required sections present
  - [ ] No broken links or missing images
  - [ ] Code examples are runnable
  - [ ] Citations are complete and accurate

- [ ] **ML Accessibility**
  - [ ] Non-biologists can understand the data
  - [ ] Clear target variables identified
  - [ ] Potential use cases explained
  - [ ] Limitations and caveats documented

### **Phase 5: Upload & Final Polish**

#### 5.1 HuggingFace Upload
```bash
# Clone your repository
git clone https://huggingface.co/datasets/longevity-db/{group_name}
cd {group_name}

# Add your files
cp *.parquet ./
cp *.json ./
cp README.md ./

# Commit and push
git add .
git commit -m "Initial dataset upload"
git push
```

#### 5.2 Final Checklist
- [ ] **Repository is live** and publicly accessible
- [ ] **Dataset viewer works** (or limitation is documented)
- [ ] **All links are functional**
- [ ] **Images are properly embedded**
- [ ] **Code examples run without errors**
- [ ] **Posted completion** in Discord

---

## 🎯 Scoring Optimization Tips

### **Maximum Impact Areas:**
1. **Documentation Quality (40% of score)**
   - Focus on biological context and ML accessibility
   - Include comprehensive data schema
   - Add visual examples and professional formatting

2. **Data Processing Excellence (30% of score)**
   - Follow exact file structure from example
   - Maintain data integrity and efficiency
   - Include all required metadata types

3. **Biological Translation (20% of score)**
   - Make domain knowledge accessible
   - Explain aging research relevance
   - Document limitations and caveats

4. **Technical Polish (10% of score)**
   - Professional HuggingFace presentation
   - Clean code examples
   - Proper licensing and citations

### **Bonus Opportunities:**
- [ ] **Target Karl's list** → +10% score bonus
- [ ] **Non-cellxgene source** → 2x difficulty multiplier
- [ ] **Large dataset size** → Higher base score
- [ ] **Unique dataset** → No duplication penalty

### **Common Pitfalls to Avoid:**
- ❌ Using CSV instead of Parquet
- ❌ Missing biological context
- ❌ Incomplete data schema documentation
- ❌ No usage examples
- ❌ Poor visual presentation
- ❌ Duplicating existing datasets
