---
license: mit
tags:
- longevity
- aging
- tabula-muris-senis
- single-cell
- smartseq2
pretty_name: Bladder tissue from Tabula Muris Senis,  SmartSeq2 Library prep
size_categories:
- 10K<n<100K
---

# Bladder Tissue from Tabula Muris Senis

Tabula Muris Senis is a mammalian aging single-cell gene expression dataset, downloaded from https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb. This dataset represents the Bladder tissue, using the [SmartSeq2](https://www.nature.com/articles/nprot.2014.006) full-length mRNA library preparation method for single cells.


Code to download and process this dataset is available in: https://github.com/seanome/2025-longevity-x-ai-hackathon

> Ageing is characterized by a progressive loss of physiological integrity, leading to impaired function and increased vulnerability to death. Despite rapid advances over recent years, many of the molecular and cellular processes that underlie the progressive loss of healthy physiology are poorly understood. To gain a better insight into these processes, here we generate a single-cell transcriptomic atlas across the lifespan of Mus musculus that includes data from 23 tissues and organs. We found cell-specific changes occurring across multiple cell types and organs, as well as age-related changes in the cellular composition of different organs. Using single-cell transcriptomic data, we assessed cell-type-specific manifestations of different hallmarks of ageing—such as senescence, genomic instability and changes in the immune system. This transcriptomic atlas—which we denote Tabula Muris Senis, or ‘Mouse Ageing Cell Atlas’—provides molecular information about how the most important hallmarks of ageing are reflected in a broad range of tissues and cell types.

Dataset structure is originally from [AnnData](https://anndata.readthedocs.io/en/latest/index.html),

Descriptions of each data file is below.

## `bladder_smartseq2_expression.parquet`

`bladder_smartseq2_expression.parquet` is a 2,432 rows x 21,069 columns dataset. Each row is a single cell's gene expression across 21,069 mouse genes. This is typically the `X` matrix for ML modeling, and would need to be randomly split for test/train/validation sets.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/684ce3e549cb60c8c1a7fabf/m0hJOd8X7QRwD4Jyhv1k0.png)

## `bladder_smartseq2_sample_metadata.parquet`

`bladder_smartseq2_sample_metadata.parquet` is a 2,432 rows x 30 columns dataset. Each row represents the metadata for a single cell, e.g. what mouse it came from (`donor_id`), the sex of the mouse, number of genes expressed (`n_genes`), number of total read counts per cell (`n_counts`), cell type annotation (`cell_type`), age of the mouse (`age` or also `development_stage`)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/684ce3e549cb60c8c1a7fabf/A2LYhG7TdfiYB8j119u-Z.png)

## `bladder_smartseq2_feature_metadata.parquet`

`bladder_smartseq2_feature_metadata.parquet` is a 21,069 rows x 11 columns dataset. Each row represents the metadata for each gene, e.g. number of cells expressing it (`n_cells`), mean gene expression (`means`), if it's a highly variable gene (`highly_variable`), the type of the feature (`feature_type`)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/684ce3e549cb60c8c1a7fabf/dY4ZuCO97ZCWBbFsRPToW.png)


## `bladder_smartseq2_unstructured_metadata.json`

`bladder_smartseq2_unstructured_metadata.json` is a key-value store of unstructured metadata information about the dataset.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/684ce3e549cb60c8c1a7fabf/tZOwfs4Svf3TG0SUF0Wj6.png)


## `bladder_smartseq2_projection_*.parquet`

`bladder_smartseq2_projection_*.parquet` are transformations of the expression data using either PCA (first 50 PCs), tSNE (2 dimensions for visualizationA), or UMAP (2 dimensions for visualization).

```
bladder_smartseq2_projection_X_pca.parquet
bladder_smartseq2_projection_X_tsne.parquet
bladder_smartseq2_projection_X_umap.parquet
```
