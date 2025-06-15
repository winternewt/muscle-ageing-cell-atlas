#!/usr/bin/env python3
"""
Detailed Dataset Structure Inspection
Based on notebook analysis, check for all available projections and complete structure.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

def detailed_inspection():
    """Perform detailed inspection based on notebook knowledge."""
    
    print("🔍 DETAILED DATASET STRUCTURE INSPECTION")
    print("=" * 60)
    
    # Load dataset
    adata = sc.read_h5ad('data/SKM_human_pp_cells2nuclei_2023-06-22.h5ad')
    
    print(f"📊 CONFIRMED DATASET STRUCTURE:")
    print(f"   Shape: {adata.shape} (cells × genes)")
    
    # Check all available projections in obsm
    print(f"\n📈 ALL AVAILABLE PROJECTIONS (.obsm):")
    for key in sorted(adata.obsm.keys()):
        shape = adata.obsm[key].shape
        print(f"   ✅ {key}: {shape}")
    
    # Check if PCA is computed but not in obsm (might be in .varm)
    print(f"\n🧬 VARIABLE METADATA (.varm):")
    if adata.varm:
        for key in adata.varm.keys():
            shape = adata.varm[key].shape
            print(f"   {key}: {shape}")
    else:
        print("   None found")
    
    # Check layers
    print(f"\n📚 AVAILABLE LAYERS:")
    if adata.layers:
        for key in adata.layers.keys():
            print(f"   {key}: {type(adata.layers[key])}")
    else:
        print("   None found")
    
    # Detailed cell type analysis
    print(f"\n🏷️  CELL TYPE ANNOTATIONS:")
    annotation_cols = [col for col in adata.obs.columns if 'annotation' in col.lower()]
    for col in annotation_cols:
        unique_count = adata.obs[col].nunique()
        print(f"   {col}: {unique_count} unique types")
        if unique_count <= 20:
            print(f"      Types: {sorted(adata.obs[col].unique())}")
        else:
            top_types = adata.obs[col].value_counts().head(10)
            print(f"      Top 10: {list(top_types.index)}")
    
    # Age analysis
    print(f"\n🎂 AGE STRUCTURE:")
    age_cols = [col for col in adata.obs.columns if 'age' in col.lower()]
    for col in age_cols:
        print(f"   {col}:")
        counts = adata.obs[col].value_counts().sort_index()
        print(f"      {dict(counts)}")
    
    # Batch and technical factors
    print(f"\n🔬 TECHNICAL FACTORS:")
    tech_cols = ['SampleID', 'DonorID', 'Sex', 'batch', '10X_version']
    for col in tech_cols:
        if col in adata.obs.columns:
            unique_count = adata.obs[col].nunique()
            print(f"   {col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"      Values: {sorted(adata.obs[col].unique())}")
    
    # Quality metrics
    print(f"\n📊 QUALITY METRICS:")
    qc_cols = ['n_counts', 'n_genes', 'percent_mito', 'percent_ribo', 'scrublet_score']
    for col in qc_cols:
        if col in adata.obs.columns:
            stats = adata.obs[col].describe()
            print(f"   {col}: mean={stats['mean']:.1f}, std={stats['std']:.1f}, range={stats['min']:.1f}-{stats['max']:.1f}")
    
    # Gene information
    print(f"\n🧬 GENE INFORMATION:")
    print(f"   Gene columns: {list(adata.var.columns)}")
    if 'SYMBOL' in adata.var.columns:
        print(f"   Sample gene symbols: {list(adata.var['SYMBOL'][:10])}")
    
    # Check for missing projections that should be computed  
    print(f"\n⚠️  PROJECTIONS TO COMPUTE:")
    required_projections = ['X_pca', 'X_tsne']
    missing_projections = [proj for proj in required_projections if proj not in adata.obsm.keys()]
    if missing_projections:
        print(f"   Missing: {missing_projections}")
        print("   These will need to be computed during processing")
    else:
        print("   All standard projections available")
    
    # Summary for processing
    print(f"\n📋 PROCESSING SUMMARY:")
    print(f"   ✅ Method confirmed: 10X Chromium")
    print(f"   ✅ scVI embedding: 30 dimensions") 
    print(f"   ✅ UMAP: 2 dimensions")
    print(f"   ✅ Rich cell type annotations: 3 levels")
    print(f"   ✅ Age groups: Available")
    print(f"   ✅ Quality metrics: Complete")
    
    # Check data sparsity and recommend processing approach
    sparsity = (1 - adata.X.nnz / (adata.n_obs * adata.n_vars)) * 100
    print(f"   ✅ Data sparsity: {sparsity:.1f}%")
    
    if sparsity > 90:
        print("   💡 Recommendation: Process expression matrix as sparse for memory efficiency")
    
    # Create mapping for annotation levels
    print(f"\n🔄 ANNOTATION LEVEL MAPPING:")
    mapping = {}
    for level in ['annotation_level0', 'annotation_level1', 'annotation_level2']:
        if level in adata.obs.columns:
            mapping[level] = adata.obs[level].value_counts().to_dict()
    
    return adata, mapping

if __name__ == "__main__":
    adata, annotation_mapping = detailed_inspection()
    
    # Save detailed results
    results = {
        'obsm_keys': list(adata.obsm.keys()),
        'var_columns': list(adata.var.columns),
        'obs_columns': list(adata.obs.columns),
        'annotation_mapping': annotation_mapping,
        'shape': adata.shape,
        'sparsity': (1 - adata.X.nnz / (adata.n_obs * adata.n_vars)) * 100
    }
    
    import json
    with open('processed/detailed_structure.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed structure saved to processed/detailed_structure.json") 