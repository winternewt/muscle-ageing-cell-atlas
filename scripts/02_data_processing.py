#!/usr/bin/env python3
"""
Phase 2: Data Processing for Human Skeletal Muscle Aging Atlas
==============================================================

Processes the H5AD file into HuggingFace-compatible parquet files:
- Expression matrix (sparse -> dense conversion)
- Sample metadata (cell-level information)
- Feature metadata (gene information)
- Dimensionality reduction projections (scVI, UMAP, PCA, t-SNE)
- Unstructured metadata (all additional data)

Requirements:
- Large memory for 183K × 29K matrix processing
- Sparse matrix handling for efficiency
- Proper data type optimization
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import pyarrow.parquet as pq
import warnings

# Configure scanpy
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def make_json_serializable(obj: Any) -> Any:
    """Convert numpy arrays and other non-serializable objects for JSON"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def log_memory_usage(stage: str, adata: sc.AnnData) -> None:
    """Log memory usage and dataset info"""
    memory_mb = adata.X.data.nbytes / 1024**2 if sparse.issparse(adata.X) else adata.X.nbytes / 1024**2
    logger.info(f"{stage}: Shape {adata.shape}, Memory: {memory_mb:.1f}MB")

def fix_pandas_index_column_bug(parquet_file: Path) -> bool:
    """
    Fix the pandas __index_level_0__ bug in parquet files
    
    This is a known bug in pandas/PyArrow where pandas saves the index as an extra 
    '__index_level_0__' column when writing to parquet format. 
    This is a known upstream issue with no planned fix
    
    References:
    - https://github.com/pandas-dev/pandas/issues/51664
    - https://github.com/pola-rs/polars/issues/7291

    Args:
        parquet_file: Path to the parquet file to fix
        
    Returns:
        bool: True if fix was applied successfully, False otherwise
    """
    logger.info(f"🔧 Checking for pandas __index_level_0__ bug in {parquet_file.name}")
    
    try:
        # Check if the bug exists
        pf = pq.ParquetFile(parquet_file)
        schema_names = pf.schema_arrow.names
        
        if '__index_level_0__' not in schema_names:
            logger.info("✅ No __index_level_0__ column found - file is clean")
            return True
            
        logger.warning(f"🐛 Found pandas __index_level_0__ bug - fixing...")
        logger.info(f"   Current columns: {len(schema_names)} (expected: {len(schema_names)-1})")
        
        # Create backup
        backup_file = parquet_file.with_suffix('.backup.parquet')
        if not backup_file.exists():
            shutil.copy2(parquet_file, backup_file)
            logger.info(f"📦 Backup created: {backup_file.name}")
        
        # Apply fix using PyArrow
        table = pq.read_table(parquet_file)
        
        # Filter out the problematic column
        columns_to_keep = [name for name in table.column_names if name != '__index_level_0__']
        clean_table = table.select(columns_to_keep)
        
        # Write clean table to temporary file first
        temp_file = parquet_file.with_suffix('.temp.parquet')
        pq.write_table(clean_table, temp_file, compression='snappy')
        
        # Verify the fix
        temp_pf = pq.ParquetFile(temp_file)
        temp_schema_names = temp_pf.schema_arrow.names
        
        if '__index_level_0__' not in temp_schema_names:
            # Replace original with fixed version
            shutil.move(temp_file, parquet_file)
            logger.info(f"✅ Fixed pandas __index_level_0__ bug")
            logger.info(f"   Column count: {len(schema_names)} → {len(temp_schema_names)}")
            return True
        else:
            # Fix failed, clean up
            temp_file.unlink()
            logger.error("❌ Fix verification failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error fixing pandas index bug: {e}")
        return False

def process_expression_matrix(adata: sc.AnnData, method: str, output_dir: Path) -> Dict[str, Any]:
    """
    Process and save expression matrix
    
    Strategy:
    - Check sparsity and memory requirements
    - Convert to dense if manageable, keep sparse if too large
    - Use appropriate data types (float32) for efficiency
    """
    logger.info("Starting expression matrix processing...")
    log_memory_usage("Expression matrix", adata)
    
    # Calculate memory requirements for dense conversion
    dense_memory_gb = (adata.n_obs * adata.n_vars * 4) / (1024**3)  # float32 = 4 bytes
    sparsity = 1.0 - (adata.X.nnz / (adata.n_obs * adata.n_vars))
    
    logger.info(f"Dense conversion would require: {dense_memory_gb:.2f}GB")
    logger.info(f"Current sparsity: {sparsity:.2%}")
    
    output_file = output_dir / f"skeletal_muscle_{method}_expression.parquet"
    
    if dense_memory_gb > 8.0:  # If >8GB, process in chunks
        logger.info("Large matrix detected, processing in chunks...")
        chunk_size = 10000
        chunks = []
        
        for i in range(0, adata.n_obs, chunk_size):
            end_idx = min(i + chunk_size, adata.n_obs)
            chunk = adata[i:end_idx, :].copy()
            
            if sparse.issparse(chunk.X):
                chunk_dense = chunk.X.toarray().astype(np.float32)
            else:
                chunk_dense = chunk.X.astype(np.float32)
            
            chunk_df = pd.DataFrame(
                chunk_dense,
                index=chunk.obs_names,
                columns=chunk.var_names
            )
            chunks.append(chunk_df)
            logger.info(f"Processed chunk {i//chunk_size + 1}/{(adata.n_obs-1)//chunk_size + 1}")
        
        # Combine chunks
        expression_df = pd.concat(chunks, axis=0)
        del chunks  # Free memory
        
    else:
        # Convert to dense in one go
        logger.info("Converting to dense matrix...")
        if sparse.issparse(adata.X):
            expression_data = adata.X.toarray().astype(np.float32)
        else:
            expression_data = adata.X.astype(np.float32)
        
        expression_df = pd.DataFrame(
            expression_data,
            index=adata.obs_names,
            columns=adata.var_names
        )
    
    # Save with compression
    logger.info(f"Saving expression matrix: {expression_df.shape}")
    expression_df.to_parquet(output_file, compression='snappy')
    
    # Apply pandas __index_level_0__ bug fix
    # This is a known issue where pandas saves the index as an extra column
    fix_success = fix_pandas_index_column_bug(output_file)
    
    stats = {
        'file': str(output_file),
        'shape': list(expression_df.shape),
        'memory_gb': dense_memory_gb,
        'sparsity_percent': sparsity * 100,
        'dtype': str(expression_df.dtypes.iloc[0]),
        'pandas_index_bug_fixed': fix_success
    }
    
    logger.info(f"✅ Expression matrix saved: {expression_df.shape}")
    return stats

def process_sample_metadata(adata: sc.AnnData, method: str, output_dir: Path) -> Dict[str, Any]:
    """Process and save sample (cell) metadata"""
    logger.info("Processing sample metadata...")
    
    sample_metadata = adata.obs.copy()
    
    # Verify critical columns exist
    critical_cols = ['Age_group', 'Sex', 'annotation_level0', 'DonorID', 'batch']
    missing_cols = [col for col in critical_cols if col not in sample_metadata.columns]
    
    if missing_cols:
        logger.warning(f"Missing critical columns: {missing_cols}")
    else:
        logger.info("✅ All critical metadata columns present")
    
    # Add standardized age column if needed
    if 'age_numeric' not in sample_metadata.columns and 'Age_group' in sample_metadata.columns:
        # Convert age groups to numeric (use midpoint of range)
        age_mapping = {
            '15-20': 17.5, '20-25': 22.5, '25-30': 27.5, '35-40': 37.5,
            '50-55': 52.5, '55-60': 57.5, '60-65': 62.5, '70-75': 72.5
        }
        sample_metadata['age_numeric'] = sample_metadata['Age_group'].map(age_mapping)
        logger.info("Added numeric age column")
    
    # Optimize data types
    for col in sample_metadata.columns:
        if sample_metadata[col].dtype == 'object':
            # Convert categorical strings to category type for efficiency
            if sample_metadata[col].nunique() < len(sample_metadata) * 0.5:
                sample_metadata[col] = sample_metadata[col].astype('category')
    
    output_file = output_dir / f"skeletal_muscle_{method}_sample_metadata.parquet"
    sample_metadata.to_parquet(output_file, compression='snappy')
    
    stats = {
        'file': str(output_file),
        'shape': list(sample_metadata.shape),
        'columns': list(sample_metadata.columns),
        'missing_columns': missing_cols,
        'age_groups': sample_metadata['Age_group'].value_counts().to_dict() if 'Age_group' in sample_metadata.columns else {},
        'cell_types': sample_metadata['annotation_level0'].value_counts().head(10).to_dict() if 'annotation_level0' in sample_metadata.columns else {}
    }
    
    logger.info(f"✅ Sample metadata saved: {sample_metadata.shape}")
    return stats

def process_feature_metadata(adata: sc.AnnData, method: str, output_dir: Path) -> Dict[str, Any]:
    """Process and save feature (gene) metadata"""
    logger.info("Processing feature metadata...")
    
    feature_metadata = adata.var.copy()
    
    # Ensure gene IDs are present
    if 'gene_ids' not in feature_metadata.columns:
        feature_metadata['gene_ids'] = feature_metadata.index
        logger.info("Added gene_ids column from index")
    
    # Verify gene symbols
    if 'SYMBOL' in feature_metadata.columns:
        n_symbols = feature_metadata['SYMBOL'].notna().sum()
        logger.info(f"Gene symbols available for {n_symbols}/{len(feature_metadata)} genes")
    
    output_file = output_dir / f"skeletal_muscle_{method}_feature_metadata.parquet"
    feature_metadata.to_parquet(output_file, compression='snappy')
    
    stats = {
        'file': str(output_file),
        'shape': list(feature_metadata.shape),
        'columns': list(feature_metadata.columns),
        'has_symbols': 'SYMBOL' in feature_metadata.columns,
        'has_ensembl': 'ENSEMBL' in feature_metadata.columns
    }
    
    logger.info(f"✅ Feature metadata saved: {feature_metadata.shape}")
    return stats

def compute_missing_projections(adata: sc.AnnData) -> Dict[str, bool]:
    """Compute missing dimensionality reductions"""
    logger.info("Checking and computing missing projections...")
    
    computed = {}
    
    # Check PCA
    if 'X_pca' not in adata.obsm:
        logger.info("Computing PCA (50 components)...")
        try:
            sc.pp.pca(adata, n_comps=50, svd_solver='arpack')
            computed['X_pca'] = True
            logger.info("✅ PCA computed")
        except Exception as e:
            logger.error(f"PCA computation failed: {e}")
            computed['X_pca'] = False
    else:
        computed['X_pca'] = True
        logger.info("✅ PCA already exists")
    
    # Check t-SNE
    if 'X_tsne' not in adata.obsm:
        logger.info("Computing t-SNE...")
        try:
            # Use existing neighbors if available, otherwise compute
            if 'neighbors' not in adata.uns:
                logger.info("Computing neighbors for t-SNE...")
                sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
            
            sc.tl.tsne(adata, perplexity=30, n_jobs=8)
            computed['X_tsne'] = True
            logger.info("✅ t-SNE computed")
        except Exception as e:
            logger.error(f"t-SNE computation failed: {e}")
            computed['X_tsne'] = False
    else:
        computed['X_tsne'] = True
        logger.info("✅ t-SNE already exists")
    
    return computed

def process_projections(adata: sc.AnnData, method: str, output_dir: Path) -> Dict[str, Any]:
    """Process and save all dimensionality reduction projections"""
    logger.info("Processing dimensionality reduction projections...")
    
    # First compute any missing projections
    computed_status = compute_missing_projections(adata)
    
    projection_stats = {}
    expected_projections = ['X_scVI', 'X_umap', 'X_pca', 'X_tsne']
    
    for proj_name in expected_projections:
        if proj_name in adata.obsm:
            proj_data = adata.obsm[proj_name]
            
            # Convert to DataFrame
            proj_df = pd.DataFrame(
                proj_data,
                index=adata.obs_names,
                columns=[f"{proj_name.split('_')[1].upper()}{i+1}" for i in range(proj_data.shape[1])]
            )
            
            # Save projection
            output_file = output_dir / f"skeletal_muscle_{method}_projection_{proj_name}.parquet"
            proj_df.to_parquet(output_file, compression='snappy')
            
            projection_stats[proj_name] = {
                'file': str(output_file),
                'shape': list(proj_df.shape),
                'computed_now': computed_status.get(proj_name, False)
            }
            
            logger.info(f"✅ Saved {proj_name}: {proj_df.shape}")
        else:
            logger.warning(f"❌ {proj_name} not available")
            projection_stats[proj_name] = {'available': False}
    
    return projection_stats

def process_unstructured_metadata(adata: sc.AnnData, method: str, output_dir: Path) -> Dict[str, Any]:
    """Process and save unstructured metadata (uns)"""
    logger.info("Processing unstructured metadata...")
    
    try:
        # Make data JSON serializable
        unstructured_data = make_json_serializable(adata.uns)
        
        output_file = output_dir / f"skeletal_muscle_{method}_unstructured_metadata.json"
        
        with open(output_file, 'w') as f:
            json.dump(unstructured_data, f, indent=2)
        
        # Count keys and estimate size
        key_count = len(unstructured_data) if isinstance(unstructured_data, dict) else 0
        file_size_mb = output_file.stat().st_size / (1024**2)
        
        stats = {
            'file': str(output_file),
            'key_count': key_count,
            'file_size_mb': round(file_size_mb, 2),
            'top_keys': list(unstructured_data.keys())[:10] if isinstance(unstructured_data, dict) else []
        }
        
        logger.info(f"✅ Unstructured metadata saved: {key_count} keys, {file_size_mb:.1f}MB")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to process unstructured metadata: {e}")
        return {'error': str(e)}

def main():
    """Main processing function"""
    start_time = time.time()
    logger.info("=== Phase 2: Data Processing Started ===")
    
    # Paths
    data_file = Path("data/SKM_human_pp_cells2nuclei_2023-06-22.h5ad")
    output_dir = Path("processed")
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    method = "10x"  # From exploration results
    
    # Load data
    logger.info(f"Loading data from {data_file}...")
    try:
        adata = sc.read_h5ad(data_file)
        logger.info(f"✅ Data loaded: {adata.shape}")
        log_memory_usage("Initial", adata)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Processing results tracking
    processing_results = {
        'dataset_info': {
            'shape': list(adata.shape),
            'method': method,
            'processing_time': None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    try:
        # Task 2.1: Expression Matrix
        logger.info("\n🧬 Task 2.1: Processing Expression Matrix")
        processing_results['expression'] = process_expression_matrix(adata, method, output_dir)
        
        # Task 2.2: Sample Metadata
        logger.info("\n📊 Task 2.2: Processing Sample Metadata")
        processing_results['sample_metadata'] = process_sample_metadata(adata, method, output_dir)
        
        # Task 2.3: Feature Metadata
        logger.info("\n🧪 Task 2.3: Processing Feature Metadata")
        processing_results['feature_metadata'] = process_feature_metadata(adata, method, output_dir)
        
        # Task 2.4: Dimensionality Reductions
        logger.info("\n📈 Task 2.4: Processing Projections")
        processing_results['projections'] = process_projections(adata, method, output_dir)
        
        # Task 2.5: Unstructured Metadata
        logger.info("\n📋 Task 2.5: Processing Unstructured Metadata")
        processing_results['unstructured'] = process_unstructured_metadata(adata, method, output_dir)
        
        # Save processing summary
        processing_time = time.time() - start_time
        processing_results['dataset_info']['processing_time'] = f"{processing_time:.1f}s"
        
        summary_file = output_dir / "phase2_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(processing_results, f, indent=2)
        
        logger.info(f"\n✅ Phase 2 Processing Complete!")
        logger.info(f"⏱️  Total time: {processing_time:.1f}s")
        logger.info(f"📄 Summary saved: {summary_file}")
        
        # List all created files
        logger.info("\n📁 Created Files:")
        for file_path in output_dir.glob("skeletal_muscle_*.parquet"):
            size_mb = file_path.stat().st_size / (1024**2)
            logger.info(f"  {file_path.name} ({size_mb:.1f}MB)")
        
        for file_path in output_dir.glob("skeletal_muscle_*.json"):
            size_mb = file_path.stat().st_size / (1024**2)
            logger.info(f"  {file_path.name} ({size_mb:.1f}MB)")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        processing_results['error'] = str(e)
        
        # Save error summary
        error_file = output_dir / "phase2_error_summary.json"
        with open(error_file, 'w') as f:
            json.dump(processing_results, f, indent=2)
        
        raise

if __name__ == "__main__":
    main()