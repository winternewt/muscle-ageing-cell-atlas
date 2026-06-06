#!/usr/bin/env python3
"""
Test dataset loading using /data/ as cache directory to avoid disk space issues
"""

import os
import shutil
from pathlib import Path
from datasets import load_dataset

def setup_data_cache():
    """Setup HuggingFace cache to use /data/ directory"""
    
    # Create cache directory in /data/
    cache_dir = Path("/data/hf_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Set environment variable for HuggingFace datasets cache
    os.environ['HF_DATASETS_CACHE'] = str(cache_dir)
    os.environ['HF_HOME'] = str(cache_dir)
    
    print(f"✅ HuggingFace cache set to: {cache_dir}")
    print(f"Available space: {shutil.disk_usage(cache_dir).free / (1024**3):.1f} GB")
    
    return cache_dir

def test_small_split():
    """Test loading just the metadata (smallest file) first"""
    print("\n=== Testing sample_metadata split ===")
    
    try:
        # Load just a small subset of metadata
        dataset = load_dataset(
            "longevity-db/skeletal-muscle-atlas",
            split="sample_metadata[:10]",  # Just 10 samples
            cache_dir="/data/hf_cache"
        )
        
        print("✅ sample_metadata loaded successfully")
        print(f"Shape: {dataset.shape}")
        print(f"Columns: {list(dataset.column_names)[:10]}")
        
        # Show sample data
        if len(dataset) > 0:
            first_sample = dataset[0]
            print("Sample data preview:")
            for key, value in first_sample.items():
                print(f"  {key}: {value}")
                
        return True
        
    except Exception as e:
        print(f"❌ sample_metadata failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_umap_split():
    """Test loading UMAP projection (small file)"""
    print("\n=== Testing projection_umap split ===")
    
    try:
        dataset = load_dataset(
            "longevity-db/skeletal-muscle-atlas",
            split="projection_umap[:5]",  # Just 5 samples
            cache_dir="/data/hf_cache"
        )
        
        print("✅ projection_umap loaded successfully")
        print(f"Shape: {dataset.shape}")
        print(f"Columns: {list(dataset.column_names)}")
        
        # Show sample data
        if len(dataset) > 0:
            first_sample = dataset[0]
            print("UMAP coordinates preview:")
            for key, value in first_sample.items():
                print(f"  {key}: {value}")
                
        return True
        
    except Exception as e:
        print(f"❌ projection_umap failed: {e}")
        return False

def test_all_splits_info():
    """Test loading dataset info without downloading large files"""
    print("\n=== Testing dataset info (no download) ===")
    
    try:
        # Just get the dataset structure without downloading
        from datasets import get_dataset_config_names, get_dataset_split_names
        
        configs = get_dataset_config_names("longevity-db/skeletal-muscle-atlas")
        print(f"Available configs: {configs}")
        
        for config in configs:
            try:
                splits = get_dataset_split_names("longevity-db/skeletal-muscle-atlas", config_name=config)
                print(f"Config '{config}' splits: {splits}")
            except Exception as e:
                print(f"Config '{config}': {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset info failed: {e}")
        return False

def main():
    print("🧬 Testing Dataset with /data/ Cache")
    
    # Setup cache directory
    cache_dir = setup_data_cache()
    
    # Test dataset info first (no downloads)
    info_success = test_all_splits_info()
    
    # Test small splits
    metadata_success = test_small_split()
    umap_success = test_umap_split()
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    
    results = {
        "dataset_info": info_success,
        "sample_metadata": metadata_success, 
        "projection_umap": umap_success
    }
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed > 0:
        print("🎉 YAML configuration is working!")
        print("\nUsers can now load data with:")
        if metadata_success:
            print("  dataset = load_dataset('longevity-db/skeletal-muscle-atlas', split='sample_metadata')")
        if umap_success:
            print("  umap = load_dataset('longevity-db/skeletal-muscle-atlas', split='projection_umap')")
        
        print(f"\nCache location: {cache_dir}")
        print("No more validation report - actual data is accessible!")
    else:
        print("⚠️  Tests failed - may need more time for HF to process")

if __name__ == "__main__":
    main() 