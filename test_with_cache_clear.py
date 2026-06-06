#!/usr/bin/env python3
"""
Test dataset with cache cleared to use new YAML configuration
"""

import shutil
import os
from pathlib import Path
from datasets import load_dataset

def clear_dataset_cache():
    """Clear HuggingFace dataset cache for this specific dataset"""
    print("Clearing HuggingFace dataset cache...")
    
    # Find HF cache directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    dataset_cache_pattern = "longevity-db___skeletal-muscle-atlas"
    
    found_cache = False
    for cache_path in cache_dir.glob(f"*{dataset_cache_pattern}*"):
        print(f"Removing cached dataset: {cache_path}")
        shutil.rmtree(cache_path, ignore_errors=True)
        found_cache = True
    
    if not found_cache:
        print("No cached dataset found to clear")
    else:
        print("✅ Cache cleared successfully")

def test_with_force_redownload():
    """Test loading dataset with forced redownload"""
    print("\n=== Testing with forced redownload ===")
    
    try:
        # Force redownload to bypass cache
        dataset = load_dataset(
            "longevity-db/skeletal-muscle-atlas",
            download_mode="force_redownload",
            verification_mode="no_checks"
        )
        
        print("✅ Dataset loaded with force_redownload")
        print(f"Available splits: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {split_data.shape}")
            print(f"    Columns: {list(split_data.column_names)[:10]}")
        
        return True, dataset
        
    except Exception as e:
        print(f"❌ Force redownload failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_specific_split(split_name):
    """Test loading a specific split"""
    print(f"\n=== Testing split: {split_name} ===")
    
    try:
        dataset = load_dataset(
            "longevity-db/skeletal-muscle-atlas",
            split=f"{split_name}[:3]",  # Just 3 samples
            download_mode="force_redownload",
            verification_mode="no_checks"
        )
        
        print(f"✅ {split_name} loaded successfully")
        print(f"Shape: {dataset.shape}")
        print(f"Columns: {list(dataset.column_names)[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ {split_name} failed: {e}")
        return False

def main():
    print("🧬 Testing Dataset with Cache Clear")
    
    # Step 1: Clear cache
    clear_dataset_cache()
    
    # Step 2: Test with forced redownload
    success, dataset = test_with_force_redownload()
    
    if not success:
        print("❌ Could not load dataset even with force_redownload")
        print("The YAML configuration may need more time to process on HF servers")
        return
    
    # Step 3: Test specific splits if the default worked
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL SPLITS")
    print("="*60)
    
    splits_to_test = [
        "expression",
        "sample_metadata",
        "projection_umap"
    ]
    
    results = {}
    for split in splits_to_test:
        results[split] = test_specific_split(split)
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for split, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{split:<20} {status}")
    
    print(f"\nPassed: {passed}/{total} splits")
    
    if passed > 0:
        print("🎉 YAML configuration is working for some splits")
        print("Users can now access data via:")
        for split, success in results.items():
            if success:
                print(f"  load_dataset('longevity-db/skeletal-muscle-atlas', split='{split}')")
    else:
        print("⚠️  YAML configuration not yet active - may need more time to process")

if __name__ == "__main__":
    main() 