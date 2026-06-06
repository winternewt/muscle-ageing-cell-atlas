#!/usr/bin/env python3
"""
Quick validation - test only small files to verify YAML config works
"""

import os
from datasets import load_dataset, get_dataset_split_names

# Use /data/ for cache
os.environ['HF_DATASETS_CACHE'] = '/data/hf_cache'
os.environ['HF_HOME'] = '/data/hf_cache'

def quick_test():
    print("🚀 QUICK VALIDATION - Small Files Only")
    print("=" * 50)
    
    # First check if we can see the splits
    try:
        splits = get_dataset_split_names("longevity-db/skeletal-muscle-atlas")
        print(f"✅ Available splits: {splits}")
        
        if 'sample_metadata' in splits:
            print("✅ sample_metadata split found")
        if 'projection_umap' in splits:
            print("✅ projection_umap split found")
            
    except Exception as e:
        print(f"❌ Could not get splits: {e}")
        return False
    
    # Test 1: Load just metadata (small file)
    print("\n--- Testing sample_metadata ---")
    try:
        metadata = load_dataset(
            "longevity-db/skeletal-muscle-atlas", 
            split="sample_metadata[:3]"  # Just 3 rows
        )
        print(f"✅ Metadata loaded: {metadata.shape}")
        print(f"Columns: {list(metadata.column_names)[:5]}...")
        
        # Show actual data
        if len(metadata) > 0:
            sample = metadata[0]
            age_group = sample.get('Age_group', 'N/A')
            cell_type = sample.get('annotation_level0', 'N/A')
            print(f"Sample data: Age={age_group}, CellType={cell_type}")
            
        metadata_ok = True
    except Exception as e:
        print(f"❌ Metadata failed: {e}")
        metadata_ok = False
    
    # Test 2: Load UMAP (tiny file)
    print("\n--- Testing projection_umap ---")
    try:
        umap = load_dataset(
            "longevity-db/skeletal-muscle-atlas",
            split="projection_umap[:3]"  # Just 3 rows
        )
        print(f"✅ UMAP loaded: {umap.shape}")
        print(f"Columns: {list(umap.column_names)}")
        
        if len(umap) > 0:
            coords = umap[0]
            print(f"UMAP coordinates: {dict(coords)}")
            
        umap_ok = True
    except Exception as e:
        print(f"❌ UMAP failed: {e}")
        umap_ok = False
    
    # Results
    print("\n" + "=" * 50)
    print("QUICK VALIDATION RESULTS")
    print("=" * 50)
    
    if metadata_ok and umap_ok:
        print("🎉 SUCCESS: YAML configuration is working!")
        print("✅ Users can now load actual data instead of validation report")
        print("\nExample usage:")
        print("  metadata = load_dataset('longevity-db/skeletal-muscle-atlas', split='sample_metadata')")
        print("  umap = load_dataset('longevity-db/skeletal-muscle-atlas', split='projection_umap')")
        print("  # expression = load_dataset('longevity-db/skeletal-muscle-atlas', split='expression')  # Large file")
        
        return True
    else:
        print("⚠️  Partial success - some splits working")
        return False

if __name__ == "__main__":
    quick_test() 