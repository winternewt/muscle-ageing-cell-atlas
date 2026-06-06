#!/usr/bin/env python3
"""
Test the new dataset splits configured via YAML front matter
"""

from datasets import load_dataset

def test_split(split_name):
    print(f"\n=== Testing split: {split_name} ===")
    try:
        dataset = load_dataset(
            "longevity-db/skeletal-muscle-atlas", 
            split=f"{split_name}[:5]"  # Load first 5 rows
        )
        print(f"✅ {split_name} loaded successfully")
        print(f"Shape: {dataset.shape}")
        print(f"Columns: {list(dataset.column_names)[:10]}")  # First 10 columns
        
        if len(dataset) > 0:
            print(f"Sample data: {list(dataset[0].keys())[:5]}")  # First 5 keys
        
        return True
    except Exception as e:
        print(f"❌ {split_name} failed: {e}")
        return False

def test_default():
    print(f"\n=== Testing default (no split specified) ===")
    try:
        dataset = load_dataset("longevity-db/skeletal-muscle-atlas")
        print(f"✅ Default loaded successfully")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Show info about each split
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {split_data.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Default failed: {e}")
        return False

def main():
    print("🧬 Testing Dataset Splits (YAML Configuration)")
    
    # Test the configured splits
    splits_to_test = [
        "expression",
        "sample_metadata", 
        "feature_metadata",
        "projection_umap",
        "projection_pca"
    ]
    
    results = {}
    
    for split in splits_to_test:
        success = test_split(split)
        results[split] = success
    
    # Test default behavior
    default_success = test_default()
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    
    for split, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{split:<20} {status}")
    
    print(f"{'default':<20} {'✅ PASS' if default_success else '❌ FAIL'}")
    
    total_passed = sum(results.values()) + (1 if default_success else 0)
    total_tests = len(results) + 1
    
    print(f"\nPassed: {total_passed}/{total_tests} tests")
    
    if total_passed == total_tests:
        print("🎉 Dataset configuration is working properly")
    else:
        print("⚠️  Some splits still need fixing or HF needs more time to process")

if __name__ == "__main__":
    main() 