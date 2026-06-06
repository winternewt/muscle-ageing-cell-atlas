#!/usr/bin/env python3
"""
Test specific dataset configurations to see if the fix worked
"""

from datasets import load_dataset

def test_config(config_name):
    print(f"\n=== Testing config: {config_name} ===")
    try:
        dataset = load_dataset(
            "longevity-db/skeletal-muscle-atlas", 
            name=config_name,
            split="train[:5]"  # Just 5 samples for testing
        )
        print(f"Success: {config_name}")
        print(f"Shape: {dataset.shape}")
        print(f"Features: {list(dataset.features.keys())[:10]}")  # First 10 features
        return True
    except Exception as e:
        print(f"Failed: {config_name} - {e}")
        return False

def main():
    print("Testing new dataset configurations...")
    
    configs = [
        "sample_metadata",
        "feature_metadata", 
        "projection_umap"
    ]
    
    for config in configs:
        success = test_config(config)
        if success:
            print("New configuration is working")
            break
    else:
        print("New configurations not working yet - may need cache refresh")
        
    # Test default (should still show validation until cache refreshes)
    print("\n=== Testing default (no config) ===")
    try:
        dataset = load_dataset("longevity-db/skeletal-muscle-atlas")
        print(f"Default still loads: {list(dataset.keys())}")
    except Exception as e:
        print(f"Default failed: {e}")

if __name__ == "__main__":
    main() 