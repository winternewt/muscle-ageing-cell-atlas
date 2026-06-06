#!/usr/bin/env python3
"""
Test the HuggingFace dataset configuration for skeletal muscle atlas
"""

import sys
from datasets import load_dataset

def test_config(config_name: str):
    """Test loading a specific configuration"""
    print(f"\n{'='*60}")
    print(f"TESTING CONFIG: {config_name}")
    print(f"{'='*60}")
    
    try:
        # For local testing, use the script path
        # For remote testing, use the repo name
        dataset = load_dataset(
            "longevity-db/skeletal-muscle-atlas",
            name=config_name,
            split="train[:10]"  # Load only first 10 samples for testing
        )
        
        print(f"✅ Config '{config_name}' loaded successfully!")
        print(f"Shape: {dataset.shape}")
        print(f"Features: {list(dataset.features.keys())}")
        
        # Show first example
        if len(dataset) > 0:
            first_example = dataset[0]
            print(f"First example keys: {list(first_example.keys())}")
            
            # Show a few values (truncated)
            for key, value in first_example.items():
                if isinstance(value, str):
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
                    
                if len(str(first_example)) > 500:  # Don't print too much
                    break
        
        return True
        
    except Exception as e:
        print(f"❌ Config '{config_name}' failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all configurations"""
    print("🧬 Testing HuggingFace Dataset Configurations")
    
    configs_to_test = [
        "sample_metadata",  # Start with metadata (smaller)
        "feature_metadata", 
        "projection_umap",
        "expression",       # Test expression last (largest)
        "all"
    ]
    
    results = {}
    
    for config in configs_to_test:
        success = test_config(config)
        results[config] = success
        
        if not success:
            print(f"⚠️  Stopping tests due to failure in {config}")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for config, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{config:<20} {status}")
    
    total_tested = len(results)
    total_passed = sum(results.values())
    print(f"\nTotal: {total_passed}/{total_tested} configurations working")
    
    if total_passed == total_tested:
        print("🎉 All tested configurations working!")
    else:
        print("⚠️  Some configurations need fixing")
        sys.exit(1)

if __name__ == "__main__":
    main() 