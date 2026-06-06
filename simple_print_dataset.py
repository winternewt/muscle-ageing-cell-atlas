#!/usr/bin/env python3
"""
Simple script to load and print dataset structure
"""

from datasets import load_dataset
import pandas as pd

print("Loading dataset: longevity-db/skeletal-muscle-atlas")
dataset = load_dataset('longevity-db/skeletal-muscle-atlas')

print(f"\n=== DATASET OBJECT ===")
print(f"Type: {type(dataset)}")
print(f"Keys: {list(dataset.keys())}")

for split_name, split_data in dataset.items():
    print(f"\n=== SPLIT: {split_name} ===")
    print(f"Type: {type(split_data)}")
    print(f"Shape: {split_data.shape}")
    print(f"Column names: {split_data.column_names}")
    
    # Convert to pandas and print first few rows
    df = split_data.to_pandas()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.to_string())
    
    print(f"\nColumn details:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype} - {type(df[col].iloc[0])}")
        if hasattr(df[col].iloc[0], '__len__') and not isinstance(df[col].iloc[0], str):
            print(f"    Content preview: {str(df[col].iloc[0])[:200]}...")
        else:
            print(f"    Content: {df[col].iloc[0]}") 