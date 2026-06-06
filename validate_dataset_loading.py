#!/usr/bin/env python3
"""
Validate HuggingFace Dataset Loading - Skeletal Muscle Atlas
============================================================

This script validates that the longevity-db/skeletal-muscle-atlas dataset
can be properly loaded using HuggingFace's load_dataset and checks for
common issues like phantom index columns and row name problems.
"""

import sys
import traceback
from typing import Dict, List, Any
import pandas as pd

def test_basic_loading():
    """Test basic dataset loading"""
    print("=" * 60)
    print("1. TESTING BASIC DATASET LOADING")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        
        print("Loading dataset: longevity-db/skeletal-muscle-atlas")
        dataset = load_dataset('longevity-db/skeletal-muscle-atlas')
        
        print("✅ Dataset loaded successfully!")
        print(f"Dataset keys: {list(dataset.keys())}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return None

def analyze_dataset_structure(dataset):
    """Analyze the structure of the loaded dataset"""
    print("\n" + "=" * 60)
    print("2. ANALYZING DATASET STRUCTURE")
    print("=" * 60)
    
    if dataset is None:
        print("❌ No dataset to analyze")
        return None
    
    results = {}
    
    for split_name, split_data in dataset.items():
        print(f"\n--- Split: {split_name} ---")
        print(f"Shape: {split_data.shape}")
        print(f"Number of columns: {len(split_data.column_names)}")
        print(f"First 10 columns: {split_data.column_names[:10]}")
        
        # Store results
        results[split_name] = {
            'shape': split_data.shape,
            'columns': split_data.column_names,
            'split_data': split_data
        }
    
    return results

def check_phantom_columns(dataset_results: Dict):
    """Check for phantom index columns"""
    print("\n" + "=" * 60)
    print("3. CHECKING FOR PHANTOM INDEX COLUMNS")
    print("=" * 60)
    
    issues_found = False
    
    for split_name, split_info in dataset_results.items():
        print(f"\n--- Split: {split_name} ---")
        columns = split_info['columns']
        
        # Check for common phantom column patterns
        phantom_patterns = [
            '__index_level_',
            'Unnamed:',
            'index',
            'level_0'
        ]
        
        phantom_cols = []
        for col in columns:
            for pattern in phantom_patterns:
                if pattern in str(col):
                    phantom_cols.append(col)
                    break
        
        if phantom_cols:
            print(f"⚠️  Found phantom columns: {phantom_cols}")
            issues_found = True
        else:
            print("✅ No phantom columns detected")
    
    return not issues_found

def test_pandas_conversion(dataset_results: Dict):
    """Test conversion to pandas and check for issues"""
    print("\n" + "=" * 60)
    print("4. TESTING PANDAS CONVERSION")
    print("=" * 60)
    
    conversion_results = {}
    
    for split_name, split_info in dataset_results.items():
        print(f"\n--- Split: {split_name} ---")
        
        try:
            # Convert to pandas
            df = split_info['split_data'].to_pandas()
            print(f"✅ Converted to pandas successfully")
            print(f"Pandas shape: {df.shape}")
            print(f"Index name: {df.index.name}")
            print(f"Index type: {type(df.index)}")
            
            # Check first few index values
            if len(df) > 0:
                print(f"First 5 index values: {df.index[:5].tolist()}")
                print(f"Index is unique: {df.index.is_unique}")
            
            # Check for issues with column names
            problematic_cols = [col for col in df.columns if 
                              '__index_level_' in str(col) or 
                              str(col).startswith('Unnamed:')]
            
            if problematic_cols:
                print(f"⚠️  Problematic columns in pandas: {problematic_cols}")
            else:
                print("✅ No problematic columns in pandas conversion")
            
            conversion_results[split_name] = {
                'success': True,
                'dataframe': df,
                'issues': problematic_cols
            }
            
        except Exception as e:
            print(f"❌ Error converting to pandas: {e}")
            conversion_results[split_name] = {
                'success': False,
                'error': str(e)
            }
    
    return conversion_results

def test_specific_files(dataset_results: Dict):
    """Test loading specific expected files"""
    print("\n" + "=" * 60)
    print("5. TESTING SPECIFIC FILE ACCESS")
    print("=" * 60)
    
    expected_files = [
        'skeletal_muscle_10x_expression.parquet',
        'skeletal_muscle_10x_sample_metadata.parquet',
        'skeletal_muscle_10x_feature_metadata.parquet',
        'skeletal_muscle_10x_projection_X_umap.parquet',
        'skeletal_muscle_10x_projection_X_scVI.parquet'
    ]
    
    for split_name, split_info in dataset_results.items():
        print(f"\n--- Split: {split_name} ---")
        columns = split_info['columns']
        
        # Try to identify which files are present based on column patterns
        print(f"Total columns: {len(columns)}")
        
        # Check if this looks like expression data (many gene columns)
        if len(columns) > 1000:
            print("✅ Likely contains expression matrix (many columns)")
        
        # Check for typical metadata columns
        metadata_indicators = ['Age_group', 'annotation_level0', 'Sex', 'DonorID']
        found_metadata = [col for col in columns if col in metadata_indicators]
        if found_metadata:
            print(f"✅ Found metadata columns: {found_metadata}")
        
        # Check for projection columns (typically fewer columns with numeric names)
        if len(columns) < 100 and all(str(col).replace('.', '').replace('-', '').isdigit() 
                                     or str(col).startswith('X') for col in columns[:10]):
            print("✅ Likely contains projection data")

def test_data_integrity(conversion_results: Dict):
    """Test data integrity and common issues"""
    print("\n" + "=" * 60)
    print("6. TESTING DATA INTEGRITY")
    print("=" * 60)
    
    for split_name, result in conversion_results.items():
        if result['success']:
            print(f"\n--- Split: {split_name} ---")
            df = result['dataframe']
            
            # Check for common data issues
            print(f"Shape: {df.shape}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
            
            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.any():
                print(f"⚠️  Columns with null values: {null_counts[null_counts > 0].head()}")
            else:
                print("✅ No null values found")
            
            # Check data types
            print(f"Data types: {df.dtypes.value_counts().to_dict()}")
            
            # Check for duplicate rows
            if df.duplicated().any():
                print(f"⚠️  Found {df.duplicated().sum()} duplicate rows")
            else:
                print("✅ No duplicate rows found")

def suggest_fixes(dataset_results: Dict, conversion_results: Dict):
    """Suggest fixes for common issues"""
    print("\n" + "=" * 60)
    print("7. SUGGESTED FIXES")
    print("=" * 60)
    
    issues_found = []
    fixes = []
    
    # Check for phantom columns
    for split_name, split_info in dataset_results.items():
        columns = split_info['columns']
        phantom_cols = [col for col in columns if '__index_level_' in str(col)]
        
        if phantom_cols:
            issues_found.append(f"Phantom columns in {split_name}: {phantom_cols}")
            fixes.append(f"""
# Fix phantom columns in {split_name}:
df = df.drop(columns={phantom_cols})
# or when saving:
df.to_parquet('file.parquet', index=False)
""")
    
    # Check for index issues
    for split_name, result in conversion_results.items():
        if result['success']:
            df = result['dataframe']
            if df.index.name is None and not df.index.equals(pd.RangeIndex(len(df))):
                issues_found.append(f"Unnamed meaningful index in {split_name}")
                fixes.append(f"""
# Fix unnamed index in {split_name}:
df.index.name = 'cell_id'  # or appropriate name
df.reset_index(inplace=True)  # if you want index as column
""")
    
    if issues_found:
        print("Issues found:")
        for issue in issues_found:
            print(f"  - {issue}")
        
        print("\nSuggested fixes:")
        for fix in fixes:
            print(fix)
    else:
        print("✅ No major issues detected!")

def main():
    """Main validation function"""
    print("🧬 HuggingFace Dataset Validation - Skeletal Muscle Atlas")
    print("=" * 60)
    
    # Test basic loading
    dataset = test_basic_loading()
    if dataset is None:
        print("\n❌ Cannot proceed with validation - dataset failed to load")
        sys.exit(1)
    
    # Analyze structure
    dataset_results = analyze_dataset_structure(dataset)
    if dataset_results is None:
        print("\n❌ Cannot proceed with validation - failed to analyze structure")
        sys.exit(1)
    
    # Check for phantom columns
    no_phantom_issues = check_phantom_columns(dataset_results)
    
    # Test pandas conversion
    conversion_results = test_pandas_conversion(dataset_results)
    
    # Test specific file access patterns
    test_specific_files(dataset_results)
    
    # Test data integrity
    test_data_integrity(conversion_results)
    
    # Suggest fixes
    suggest_fixes(dataset_results, conversion_results)
    
    # Final summary
    print("\n" + "=" * 60)
    print("8. VALIDATION SUMMARY")
    print("=" * 60)
    
    all_success = all(result.get('success', False) for result in conversion_results.values())
    
    if all_success and no_phantom_issues:
        print("🎉 VALIDATION PASSED - Dataset loads correctly!")
    else:
        print("⚠️  VALIDATION ISSUES DETECTED - See details above")
    
    print(f"\nDataset splits validated: {len(dataset_results)}")
    print(f"Successful pandas conversions: {sum(1 for r in conversion_results.values() if r.get('success', False))}")

if __name__ == "__main__":
    main() 