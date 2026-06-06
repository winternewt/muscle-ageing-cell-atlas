#!/usr/bin/env python3
"""
Generate corrected YAML front matter for the README
"""

def generate_yaml_config():
    """Generate the corrected YAML configuration"""
    
    yaml_config = '''---
license: mit
tags:
- longevity
- aging
- skeletal-muscle
- human
- single-cell-rna-seq
- muscle-aging
- sarcopenia
- 10x-genomics
- muscle-stem-cells
- regenerative-medicine
pretty_name: "Human Skeletal Muscle Aging Atlas - 10X Chromium"
size_categories:
- 100K<n<1M
language:
- en

# Dataset configurations - separate parquet and json files
configs:
- config_name: default
  data_files:
    - split: expression
      path: "skeletal_muscle_10x_expression.parquet"
    - split: sample_metadata
      path: "skeletal_muscle_10x_sample_metadata.parquet"
    - split: feature_metadata
      path: "skeletal_muscle_10x_feature_metadata.parquet"
    - split: projection_pca
      path: "skeletal_muscle_10x_projection_X_pca.parquet"
    - split: projection_tsne
      path: "skeletal_muscle_10x_projection_X_tsne.parquet"
    - split: projection_umap
      path: "skeletal_muscle_10x_projection_X_umap.parquet"
    - split: projection_scvi
      path: "skeletal_muscle_10x_projection_X_scVI.parquet"

- config_name: metadata_json
  data_files:
    - split: unstructured_metadata
      path: "skeletal_muscle_10x_unstructured_metadata.json"
---'''
    
    return yaml_config

def update_readme():
    """Update README.md with corrected YAML front matter"""
    
    # Read current README
    with open('README.md', 'r') as f:
        content = f.read()
    
    # Find where the YAML front matter ends
    lines = content.split('\n')
    yaml_end_idx = -1
    
    for i, line in enumerate(lines):
        if i > 0 and line.strip() == '---':  # Second --- marks end of YAML
            yaml_end_idx = i
            break
    
    if yaml_end_idx == -1:
        print("Could not find end of YAML front matter")
        return False
    
    # Keep content after YAML
    content_after_yaml = '\n'.join(lines[yaml_end_idx + 1:])
    
    # Generate new YAML + content
    new_yaml = generate_yaml_config()
    new_content = new_yaml + '\n\n' + content_after_yaml
    
    # Write back
    with open('README.md', 'w') as f:
        f.write(new_content)
    
    print("✅ README.md updated with corrected YAML configuration")
    print("Changes made:")
    print("  - Separated parquet files (default config) from JSON files (metadata_json config)")
    print("  - This fixes the mixed file format issue")
    
    return True

def main():
    print("🔧 Fixing YAML Configuration")
    
    success = update_readme()
    
    if success:
        print("\nNext steps:")
        print("1. Upload the corrected README.md")
        print("2. Test with: load_dataset('longevity-db/skeletal-muscle-atlas', split='expression')")
        print("3. For JSON metadata: load_dataset('longevity-db/skeletal-muscle-atlas', name='metadata_json')")
    
    return success

if __name__ == "__main__":
    main() 