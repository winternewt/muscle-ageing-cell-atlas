#!/usr/bin/env python3
"""
Strip embedded images from Jupyter notebooks to make them readable.
This removes base64 encoded images while preserving code and markdown content.
"""

import json
import sys
from pathlib import Path

def strip_notebook_images(input_path, output_path):
    """Remove embedded images from a Jupyter notebook."""
    
    print(f"🔧 Stripping images from {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Track removed content
    removed_images = 0
    
    # Process each cell
    for cell in notebook.get('cells', []):
        # Check outputs for embedded images
        if 'outputs' in cell:
            for output in cell['outputs']:
                if 'data' in output:
                    # Remove common image formats
                    image_types = ['image/png', 'image/jpeg', 'image/svg+xml']
                    for img_type in image_types:
                        if img_type in output['data']:
                            output['data'][img_type] = "[IMAGE REMOVED]"
                            removed_images += 1
                
                # Also check for matplotlib/plotly outputs
                if 'text/html' in output.get('data', {}):
                    html_content = output['data']['text/html']
                    if isinstance(html_content, list):
                        html_content = ''.join(html_content)
                    if len(html_content) > 10000:  # Large HTML likely contains embedded images
                        output['data']['text/html'] = "[LARGE HTML OUTPUT REMOVED]"
                        removed_images += 1
    
    # Save cleaned notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Cleaned notebook saved to {output_path}")
    print(f"   Removed {removed_images} image/large outputs")
    
    # Show size reduction
    original_size = Path(input_path).stat().st_size / (1024 * 1024)  # MB
    new_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"   Size: {original_size:.1f}MB → {new_size:.1f}MB ({100 * (1 - new_size/original_size):.1f}% reduction)")

if __name__ == "__main__":
    # Target notebooks
    notebooks = [
        "SKM_ageing_atlas/Annotation/human_SKM_v2.0_single-cell_v1.ipynb",
        "SKM_ageing_atlas/Figures/Figure1/Human_SKM_single-cell2nuclei.ipynb"
    ]
    
    for notebook_path in notebooks:
        if Path(notebook_path).exists():
            output_path = f"processed/{Path(notebook_path).stem}_clean.ipynb"
            strip_notebook_images(notebook_path, output_path)
        else:
            print(f"❌ Notebook not found: {notebook_path}") 