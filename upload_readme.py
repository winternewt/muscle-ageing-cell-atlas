#!/usr/bin/env python3
"""
Upload the updated README.md with proper YAML front matter for dataset configuration
"""

import os
import sys

from huggingface_hub import HfApi

repo_id = 'longevity-db/skeletal-muscle-atlas'

if not os.environ.get('HF_TOKEN'):
    print('Error: set HF_TOKEN environment variable', file=sys.stderr)
    sys.exit(1)

api = HfApi()

print('Uploading updated README.md with proper dataset configuration...')

try:
    api.upload_file(
        path_or_fileobj='README.md',
        path_in_repo='README.md',
        repo_id=repo_id,
        repo_type='dataset',
        commit_message='Update README with proper YAML front matter for dataset splits'
    )
    print('Success: README.md uploaded with dataset configuration')
    print(f'Dataset: https://huggingface.co/datasets/{repo_id}')
    print('This should fix the load_dataset issue by properly mapping files to splits')
except Exception as e:
    print(f'Upload failed: {e}')
    import traceback
    traceback.print_exc() 