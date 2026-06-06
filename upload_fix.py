#!/usr/bin/env python3
"""
Upload the dataset loading script to fix HuggingFace dataset loading
"""

import os
import sys

from huggingface_hub import HfApi

repo_id = 'longevity-db/skeletal-muscle-atlas'

if not os.environ.get('HF_TOKEN'):
    print('Error: set HF_TOKEN environment variable', file=sys.stderr)
    sys.exit(1)

api = HfApi()

print('Uploading dataset loading script to fix load_dataset issue...')

try:
    api.upload_file(
        path_or_fileobj='skeletal_muscle_atlas.py',
        path_in_repo='skeletal_muscle_atlas.py',
        repo_id=repo_id,
        repo_type='dataset',
        commit_message='Add dataset loading script to fix load_dataset functionality'
    )
    print('Success: skeletal_muscle_atlas.py uploaded')
    print(f'Dataset: https://huggingface.co/datasets/{repo_id}')
except Exception as e:
    print(f'Upload failed: {e}')
    import traceback
    traceback.print_exc() 