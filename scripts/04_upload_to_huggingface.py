#!/usr/bin/env python3
"""
Selective HuggingFace Dataset Upload Script

This script uploads only the essential files for the skeletal muscle dataset:
- Processed parquet files (data)
- JSON metadata files (including unstructured metadata and exploration results)
- README.md
- LICENSE
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from typing import List, Optional
import typer

app = typer.Typer()

def get_files_to_upload(base_path: Path) -> List[tuple[Path, str]]:
    """
    Get list of (local_path, repo_path) tuples for files to upload
    """
    files_to_upload = []
    
    # Processed data files
    processed_dir = base_path / "processed"
    
    # Core dataset files (required by BEST_SCORING.md)
    core_patterns = [
        "skeletal_muscle_10x_expression.parquet",
        "skeletal_muscle_10x_sample_metadata.parquet", 
        "skeletal_muscle_10x_feature_metadata.parquet",
        "skeletal_muscle_10x_projection_X_pca.parquet",
        "skeletal_muscle_10x_projection_X_tsne.parquet",
        "skeletal_muscle_10x_projection_X_umap.parquet",
        "skeletal_muscle_10x_projection_X_scVI.parquet",
        "skeletal_muscle_10x_unstructured_metadata.json"
    ]
    
    # Additional metadata files for completeness
    additional_patterns = [
        "detailed_structure.json", 
        "validation_report.json"
    ]
    
    # Add core required files
    for pattern in core_patterns:
        file_path = processed_dir / pattern
        if file_path.exists():
            files_to_upload.append((file_path, file_path.name))
        else:
            print(f"⚠️  Missing required file: {pattern}")
    
    # Add additional metadata files
    for pattern in additional_patterns:
        file_path = processed_dir / pattern
        if file_path.exists():
            files_to_upload.append((file_path, file_path.name))
    
    # Documentation files
    doc_files = ["README.md", "LICENSE"]
    for doc_file in doc_files:
        file_path = base_path / doc_file
        if file_path.exists():
            files_to_upload.append((file_path, doc_file))
        else:
            print(f"⚠️  Missing documentation file: {doc_file}")
    
    return files_to_upload

@app.command()
def upload_dataset(
    repo_id: str = typer.Argument(..., help="HuggingFace repo ID (e.g., 'longevity-db/skeletal-muscle-atlas')"),
    token: Optional[str] = typer.Option(None, help="HuggingFace token (or set HF_TOKEN env var)"),
    base_path: str = typer.Option(".", help="Base path of the project"),
    dry_run: bool = typer.Option(False, help="Show what would be uploaded without doing it"),
    create_if_not_exists: bool = typer.Option(True, help="Create repo if it doesn't exist")
):
    """
    Upload skeletal muscle dataset to HuggingFace Hub
    """
    base_path = Path(base_path).resolve()
    
    # Get files to upload
    files_to_upload = get_files_to_upload(base_path)
    
    print(f"📁 Base path: {base_path}")
    print(f"🎯 Repository: {repo_id}")
    print(f"📊 Files to upload: {len(files_to_upload)}")
    print()
    
    # Show files categorized by type
    core_files = []
    metadata_files = []
    doc_files = []
    
    for local_path, repo_path in files_to_upload:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        
        if repo_path.endswith('.parquet'):
            core_files.append((repo_path, size_mb))
        elif repo_path.endswith('.json'):
            metadata_files.append((repo_path, size_mb))
        else:
            doc_files.append((repo_path, size_mb))
    
    # Display file inventory
    print("🗂️  **CORE DATASET FILES**:")
    total_core_size = 0
    for filename, size_mb in core_files:
        total_core_size += size_mb
        print(f"  📄 {filename:<55} ({size_mb:>8.1f} MB)")
    
    print(f"\n📊 **METADATA FILES**:")
    total_meta_size = 0
    for filename, size_mb in metadata_files:
        total_meta_size += size_mb
        print(f"  📄 {filename:<55} ({size_mb:>8.1f} MB)")
    
    print(f"\n📚 **DOCUMENTATION FILES**:")
    total_doc_size = 0
    for filename, size_mb in doc_files:
        total_doc_size += size_mb
        print(f"  📄 {filename:<55} ({size_mb:>8.1f} MB)")
    
    total_size = total_core_size + total_meta_size + total_doc_size
    print(f"\n💾 **TOTAL SIZE**: {total_size:.1f} MB")
    print(f"   ├── Core data: {total_core_size:.1f} MB")
    print(f"   ├── Metadata: {total_meta_size:.1f} MB")
    print(f"   └── Documentation: {total_doc_size:.1f} MB")
    
    if dry_run:
        print("\n🔍 DRY RUN - No files uploaded")
        return
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Create repo if needed
    if create_if_not_exists:
        try:
            create_repo(
                repo_id=repo_id,
                token=token,
                repo_type="dataset",
                exist_ok=True
            )
            print(f"✅ Repository {repo_id} ready")
        except Exception as e:
            print(f"⚠️  Repo creation warning: {e}")
    
    # Upload files
    print(f"\n🚀 Uploading files...")
    
    for local_path, repo_path in files_to_upload:
        try:
            print(f"  📤 Uploading {repo_path}...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"  ✅ {repo_path} uploaded successfully")
        except Exception as e:
            print(f"  ❌ Failed to upload {repo_path}: {e}")
    
    print(f"\n🎉 Upload complete! View at: https://huggingface.co/datasets/{repo_id}")

@app.command()
def list_files(
    base_path: str = typer.Option(".", help="Base path of the project")
):
    """
    List files that would be uploaded (dry run)
    """
    base_path = Path(base_path).resolve()
    files_to_upload = get_files_to_upload(base_path)
    
    print(f"Files to upload from {base_path}:")
    for local_path, repo_path in files_to_upload:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  {repo_path:<50} ({size_mb:>8.1f} MB)")

@app.command()
def check_completeness():
    """
    Check dataset completeness for upload readiness
    """
    print("🏆 **DATASET COMPLETENESS CHECK**\n")
    
    base_path = Path(".").resolve()
    processed_dir = base_path / "processed"
    
    # Required files for complete dataset
    required_files = {
        "skeletal_muscle_10x_expression.parquet": "Main X matrix",
        "skeletal_muscle_10x_sample_metadata.parquet": "Per-sample metadata",
        "skeletal_muscle_10x_feature_metadata.parquet": "Per-feature metadata", 
        "skeletal_muscle_10x_projection_X_pca.parquet": "PCA embeddings",
        "skeletal_muscle_10x_projection_X_tsne.parquet": "t-SNE visualization",
        "skeletal_muscle_10x_projection_X_umap.parquet": "UMAP visualization",
        "skeletal_muscle_10x_unstructured_metadata.json": "Additional metadata"
    }
    
    documentation_files = {
        "README.md": "Dataset card"
    }
    
    print("📊 **REQUIRED DATA FILES**:")
    missing_core = []
    for filename, description in required_files.items():
        file_path = processed_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename:<55} ({size_mb:>8.1f} MB) - {description}")
        else:
            print(f"  ❌ {filename:<55} MISSING - {description}")
            missing_core.append(filename)
    
    print(f"\n📚 **DOCUMENTATION FILES**:")
    missing_docs = []
    for filename, description in documentation_files.items():
        file_path = base_path / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename:<55} ({size_mb:>8.1f} MB) - {description}")
        else:
            print(f"  ❌ {filename:<55} MISSING - {description}")
            missing_docs.append(filename)
    
    # Summary
    total_required = len(required_files) + len(documentation_files)
    total_missing = len(missing_core) + len(missing_docs)
    completion_rate = ((total_required - total_missing) / total_required) * 100
    
    print(f"\n🎯 **COMPLETION SUMMARY**:")
    print(f"   📊 Required files: {total_required}")
    print(f"   ✅ Present: {total_required - total_missing}")
    print(f"   ❌ Missing: {total_missing}")
    print(f"   📈 Completion rate: {completion_rate:.1f}%")
    
    if total_missing == 0:
        print(f"\n🏆 **UPLOAD READY!** All required files present.")
    else:
        print(f"\n⚠️  **MISSING FILES** - Upload readiness: {completion_rate:.1f}%")
        if missing_core:
            print(f"     Missing core files: {', '.join(missing_core)}")
        if missing_docs:
            print(f"     Missing documentation: {', '.join(missing_docs)}")

if __name__ == "__main__":
    app() 