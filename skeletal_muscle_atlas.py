"""
HuggingFace Dataset Loading Script for Skeletal Muscle Aging Atlas

This script defines how to load the skeletal muscle dataset parquet files
into a structured HuggingFace dataset with multiple configurations.
"""

import datasets
import pandas as pd
from typing import Dict, List, Any


# Dataset metadata
_CITATION = """
@article{kedlian2024human,
  title={Human skeletal muscle aging atlas},
  author={Kedlian, Veronika R and Wang, Yaning and Liu, Tianliang and Chen, Xiaoping and Bolt, Liam and Tudor, Catherine and Shen, Zhuojian and Fasouli, Eirini S and Prigmore, Elena and Kleshchevnikov, Vitalii and others},
  journal={Nature Aging},
  volume={4},
  pages={727--744},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s43587-024-00613-3},
  url={https://www.nature.com/articles/s43587-024-00613-3}
}
"""

_DESCRIPTION = """
A comprehensive single-cell RNA-seq atlas of human skeletal muscle aging across the lifespan (15-75 years).
This dataset provides 183,161 cells from 17 donors with gene expression, metadata, and pre-computed embeddings.
"""

_HOMEPAGE = "https://www.muscleageingcellatlas.org/"

_LICENSE = "MIT"

_URLS = {
    "expression": "skeletal_muscle_10x_expression.parquet",
    "sample_metadata": "skeletal_muscle_10x_sample_metadata.parquet", 
    "feature_metadata": "skeletal_muscle_10x_feature_metadata.parquet",
    "projection_pca": "skeletal_muscle_10x_projection_X_pca.parquet",
    "projection_umap": "skeletal_muscle_10x_projection_X_umap.parquet",
    "projection_tsne": "skeletal_muscle_10x_projection_X_tsne.parquet",
    "projection_scvi": "skeletal_muscle_10x_projection_X_scVI.parquet",
    "unstructured_metadata": "skeletal_muscle_10x_unstructured_metadata.json"
}


class SkeletalMuscleAtlasConfig(datasets.BuilderConfig):
    """Configuration for Skeletal Muscle Atlas dataset."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SkeletalMuscleAtlas(datasets.GeneratorBasedBuilder):
    """Skeletal Muscle Aging Atlas dataset."""

    BUILDER_CONFIGS = [
        SkeletalMuscleAtlasConfig(
            name="expression",
            version=datasets.Version("1.0.0"),
            description="Gene expression matrix (183,161 cells × 29,400 genes)",
        ),
        SkeletalMuscleAtlasConfig(
            name="sample_metadata", 
            version=datasets.Version("1.0.0"),
            description="Cell-level metadata (age, cell type, sex, etc.)",
        ),
        SkeletalMuscleAtlasConfig(
            name="feature_metadata",
            version=datasets.Version("1.0.0"), 
            description="Gene-level metadata (symbols, IDs, etc.)",
        ),
        SkeletalMuscleAtlasConfig(
            name="projection_pca",
            version=datasets.Version("1.0.0"),
            description="PCA embeddings (50 components)",
        ),
        SkeletalMuscleAtlasConfig(
            name="projection_umap",
            version=datasets.Version("1.0.0"),
            description="UMAP embeddings (2D visualization)",
        ),
        SkeletalMuscleAtlasConfig(
            name="projection_tsne", 
            version=datasets.Version("1.0.0"),
            description="t-SNE embeddings (2D visualization)",
        ),
        SkeletalMuscleAtlasConfig(
            name="projection_scvi",
            version=datasets.Version("1.0.0"),
            description="scVI embeddings (30D latent space)",
        ),
        SkeletalMuscleAtlasConfig(
            name="all",
            version=datasets.Version("1.0.0"),
            description="All data combined (expression + metadata + embeddings)",
        ),
    ]

    DEFAULT_CONFIG_NAME = "all"

    def _info(self):
        if self.config.name == "expression":
            # Dynamic features for expression matrix
            features = datasets.Features({
                "cell_id": datasets.Value("string"),
                **{f"gene_{i}": datasets.Value("float32") for i in range(29400)}
            })
        elif self.config.name == "sample_metadata":
            features = datasets.Features({
                "cell_id": datasets.Value("string"),
                "Age_group": datasets.Value("string"),
                "age_numeric": datasets.Value("float32"),
                "Sex": datasets.Value("string"),
                "annotation_level0": datasets.Value("string"),
                "annotation_level1": datasets.Value("string"),
                "annotation_level2": datasets.Value("string"),
                "DonorID": datasets.Value("string"),
                "batch": datasets.Value("string"),
                # Add other metadata columns as needed
            })
        elif self.config.name == "feature_metadata":
            features = datasets.Features({
                "gene_id": datasets.Value("string"),
                "SYMBOL": datasets.Value("string"),
                "ENSEMBL": datasets.Value("string"),
                "n_cells": datasets.Value("int32"),
            })
        elif self.config.name.startswith("projection_"):
            # Dynamic features for embeddings
            if self.config.name == "projection_pca":
                n_dims = 50
            elif self.config.name == "projection_scvi":
                n_dims = 30
            else:  # umap, tsne
                n_dims = 2
            
            features = datasets.Features({
                "cell_id": datasets.Value("string"),
                **{f"dim_{i}": datasets.Value("float32") for i in range(n_dims)}
            })
        else:  # "all" configuration
            features = datasets.Features({
                "cell_id": datasets.Value("string"),
                "data_type": datasets.Value("string"),
                "data": datasets.Value("string"),  # JSON string of the data
            })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Download and prepare the data."""
        
        # Download all files
        downloaded_files = dl_manager.download(_URLS)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files": downloaded_files,
                    "config_name": self.config.name,
                },
            ),
        ]

    def _generate_examples(self, files: Dict[str, str], config_name: str):
        """Generate examples based on the configuration."""
        
        if config_name == "expression":
            # Load expression matrix
            df = pd.read_parquet(files["expression"])
            
            for idx, (cell_id, row) in enumerate(df.iterrows()):
                yield idx, {
                    "cell_id": str(cell_id),
                    **{f"gene_{i}": float(row.iloc[i]) for i in range(len(row))}
                }
                
        elif config_name == "sample_metadata":
            # Load sample metadata
            df = pd.read_parquet(files["sample_metadata"])
            
            for idx, (cell_id, row) in enumerate(df.iterrows()):
                yield idx, {
                    "cell_id": str(cell_id),
                    "Age_group": str(row.get("Age_group", "")),
                    "age_numeric": float(row.get("age_numeric", 0.0)),
                    "Sex": str(row.get("Sex", "")),
                    "annotation_level0": str(row.get("annotation_level0", "")),
                    "annotation_level1": str(row.get("annotation_level1", "")),
                    "annotation_level2": str(row.get("annotation_level2", "")),
                    "DonorID": str(row.get("DonorID", "")),
                    "batch": str(row.get("batch", "")),
                }
                
        elif config_name == "feature_metadata":
            # Load feature metadata
            df = pd.read_parquet(files["feature_metadata"])
            
            for idx, (gene_id, row) in enumerate(df.iterrows()):
                yield idx, {
                    "gene_id": str(gene_id),
                    "SYMBOL": str(row.get("SYMBOL", "")),
                    "ENSEMBL": str(row.get("ENSEMBL", "")),
                    "n_cells": int(row.get("n_cells", 0)),
                }
                
        elif config_name.startswith("projection_"):
            # Load projection data
            projection_key = config_name  # e.g., "projection_pca"
            df = pd.read_parquet(files[projection_key])
            
            for idx, (cell_id, row) in enumerate(df.iterrows()):
                yield idx, {
                    "cell_id": str(cell_id),
                    **{f"dim_{i}": float(row.iloc[i]) for i in range(len(row))}
                }
                
        elif config_name == "all":
            # Load all data types and provide a unified interface
            sample_idx = 0
            
            # Expression data
            expr_df = pd.read_parquet(files["expression"])
            for cell_id, row in expr_df.iterrows():
                yield sample_idx, {
                    "cell_id": str(cell_id),
                    "data_type": "expression",
                    "data": row.to_json(),
                }
                sample_idx += 1
            
            # Sample metadata
            meta_df = pd.read_parquet(files["sample_metadata"])
            for cell_id, row in meta_df.iterrows():
                yield sample_idx, {
                    "cell_id": str(cell_id),
                    "data_type": "sample_metadata", 
                    "data": row.to_json(),
                }
                sample_idx += 1 