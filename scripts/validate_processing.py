#!/usr/bin/env python3
"""
Comprehensive Dataset Validation for Human Skeletal Muscle Aging Atlas
=====================================================================

Unified validation script that:
- Validates all processed files with memory-safe approaches
- Performs comprehensive data integrity checks
- Generates UMAP/t-SNE visualizations for validation
- Provides detailed reporting and logging

Usage: python3 scripts/validate_processing.py
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveDatasetValidator:
    """Unified comprehensive validator for processed dataset files"""
    
    def __init__(self, processed_dir: str = "processed", generate_visualizations: bool = True):
        self.processed_dir = Path(processed_dir)
        self.method = "10x"
        self.enable_visualizations = generate_visualizations
        self.expected_files = {
            'expression': f'skeletal_muscle_{self.method}_expression.parquet',
            'sample_metadata': f'skeletal_muscle_{self.method}_sample_metadata.parquet',
            'feature_metadata': f'skeletal_muscle_{self.method}_feature_metadata.parquet',
            'projection_scVI': f'skeletal_muscle_{self.method}_projection_X_scVI.parquet',
            'projection_umap': f'skeletal_muscle_{self.method}_projection_X_umap.parquet',
            'projection_pca': f'skeletal_muscle_{self.method}_projection_X_pca.parquet',
            'projection_tsne': f'skeletal_muscle_{self.method}_projection_X_tsne.parquet',
            'unstructured': f'skeletal_muscle_{self.method}_unstructured_metadata.json',
            'summary': 'phase2_processing_summary.json'
        }
        self.validation_results = {}
        self.memory_stats = []
        
        # Create visualizations directory
        self.viz_dir = self.processed_dir / "validation_visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def log_memory(self, stage: str):
        """Log current memory usage"""
        mem = self.get_memory_usage()
        self.memory_stats.append({"stage": stage, "memory_mb": mem})
        logger.info(f"💾 Memory at {stage}: {mem:.1f}MB")
        
    def log_status(self, test_name: str, status: bool, message: str = ""):
        """Log test status with emoji indicators"""
        emoji = "✅" if status else "❌"
        self.validation_results[test_name] = {"status": bool(status), "message": message}
        logger.info(f"{emoji} {test_name}: {message}")
        
    def validate_file_existence(self) -> bool:
        """Check all expected files exist"""
        logger.info("🔍 Validating File Existence")
        all_exist = True
        
        for file_type, filename in self.expected_files.items():
            file_path = self.processed_dir / filename
            exists = file_path.exists()
            
            if exists:
                size_mb = file_path.stat().st_size / (1024**2)
                self.log_status(f"File exists: {filename}", True, f"{size_mb:.1f}MB")
            else:
                self.log_status(f"File exists: {filename}", False, "MISSING")
                all_exist = False
                
        self.log_memory("file_existence")
        return all_exist
    
    def validate_expression_matrix(self) -> bool:
        """Validate expression matrix file with MEMORY-EFFICIENT loading tests"""
        logger.info("🧬 Validating Expression Matrix (Memory-Safe)")
        
        try:
            file_path = self.processed_dir / self.expected_files['expression']
            
            # Schema validation without loading data
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(file_path)
            
            # Get metadata
            schema = parquet_file.schema_arrow
            num_rows = parquet_file.metadata.num_rows
            num_cols = len(schema)
            
            # Validate dimensions
            expected_rows, expected_cols = 183161, 29400  # Fixed: was 29401 due to pandas __index_level_0__ bug
            row_count_correct = num_rows == expected_rows
            col_count_correct = num_cols == expected_cols
            
            self.log_status("Expression dimensions", 
                          row_count_correct and col_count_correct,
                          f"Shape: {num_rows:,} × {num_cols:,}")
            
            # Small sample loading test
            logger.info("  Testing small sample loading (100 cells)...")
            table_sample = parquet_file.read_row_group(0, columns=None)
            expr_sample = table_sample.to_pandas().head(100)
            
            # Validate data types
            dtype_check = expr_sample.dtypes.iloc[0] == 'float32'
            self.log_status("Expression dtype", dtype_check, f"Type: {expr_sample.dtypes.iloc[0]}")
            
            # Value range validation
            sample_values = expr_sample.iloc[0, :10].values
            reasonable_range = np.all((sample_values >= 0) & (sample_values <= 50))
            self.log_status("Expression values", reasonable_range, 
                          f"Sample range: {sample_values.min():.2f}-{sample_values.max():.2f}")
            
            # Index validation
            index_valid = expr_sample.index.is_unique
            self.log_status("Expression index", index_valid, f"Unique: {index_valid}")
            
            # Cleanup
            del parquet_file, table_sample, expr_sample
            self.log_memory("expression_validation")
            
            return row_count_correct and col_count_correct and dtype_check and reasonable_range and index_valid
            
        except Exception as e:
            self.log_status("Expression validation", False, f"Error: {e}")
            return False
    
    def validate_sample_metadata(self) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Validate sample metadata and return it for further use"""
        logger.info("👥 Validating Sample Metadata")
        
        try:
            file_path = self.processed_dir / self.expected_files['sample_metadata']
            metadata = pd.read_parquet(file_path)
            
            # Dimension check
            expected_rows = 183161
            actual_rows = len(metadata)
            dimension_check = actual_rows == expected_rows
            self.log_status("Sample metadata rows", dimension_check, f"{actual_rows:,} cells")
            
            # Required columns
            required_cols = ['Age_group', 'annotation_level0', 'Sex', 'DonorID']
            missing_cols = [col for col in required_cols if col not in metadata.columns]
            cols_present = len(missing_cols) == 0
            
            if cols_present:
                self.log_status("Required columns", True, f"All {len(required_cols)} present")
            else:
                self.log_status("Required columns", False, f"Missing: {missing_cols}")
            
            # Age validation
            age_groups = metadata['Age_group'].nunique()
            age_check = 6 <= age_groups <= 10  # Flexible range
            self.log_status("Age groups", age_check, f"{age_groups} unique groups")
            
            # Cell type validation
            cell_types = metadata['annotation_level0'].nunique()
            cell_check = cell_types > 20  # Should have many cell types
            self.log_status("Cell types", cell_check, f"{cell_types} unique types")
            
            # Donor validation
            donors = metadata['DonorID'].nunique()
            donor_check = donors >= 15  # Multiple donors
            self.log_status("Donors", donor_check, f"{donors} unique donors")
            
            self.log_memory("sample_metadata")
            
            success = dimension_check and cols_present and age_check and cell_check and donor_check
            return success, metadata if success else None
            
        except Exception as e:
            self.log_status("Sample metadata", False, f"Error: {e}")
            return False, None
    
    def validate_feature_metadata(self) -> bool:
        """Validate feature metadata"""
        logger.info("🧬 Validating Feature Metadata")
        
        try:
            file_path = self.processed_dir / self.expected_files['feature_metadata']
            features = pd.read_parquet(file_path)
            
            # Dimension check
            expected_rows = 29400  # Fixed: was 29401 due to pandas __index_level_0__ bug
            actual_rows = len(features)
            dimension_check = actual_rows == expected_rows
            self.log_status("Feature metadata rows", dimension_check, f"{actual_rows:,} genes")
            
            # Gene annotation coverage
            symbol_coverage = features['SYMBOL'].notna().mean()
            ensembl_coverage = features['ENSEMBL'].notna().mean()
            
            symbol_check = symbol_coverage > 0.95
            ensembl_check = ensembl_coverage > 0.95
            
            self.log_status("Gene symbols", symbol_check, f"{symbol_coverage:.1%} coverage")
            self.log_status("ENSEMBL IDs", ensembl_check, f"{ensembl_coverage:.1%} coverage")
            
            del features
            self.log_memory("feature_metadata")
            
            return dimension_check and symbol_check and ensembl_check
            
        except Exception as e:
            self.log_status("Feature metadata", False, f"Error: {e}")
            return False
    
    def validate_projections(self) -> Tuple[bool, Dict[str, pd.DataFrame]]:
        """Validate all projection files and return ONLY small ones for visualization"""
        logger.info("📊 Validating Projections (Memory-Safe)")
        
        projections = {}
        all_valid = True
        
        projection_specs = {
            'scVI': (30, 'scVI embedding'),
            'umap': (2, 'UMAP projection'),  
            'pca': (50, 'PCA projection'),
            'tsne': (2, 't-SNE projection')
        }
        
        for proj_name, (expected_dims, description) in projection_specs.items():
            try:
                file_key = f'projection_{proj_name}'
                file_path = self.processed_dir / self.expected_files[file_key]
                
                # MEMORY-SAFE: Load projection file
                projection = pd.read_parquet(file_path)
                
                # Dimension validation
                expected_rows = 183161
                actual_rows, actual_cols = projection.shape
                
                row_check = actual_rows == expected_rows
                col_check = actual_cols == expected_dims
                
                self.log_status(f"{description} shape", 
                              row_check and col_check,
                              f"{actual_rows:,} × {actual_cols}")
                
                # Value range validation on SMALL sample to avoid memory issues
                sample_values = projection.iloc[:100, :].values  # Only first 100 rows
                finite_check = np.all(np.isfinite(sample_values))
                self.log_status(f"{description} values", finite_check, 
                              f"Sample range: {sample_values.min():.2f} to {sample_values.max():.2f}")
                
                # CRITICAL: Only keep small projections for visualization (UMAP, t-SNE)
                # DO NOT keep large projections (PCA, scVI) in memory!
                if row_check and col_check and finite_check:
                    if proj_name in ['umap', 'tsne']:  # Only small 2D projections
                        projections[proj_name] = projection
                        logger.info(f"  Keeping {proj_name} in memory for visualization")
                    else:
                        # Free large projections immediately
                        del projection
                        logger.info(f"  Validated {proj_name} but freed from memory (too large)")
                else:
                    all_valid = False
                    del projection  # Free memory even on failure
                    
            except Exception as e:
                self.log_status(f"{description}", False, f"Error: {e}")
                all_valid = False
        
        self.log_memory("projections")
        return all_valid, projections
    
    def generate_visualizations(self, metadata: pd.DataFrame, projections: Dict[str, pd.DataFrame]):
        """Generate UMAP and t-SNE visualizations for validation"""
        if not self.enable_visualizations:
            logger.info("📊 Visualization generation disabled")
            return
            
        logger.info("🖼️  Generating Validation Visualizations")
        
        # Set up matplotlib for high-quality plots
        plt.style.use('default')
        
        # UMAP visualizations
        if 'umap' in projections:
            self._create_umap_plots(metadata, projections['umap'])
            
        # t-SNE visualizations  
        if 'tsne' in projections:
            self._create_tsne_plots(metadata, projections['tsne'])
            
        self.log_status("Visualizations generated", True, f"Saved to {self.viz_dir}")
        self.log_memory("visualizations")
    
    def _create_umap_plots(self, metadata: pd.DataFrame, umap_data: pd.DataFrame):
        """Create UMAP plots colored by different attributes"""
        logger.info("  Creating UMAP visualizations...")
        
        # Combine data
        plot_data = pd.concat([umap_data, metadata[['Age_group', 'annotation_level0', 'Sex', 'DonorID']]], axis=1)
        plot_data.columns = ['UMAP1', 'UMAP2', 'Age_group', 'Cell_type', 'Sex', 'DonorID']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('UMAP Projections - Validation Plots', fontsize=16, fontweight='bold')
        
        # Age groups
        age_colors = plt.cm.viridis(np.linspace(0, 1, plot_data['Age_group'].nunique()))
        for i, age_group in enumerate(sorted(plot_data['Age_group'].unique())):
            mask = plot_data['Age_group'] == age_group
            axes[0,0].scatter(plot_data.loc[mask, 'UMAP1'], plot_data.loc[mask, 'UMAP2'],
                            c=[age_colors[i]], label=age_group, alpha=0.6, s=1)
        axes[0,0].set_title('Colored by Age Group')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].set_xlabel('UMAP1')
        axes[0,0].set_ylabel('UMAP2')
        
        # Cell types (top 10 most abundant)
        top_cell_types = plot_data['Cell_type'].value_counts().head(10).index
        cell_colors = plt.cm.tab20(np.linspace(0, 1, len(top_cell_types)))
        for i, cell_type in enumerate(top_cell_types):
            mask = plot_data['Cell_type'] == cell_type
            axes[0,1].scatter(plot_data.loc[mask, 'UMAP1'], plot_data.loc[mask, 'UMAP2'],
                            c=[cell_colors[i]], label=cell_type, alpha=0.6, s=1)
        axes[0,1].set_title('Colored by Cell Type (Top 10)')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,1].set_xlabel('UMAP1')
        axes[0,1].set_ylabel('UMAP2')
        
        # Sex
        sex_colors = {'M': 'blue', 'F': 'red'}
        for sex in plot_data['Sex'].unique():
            if pd.notna(sex):
                mask = plot_data['Sex'] == sex
                axes[1,0].scatter(plot_data.loc[mask, 'UMAP1'], plot_data.loc[mask, 'UMAP2'],
                                c=sex_colors.get(sex, 'gray'), label=sex, alpha=0.6, s=1)
        axes[1,0].set_title('Colored by Sex')
        axes[1,0].legend()
        axes[1,0].set_xlabel('UMAP1')
        axes[1,0].set_ylabel('UMAP2')
        
        # Density plot
        axes[1,1].hexbin(plot_data['UMAP1'], plot_data['UMAP2'], gridsize=50, cmap='Blues')
        axes[1,1].set_title('Cell Density')
        axes[1,1].set_xlabel('UMAP1')
        axes[1,1].set_ylabel('UMAP2')
        
        plt.tight_layout()
        umap_path = self.viz_dir / "umap_validation_plots.png"
        plt.savefig(umap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  UMAP plots saved to {umap_path}")
    
    def _create_tsne_plots(self, metadata: pd.DataFrame, tsne_data: pd.DataFrame):
        """Create t-SNE plots colored by different attributes"""
        logger.info("  Creating t-SNE visualizations...")
        
        # Combine data
        plot_data = pd.concat([tsne_data, metadata[['Age_group', 'annotation_level0', 'Sex', 'DonorID']]], axis=1)
        plot_data.columns = ['tSNE1', 'tSNE2', 'Age_group', 'Cell_type', 'Sex', 'DonorID']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('t-SNE Projections - Validation Plots', fontsize=16, fontweight='bold')
        
        # Age groups
        age_colors = plt.cm.viridis(np.linspace(0, 1, plot_data['Age_group'].nunique()))
        for i, age_group in enumerate(sorted(plot_data['Age_group'].unique())):
            mask = plot_data['Age_group'] == age_group
            axes[0,0].scatter(plot_data.loc[mask, 'tSNE1'], plot_data.loc[mask, 'tSNE2'],
                            c=[age_colors[i]], label=age_group, alpha=0.6, s=1)
        axes[0,0].set_title('Colored by Age Group')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].set_xlabel('t-SNE1')
        axes[0,0].set_ylabel('t-SNE2')
        
        # Cell types (top 10 most abundant)
        top_cell_types = plot_data['Cell_type'].value_counts().head(10).index
        cell_colors = plt.cm.tab20(np.linspace(0, 1, len(top_cell_types)))
        for i, cell_type in enumerate(top_cell_types):
            mask = plot_data['Cell_type'] == cell_type
            axes[0,1].scatter(plot_data.loc[mask, 'tSNE1'], plot_data.loc[mask, 'tSNE2'],
                            c=[cell_colors[i]], label=cell_type, alpha=0.6, s=1)
        axes[0,1].set_title('Colored by Cell Type (Top 10)')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,1].set_xlabel('t-SNE1')
        axes[0,1].set_ylabel('t-SNE2')
        
        # Sex
        sex_colors = {'M': 'blue', 'F': 'red'}
        for sex in plot_data['Sex'].unique():
            if pd.notna(sex):
                mask = plot_data['Sex'] == sex
                axes[1,0].scatter(plot_data.loc[mask, 'tSNE1'], plot_data.loc[mask, 'tSNE2'],
                                c=sex_colors.get(sex, 'gray'), label=sex, alpha=0.6, s=1)
        axes[1,0].set_title('Colored by Sex')
        axes[1,0].legend()
        axes[1,0].set_xlabel('t-SNE1')
        axes[1,0].set_ylabel('t-SNE2')
        
        # Density plot
        axes[1,1].hexbin(plot_data['tSNE1'], plot_data['tSNE2'], gridsize=50, cmap='Reds')
        axes[1,1].set_title('Cell Density')
        axes[1,1].set_xlabel('t-SNE1')
        axes[1,1].set_ylabel('t-SNE2')
        
        plt.tight_layout()
        tsne_path = self.viz_dir / "tsne_validation_plots.png"
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  t-SNE plots saved to {tsne_path}")
    
    def validate_unstructured_metadata(self) -> bool:
        """Validate unstructured metadata JSON"""
        logger.info("📄 Validating Unstructured Metadata")
        
        try:
            file_path = self.processed_dir / self.expected_files['unstructured']
            with open(file_path, 'r') as f:
                unstructured = json.load(f)
            
            # Check expected keys
            expected_keys = ['pca', 'neighbors', 'tsne']
            present_keys = [key for key in expected_keys if key in unstructured]
            
            keys_check = len(present_keys) >= 2  # At least 2 expected keys
            self.log_status("Unstructured keys", keys_check, f"{len(present_keys)}/{len(expected_keys)} keys present")
            
            # Check JSON validity (already loaded successfully)
            json_valid = True
            self.log_status("JSON format", json_valid, "Valid JSON structure")
            
            self.log_memory("unstructured_metadata")
            return keys_check and json_valid
            
        except Exception as e:
            self.log_status("Unstructured metadata", False, f"Error: {e}")
            return False
    
    def validate_integration_tests(self, metadata: pd.DataFrame, projections: Dict[str, pd.DataFrame]) -> bool:
        """Test cross-file consistency and integration"""
        logger.info("🔗 Validating Cross-File Integration")
        
        try:
            # Index consistency across files
            base_index = metadata.index
            
            index_consistency = True
            for proj_name, proj_data in projections.items():
                if not base_index.equals(proj_data.index):
                    self.log_status(f"Index consistency {proj_name}", False, "Index mismatch")
                    index_consistency = False
                    
            if index_consistency:
                self.log_status("Index consistency", True, "All indices match")
            
            # Sample a subset for integration test
            sample_size = min(1000, len(metadata))
            sample_idx = np.random.choice(metadata.index, sample_size, replace=False)
            
            # Test metadata-projection alignment
            test_metadata = metadata.loc[sample_idx]
            test_umap = projections.get('umap')
            
            if test_umap is not None:
                test_umap_subset = test_umap.loc[sample_idx]
                alignment_test = len(test_metadata) == len(test_umap_subset)
                self.log_status("Metadata-projection alignment", alignment_test, f"{sample_size} samples tested")
            
            self.log_memory("integration_tests")
            return index_consistency
            
        except Exception as e:
            self.log_status("Integration tests", False, f"Error: {e}")
            return False
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        logger.info("📋 Generating Validation Report")
        
        # Count successes
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results.values() if result['status'])
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Memory usage summary
        if self.memory_stats:
            max_memory = max(stat['memory_mb'] for stat in self.memory_stats)
            final_memory = self.memory_stats[-1]['memory_mb']
        else:
            max_memory = final_memory = 0
        
        report = {
            'validation_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': f"{success_rate:.1f}%",
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'memory_usage': {
                'peak_memory_mb': max_memory,
                'final_memory_mb': final_memory,
                'memory_efficient': max_memory < 2000  # Less than 2GB
            },
            'test_results': self.validation_results,
            'memory_timeline': self.memory_stats,
            'visualizations': {
                'generated': self.enable_visualizations,
                'output_directory': str(self.viz_dir) if self.enable_visualizations else None
            }
        }
        
        # Save report
        report_path = self.processed_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Validation report saved to {report_path}")
        return report
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation pipeline"""
        logger.info("🚀 Starting Comprehensive Dataset Validation")
        logger.info("=" * 60)
        
        start_time = time.time()
        self.log_memory("start")
        
        # Step 1: File existence
        files_exist = self.validate_file_existence()
        if not files_exist:
            logger.error("❌ File validation failed - stopping validation")
            return self.generate_validation_report()
        
        # Step 2: Expression matrix
        expression_valid = self.validate_expression_matrix()
        
        # Step 3: Sample metadata
        metadata_valid, metadata = self.validate_sample_metadata()
        
        # Step 4: Feature metadata
        features_valid = self.validate_feature_metadata()
        
        # Step 5: Projections
        projections_valid, projections = self.validate_projections()
        
        # Step 6: Unstructured metadata
        unstructured_valid = self.validate_unstructured_metadata()
        
        # Step 7: Integration tests
        if metadata is not None and projections:
            integration_valid = self.validate_integration_tests(metadata, projections)
        else:
            integration_valid = False
            self.log_status("Integration tests", False, "Missing metadata or projections")
        
        # Step 8: Generate visualizations
        if metadata is not None and projections and self.enable_visualizations:
            try:
                self.generate_visualizations(metadata, projections)
            except Exception as e:
                self.log_status("Visualization generation", False, f"Error: {e}")
        
        # Final report
        end_time = time.time()
        validation_time = end_time - start_time
        
        logger.info("=" * 60)
        logger.info(f"⏱️  Validation completed in {validation_time:.1f} seconds")
        
        report = self.generate_validation_report()
        
        # Print summary
        summary = report['validation_summary']
        logger.info(f"🏆 VALIDATION SUMMARY:")
        logger.info(f"   Tests passed: {summary['passed_tests']}/{summary['total_tests']} ({summary['success_rate']})")
        logger.info(f"   Peak memory: {report['memory_usage']['peak_memory_mb']:.1f}MB")
        
        if summary['success_rate'] == '100.0%':
            logger.info("✅ ALL TESTS PASSED - Dataset ready for use!")
        else:
            logger.warning("⚠️  Some tests failed - check validation report")
        
        return report

def main():
    """Main execution function"""
    validator = ComprehensiveDatasetValidator(
        processed_dir="processed",
        generate_visualizations=True
    )
    
    report = validator.run_comprehensive_validation()
    
    # Return appropriate exit code
    success_rate = float(report['validation_summary']['success_rate'].rstrip('%'))
    return 0 if success_rate == 100.0 else 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 