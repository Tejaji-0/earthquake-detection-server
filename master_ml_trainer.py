#!/usr/bin/env python3
"""
Master Script for Enhanced Earthquake ML Training
This script orchestrates the complete enhanced machine learning pipeline including:
1. Data fetching from multiple APIs
2. Seismic feature extraction from .mseed files
3. Enhanced feature engineering
4. Advanced model training with hyperparameter optimization
5. Model evaluation and visualization
"""

import os
import sys
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_data_fetcher import EnhancedEarthquakeDataFetcher
    from advanced_seismic_extractor import AdvancedSeismicFeatureExtractor
    from enhanced_ml_pipeline import EnhancedEarthquakeMLPipeline
    print("✓ All enhanced modules imported successfully")
except ImportError as e:
    print(f"⚠ Import error: {e}")
    print("Make sure all required files are in the same directory")

class MasterEarthquakeMLTrainer:
    def __init__(self, 
                 base_data_path='data/database.csv',
                 seismic_data_dir='earthquake_seismic_data',
                 output_dir='master_ml_models'):
        
        self.base_data_path = base_data_path
        self.seismic_data_dir = seismic_data_dir
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize components
        self.data_fetcher = None
        self.seismic_extractor = None
        self.ml_pipeline = None
        
        # Execution log
        self.execution_log = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'errors': [],
            'data_sources': [],
            'models_trained': [],
            'performance_metrics': {}
        }
        
        print(f"Master Earthquake ML Trainer initialized")
        print(f"Base data: {self.base_data_path}")
        print(f"Seismic data: {self.seismic_data_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def step1_fetch_enhanced_data(self, include_historical=True, historical_days=365):
        """Step 1: Fetch enhanced earthquake data from multiple APIs"""
        print("\n" + "="*80)
        print("STEP 1: FETCHING ENHANCED EARTHQUAKE DATA")
        print("="*80)
        
        try:
            self.data_fetcher = EnhancedEarthquakeDataFetcher(
                output_dir=os.path.join(self.output_dir, 'fetched_data')
            )
            
            # Fetch data
            enhanced_data_file = self.data_fetcher.fetch_all_data(
                include_historical=include_historical,
                historical_days=historical_days,
                min_magnitude=4.0
            )
            
            if enhanced_data_file:
                self.execution_log['steps_completed'].append('data_fetching')
                self.execution_log['data_sources'].append('API_fetched_data')
                print(f"✓ Step 1 completed: Enhanced data saved to {enhanced_data_file}")
                return enhanced_data_file
            else:
                raise Exception("No enhanced data was fetched")
                
        except Exception as e:
            error_msg = f"Step 1 failed: {e}"
            print(f"✗ {error_msg}")
            self.execution_log['errors'].append(error_msg)
            return None
    
    def step2_extract_seismic_features(self):
        """Step 2: Extract advanced features from seismic waveform data"""
        print("\n" + "="*80)
        print("STEP 2: EXTRACTING SEISMIC FEATURES")
        print("="*80)
        
        try:
            # Check if ObsPy and PyWavelets are available
            try:
                import obspy
                import pywt
                print("✓ ObsPy and PyWavelets available for seismic processing")
            except ImportError as e:
                print(f"⚠ ObsPy/PyWavelets not available - installing...")
                import subprocess
                subprocess.run(["pip", "install", "obspy", "scipy", "PyWavelets"], check=True)
                print("✓ Dependencies installed successfully")
                
            self.seismic_extractor = AdvancedSeismicFeatureExtractor(
                seismic_data_dir=self.seismic_data_dir,
                output_dir=self.output_dir
            )
            
            # Extract features
            features_df = self.seismic_extractor.extract_all_features()
            
            if features_df is not None and not features_df.empty:
                # Aggregate features
                aggregated_df = self.seismic_extractor.aggregate_features_by_earthquake()
                
                # Create summary
                summary = self.seismic_extractor.create_feature_summary()
                
                self.execution_log['steps_completed'].append('seismic_feature_extraction')
                self.execution_log['data_sources'].append('seismic_features')
                print(f"✓ Step 2 completed: Extracted features from {len(features_df)} traces")
                return True
            else:
                print("⚠ No seismic features extracted - continuing without seismic data")
                return False
                
        except Exception as e:
            error_msg = f"Step 2 failed: {e}"
            print(f"✗ {error_msg}")
            self.execution_log['errors'].append(error_msg)
            return False
    
    def step3_prepare_combined_dataset(self, enhanced_data_file=None):
        """Step 3: Prepare combined dataset with all available data sources"""
        print("\n" + "="*80)
        print("STEP 3: PREPARING COMBINED DATASET")
        print("="*80)
        
        try:
            # Determine data sources
            data_sources = []
            
            # Base earthquake database
            if os.path.exists(self.base_data_path):
                data_sources.append(self.base_data_path)
                print(f"✓ Using base data: {self.base_data_path}")
            
            # Enhanced fetched data
            if enhanced_data_file and os.path.exists(enhanced_data_file):
                data_sources.append(enhanced_data_file)
                print(f"✓ Using enhanced data: {enhanced_data_file}")
            
            if not data_sources:
                raise Exception("No earthquake data sources available")
            
            # Use the most comprehensive data source
            primary_data_source = enhanced_data_file if enhanced_data_file else self.base_data_path
            
            print(f"Primary data source: {primary_data_source}")
            
            self.execution_log['steps_completed'].append('dataset_preparation')
            self.execution_log['data_sources'].extend(data_sources)
            return primary_data_source
            
        except Exception as e:
            error_msg = f"Step 3 failed: {e}"
            print(f"✗ {error_msg}")
            self.execution_log['errors'].append(error_msg)
            return None
    
    def step4_train_enhanced_models(self, data_source):
        """Step 4: Train enhanced ML models with all available features"""
        print("\n" + "="*80)
        print("STEP 4: TRAINING ENHANCED ML MODELS")
        print("="*80)
        
        try:
            # Initialize enhanced ML pipeline
            self.ml_pipeline = EnhancedEarthquakeMLPipeline(
                earthquake_csv=data_source,
                seismic_data_dir=self.seismic_data_dir,
                output_dir=self.output_dir
            )
            
            # Run the enhanced pipeline
            tasks = [
                'major_earthquake',      # M >= 7.0
                'very_major_earthquake', # M >= 8.0
                'significant_earthquake', # High significance
                'tsunami_generating',    # Tsunami risk
                'shallow_earthquake',    # Depth <= 35km
                'deep_earthquake'        # Depth >= 300km
            ]
            
            self.ml_pipeline.run_enhanced_pipeline(tasks=tasks)
            
            # Collect performance metrics
            for task in tasks:
                if task in self.ml_pipeline.models:
                    results = self.ml_pipeline.models[task]['results']
                    self.execution_log['performance_metrics'][task] = {}
                    
                    for model_name, result in results.items():
                        self.execution_log['performance_metrics'][task][model_name] = {
                            'cv_score': result.get('best_cv_score', 0),
                            'test_roc_auc': result.get('roc_auc', 0)
                        }
            
            self.execution_log['steps_completed'].append('model_training')
            self.execution_log['models_trained'] = tasks
            print(f"✓ Step 4 completed: Trained models for {len(tasks)} tasks")
            return True
            
        except Exception as e:
            error_msg = f"Step 4 failed: {e}"
            print(f"✗ {error_msg}")
            self.execution_log['errors'].append(error_msg)
            return False
    
    def step5_create_comprehensive_report(self):
        """Step 5: Create comprehensive training report"""
        print("\n" + "="*80)
        print("STEP 5: CREATING COMPREHENSIVE REPORT")
        print("="*80)
        
        try:
            self.execution_log['end_time'] = datetime.now().isoformat()
            self.execution_log['total_duration'] = str(
                datetime.fromisoformat(self.execution_log['end_time']) - 
                datetime.fromisoformat(self.execution_log['start_time'])
            )
            
            # Create detailed report
            report = {
                'execution_summary': self.execution_log,
                'system_info': {
                    'python_version': sys.version,
                    'working_directory': os.getcwd(),
                    'output_directory': self.output_dir
                },
                'data_summary': self.get_data_summary(),
                'model_summary': self.get_model_summary(),
                'recommendations': self.generate_recommendations()
            }
            
            # Save report
            report_file = os.path.join(self.output_dir, 'comprehensive_training_report.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Create human-readable summary
            self.create_readable_summary(report)
            
            self.execution_log['steps_completed'].append('report_generation')
            print(f"✓ Step 5 completed: Report saved to {report_file}")
            return True
            
        except Exception as e:
            error_msg = f"Step 5 failed: {e}"
            print(f"✗ {error_msg}")
            self.execution_log['errors'].append(error_msg)
            return False
    
    def get_data_summary(self):
        """Get summary of data used in training"""
        summary = {
            'sources_used': self.execution_log['data_sources'],
            'seismic_features_included': 'seismic_features' in self.execution_log['data_sources']
        }
        
        # Try to get dataset statistics
        try:
            if self.ml_pipeline and self.ml_pipeline.earthquake_df is not None:
                df = self.ml_pipeline.earthquake_df
                summary.update({
                    'total_earthquakes': len(df),
                    'date_range': {
                        'start': str(df['datetime'].min()) if 'datetime' in df.columns else 'Unknown',
                        'end': str(df['datetime'].max()) if 'datetime' in df.columns else 'Unknown'
                    },
                    'magnitude_range': {
                        'min': float(df['Magnitude'].min()),
                        'max': float(df['Magnitude'].max()),
                        'mean': float(df['Magnitude'].mean())
                    } if 'Magnitude' in df.columns else {},
                    'feature_count': len(self.ml_pipeline.feature_matrix.columns) if self.ml_pipeline.feature_matrix is not None else 0
                })
        except:
            pass
        
        return summary
    
    def get_model_summary(self):
        """Get summary of trained models"""
        summary = {
            'tasks_trained': self.execution_log['models_trained'],
            'performance_metrics': self.execution_log['performance_metrics']
        }
        
        # Find best performing models
        best_models = {}
        for task, models in self.execution_log['performance_metrics'].items():
            if models:
                best_model = max(models.items(), 
                               key=lambda x: x[1].get('test_roc_auc', 0) or x[1].get('cv_score', 0))
                best_models[task] = {
                    'model': best_model[0],
                    'performance': best_model[1]
                }
        
        summary['best_models'] = best_models
        return summary
    
    def generate_recommendations(self):
        """Generate recommendations based on training results"""
        recommendations = []
        
        # Check if seismic features were used
        if 'seismic_features' not in self.execution_log['data_sources']:
            recommendations.append(
                "Consider installing ObsPy and including seismic waveform features for improved model performance"
            )
        
        # Check for errors
        if self.execution_log['errors']:
            recommendations.append(
                f"Address the following errors for complete pipeline execution: {'; '.join(self.execution_log['errors'])}"
            )
        
        # Check model performance
        if self.execution_log['performance_metrics']:
            low_performance_tasks = []
            for task, models in self.execution_log['performance_metrics'].items():
                if models:
                    best_score = max(model.get('test_roc_auc', 0) or model.get('cv_score', 0) 
                                   for model in models.values())
                    if best_score < 0.8:
                        low_performance_tasks.append(task)
            
            if low_performance_tasks:
                recommendations.append(
                    f"Consider feature engineering or data augmentation for tasks with lower performance: {', '.join(low_performance_tasks)}"
                )
        
        # General recommendations
        recommendations.extend([
            "Regularly update the model with new earthquake data for optimal performance",
            "Consider ensemble methods for critical prediction tasks",
            "Implement real-time monitoring for production deployment",
            "Validate models on independent test datasets from different time periods"
        ])
        
        return recommendations
    
    def create_readable_summary(self, report):
        """Create human-readable summary file"""
        summary_file = os.path.join(self.output_dir, 'TRAINING_SUMMARY.md')
        
        with open(summary_file, 'w') as f:
            f.write("# Enhanced Earthquake ML Training Summary\n\n")
            
            # Execution overview
            f.write("## Execution Overview\n\n")
            f.write(f"- **Start Time:** {self.execution_log['start_time']}\n")
            f.write(f"- **End Time:** {self.execution_log['end_time']}\n")
            f.write(f"- **Duration:** {self.execution_log['total_duration']}\n")
            f.write(f"- **Steps Completed:** {', '.join(self.execution_log['steps_completed'])}\n\n")
            
            # Data summary
            f.write("## Data Summary\n\n")
            data_summary = report['data_summary']
            f.write(f"- **Data Sources:** {', '.join(data_summary['sources_used'])}\n")
            f.write(f"- **Seismic Features Included:** {'Yes' if data_summary['seismic_features_included'] else 'No'}\n")
            
            if 'total_earthquakes' in data_summary:
                f.write(f"- **Total Earthquakes:** {data_summary['total_earthquakes']:,}\n")
                f.write(f"- **Feature Count:** {data_summary['feature_count']}\n")
                
                if 'magnitude_range' in data_summary and data_summary['magnitude_range']:
                    mag_range = data_summary['magnitude_range']
                    f.write(f"- **Magnitude Range:** {mag_range['min']:.1f} - {mag_range['max']:.1f} (avg: {mag_range['mean']:.1f})\n")
            
            f.write("\n")
            
            # Model performance
            f.write("## Model Performance\n\n")
            model_summary = report['model_summary']
            
            if 'best_models' in model_summary:
                for task, info in model_summary['best_models'].items():
                    f.write(f"### {task.replace('_', ' ').title()}\n")
                    f.write(f"- **Best Model:** {info['model']}\n")
                    
                    perf = info['performance']
                    if 'test_roc_auc' in perf and perf['test_roc_auc']:
                        f.write(f"- **Test ROC-AUC:** {perf['test_roc_auc']:.3f}\n")
                    if 'cv_score' in perf and perf['cv_score']:
                        f.write(f"- **CV Score:** {perf['cv_score']:.3f}\n")
                    f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            
            # Errors
            if self.execution_log['errors']:
                f.write("\n## Errors Encountered\n\n")
                for i, error in enumerate(self.execution_log['errors'], 1):
                    f.write(f"{i}. {error}\n")
            
            f.write(f"\n---\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        print(f"✓ Human-readable summary saved to: {summary_file}")
    
    def run_complete_pipeline(self, 
                             fetch_new_data=True, 
                             include_historical=True, 
                             historical_days=365):
        """Run the complete enhanced earthquake ML training pipeline"""
        
        print("\n" + "="*100)
        print("🌍 MASTER EARTHQUAKE ML TRAINING PIPELINE 🌍")
        print("="*100)
        print(f"Starting comprehensive earthquake ML model training...")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        enhanced_data_file = None
        
        # Step 1: Fetch enhanced data (optional)
        if fetch_new_data:
            enhanced_data_file = self.step1_fetch_enhanced_data(
                include_historical=include_historical,
                historical_days=historical_days
            )
        else:
            print("\nSkipping data fetching - using existing data only")
        
        # Step 2: Extract seismic features
        self.step2_extract_seismic_features()
        
        # Step 3: Prepare combined dataset
        data_source = self.step3_prepare_combined_dataset(enhanced_data_file)
        
        if data_source is None:
            print("\n❌ Pipeline failed: No data source available")
            return False
        
        # Step 4: Train enhanced models
        training_success = self.step4_train_enhanced_models(data_source)
        
        if not training_success:
            print("\n❌ Pipeline failed: Model training unsuccessful")
            return False
        
        # Step 5: Create comprehensive report
        self.step5_create_comprehensive_report()
        
        # Final summary
        print("\n" + "="*100)
        print("🎉 MASTER PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("="*100)
        print(f"✓ Steps completed: {len(self.execution_log['steps_completed'])}")
        print(f"✓ Models trained: {len(self.execution_log['models_trained'])}")
        print(f"✓ Data sources used: {len(self.execution_log['data_sources'])}")
        
        if self.execution_log['errors']:
            print(f"⚠ Errors encountered: {len(self.execution_log['errors'])}")
        
        print(f"📁 All outputs saved to: {self.output_dir}")
        print(f"⏱ Total duration: {self.execution_log.get('total_duration', 'Unknown')}")
        print("="*100)
        
        return True


def main():
    """Main function to run the complete enhanced ML training pipeline"""
    
    # Configuration
    config = {
        'fetch_new_data': True,          # Whether to fetch new data from APIs
        'include_historical': True,      # Whether to include historical data
        'historical_days': 730,          # Days of historical data to fetch
        'base_data_path': 'data/database.csv',  # Path to existing earthquake data
        'seismic_data_dir': 'earthquake_seismic_data',  # Directory with .mseed files
        'output_dir': 'master_ml_models'  # Output directory for all results
    }
    
    print("Enhanced Earthquake ML Training Pipeline")
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize and run master trainer
    trainer = MasterEarthquakeMLTrainer(
        base_data_path=config['base_data_path'],
        seismic_data_dir=config['seismic_data_dir'],
        output_dir=config['output_dir']
    )
    
    # Run complete pipeline
    success = trainer.run_complete_pipeline(
        fetch_new_data=config['fetch_new_data'],
        include_historical=config['include_historical'],
        historical_days=config['historical_days']
    )
    
    if success:
        print(f"\n🎯 Training completed successfully!")
        print(f"📊 Check '{config['output_dir']}' for all results and reports")
    else:
        print(f"\n❌ Training failed. Check error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
