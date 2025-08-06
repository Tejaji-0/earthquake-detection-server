#!/usr/bin/env python3
"""
Sensor-Based Earthquake Detection ML Pipeline
This script creates ML models for real-time earthquake detection using
accelerometer, gyroscope, and vibratory sensor data.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from accelerometer_feature_extractor import AccelerometerFeatureExtractor
import warnings
warnings.filterwarnings('ignore')

class SensorEarthquakeMLPipeline:
    def __init__(self, output_dir='sensor_ml_models'):
        self.output_dir = output_dir
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_names = []
        self.training_data = None
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize feature extractor
        self.feature_extractor = AccelerometerFeatureExtractor(self.output_dir)
    
    def load_training_data(self, data_files):
        """
        Load training data from sensor files
        
        Parameters:
        data_files: list of tuples (filepath, label) where label is 0 for normal, 1 for earthquake
        """
        print("=== Loading Training Data ===")
        
        all_features = []
        
        for filepath, label in data_files:
            if os.path.exists(filepath):
                print(f"Processing: {filepath} (label: {label})")
                features = self.feature_extractor.process_sensor_file(filepath, label)
                if features:
                    all_features.append(features)
            else:
                print(f"Warning: File not found: {filepath}")
        
        if all_features:
            self.training_data = pd.DataFrame(all_features)
            print(f"✓ Loaded {len(all_features)} samples")
            
            # Show class distribution
            class_counts = self.training_data['label'].value_counts()
            print(f"Class distribution: Normal={class_counts.get(0, 0)}, Earthquake={class_counts.get(1, 0)}")
            
            return True
        else:
            print("⚠ No training data loaded")
            return False
    
    def generate_synthetic_training_data(self, normal_samples=500, earthquake_samples=500):
        """Generate synthetic training data for demonstration"""
        print("=== Generating Synthetic Training Data ===")
        
        all_features = []
        
        # Generate normal samples
        print(f"Generating {normal_samples} normal samples...")
        for i in range(normal_samples):
            # Create random normal sensor data
            duration = np.random.uniform(5, 15)  # 5-15 seconds
            sampling_rate = 100
            time_points = int(duration * sampling_rate)
            
            # Normal background noise
            noise_level = np.random.uniform(0.05, 0.2)
            sensor_data = {
                'acc_x': np.random.normal(0, noise_level, time_points),
                'acc_y': np.random.normal(0, noise_level, time_points),
                'acc_z': np.random.normal(9.81, noise_level, time_points),
                'gyro_x': np.random.normal(0, 0.02, time_points),
                'gyro_y': np.random.normal(0, 0.02, time_points),
                'gyro_z': np.random.normal(0, 0.01, time_points)
            }
            
            features = self.feature_extractor.extract_all_features(sensor_data, sampling_rate)
            features['label'] = 0
            features['sample_id'] = f"normal_{i:04d}"
            all_features.append(features)
        
        # Generate earthquake samples
        print(f"Generating {earthquake_samples} earthquake samples...")
        for i in range(earthquake_samples):
            # Create earthquake-like sensor data
            duration = np.random.uniform(10, 30)  # 10-30 seconds
            sampling_rate = 100
            time_points = int(duration * sampling_rate)
            t = np.linspace(0, duration, time_points)
            
            # Background noise
            noise_level = np.random.uniform(0.1, 0.3)
            acc_x = np.random.normal(0, noise_level, time_points)
            acc_y = np.random.normal(0, noise_level, time_points)
            acc_z = np.random.normal(9.81, noise_level, time_points)
            
            # Earthquake characteristics
            earthquake_start = int(np.random.uniform(0.2, 0.4) * time_points)
            earthquake_duration = int(np.random.uniform(0.3, 0.6) * time_points)
            earthquake_end = min(earthquake_start + earthquake_duration, time_points)
            
            # P-wave (faster, smaller amplitude)
            p_freq = np.random.uniform(5, 15)  # 5-15 Hz
            p_amplitude = np.random.uniform(1, 4)  # 1-4 m/s²
            
            # S-wave (slower, larger amplitude)
            s_freq = np.random.uniform(1, 5)  # 1-5 Hz
            s_amplitude = np.random.uniform(3, 10)  # 3-10 m/s²
            s_delay = int(np.random.uniform(0.1, 0.3) * earthquake_duration)
            
            # Add P-wave
            p_wave_indices = range(earthquake_start, earthquake_end)
            for idx in p_wave_indices:
                if idx < time_points:
                    p_signal = p_amplitude * np.sin(2 * np.pi * p_freq * t[idx])
                    acc_x[idx] += p_signal * np.random.uniform(0.7, 1.0)
                    acc_y[idx] += p_signal * np.random.uniform(0.5, 0.9)
                    acc_z[idx] += p_signal * np.random.uniform(0.3, 0.7)
            
            # Add S-wave (delayed)
            s_start = earthquake_start + s_delay
            s_wave_indices = range(s_start, earthquake_end)
            for idx in s_wave_indices:
                if idx < time_points:
                    s_signal = s_amplitude * np.sin(2 * np.pi * s_freq * t[idx])
                    acc_x[idx] += s_signal * np.random.uniform(0.8, 1.2)
                    acc_y[idx] += s_signal * np.random.uniform(0.6, 1.0)
                    acc_z[idx] += s_signal * np.random.uniform(0.4, 0.8)
            
            # Generate correlated gyroscope data
            gyro_x = np.gradient(acc_y) * 5 + np.random.normal(0, 0.1, time_points)
            gyro_y = -np.gradient(acc_x) * 5 + np.random.normal(0, 0.1, time_points)
            gyro_z = np.gradient(acc_z) * 2 + np.random.normal(0, 0.05, time_points)
            
            sensor_data = {
                'acc_x': acc_x,
                'acc_y': acc_y,
                'acc_z': acc_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z
            }
            
            features = self.feature_extractor.extract_all_features(sensor_data, sampling_rate)
            features['label'] = 1
            features['sample_id'] = f"earthquake_{i:04d}"
            all_features.append(features)
        
        # Create DataFrame
        self.training_data = pd.DataFrame(all_features)
        
        print(f"✓ Generated {len(all_features)} synthetic samples")
        print(f"Features per sample: {len(all_features[0]) - 2}")  # Exclude label and sample_id
        
        # Save synthetic data
        synthetic_data_file = os.path.join(self.output_dir, 'synthetic_training_data.csv')
        self.training_data.to_csv(synthetic_data_file, index=False)
        print(f"✓ Synthetic data saved to: {synthetic_data_file}")
        
        return True
    
    def prepare_features(self, test_size=0.2, feature_selection_k=50):
        """Prepare features for training"""
        print("\n=== Preparing Features ===")
        
        if self.training_data is None:
            print("⚠ No training data available")
            return False
        
        # Separate features and labels
        feature_cols = [col for col in self.training_data.columns 
                       if col not in ['label', 'sample_id', 'filename', 'timestamp']]
        
        X = self.training_data[feature_cols]
        y = self.training_data['label']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        print("Scaling features...")
        scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Feature selection
        print(f"Selecting top {feature_selection_k} features...")
        selector = SelectKBest(score_func=f_classif, k=min(feature_selection_k, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        self.feature_selectors['main'] = selector
        
        # Store selected feature names
        selected_indices = selector.get_support(indices=True)
        self.selected_feature_names = [self.feature_names[i] for i in selected_indices]
        
        print(f"Selected features: {len(self.selected_feature_names)}")
        print(f"Top 10 features: {self.selected_feature_names[:10]}")
        
        return X_train_selected, X_test_selected, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple ML models"""
        print("\n=== Training ML Models ===")
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # Define models with hyperparameter grids
        model_configs = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, class_weight='balanced', probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'MLP': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        trained_models = {}
        model_results = {}
        
        for model_name, config in model_configs.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=cv, 
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Make predictions
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc')
                
                # Store results
                trained_models[model_name] = best_model
                model_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'test_roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                print(f"✓ {model_name}: CV ROC-AUC = {grid_search.best_score_:.3f}, Test ROC-AUC = {roc_auc:.3f}")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {e}")
                continue
        
        # Create ensemble model
        if len(trained_models) >= 2:
            print(f"\nCreating ensemble model...")
            
            # Select best models for ensemble (top 3)
            sorted_models = sorted(model_results.items(), key=lambda x: x[1]['test_roc_auc'], reverse=True)
            top_models = sorted_models[:3]
            
            ensemble_estimators = [(name, trained_models[name]) for name, _ in top_models]
            ensemble_model = VotingClassifier(estimators=ensemble_estimators, voting='soft')
            ensemble_model.fit(X_train, y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = ensemble_model.predict(X_test)
            y_pred_proba_ensemble = ensemble_model.predict_proba(X_test)[:, 1]
            roc_auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)
            
            trained_models['Ensemble'] = ensemble_model
            model_results['Ensemble'] = {
                'test_roc_auc': roc_auc_ensemble,
                'predictions': y_pred_ensemble,
                'probabilities': y_pred_proba_ensemble,
                'ensemble_models': [name for name, _ in top_models],
                'classification_report': classification_report(y_test, y_pred_ensemble, output_dict=True)
            }
            
            print(f"✓ Ensemble: Test ROC-AUC = {roc_auc_ensemble:.3f}")
        
        self.models = trained_models
        
        return model_results, y_test
    
    def create_visualizations(self, model_results, y_test):
        """Create visualizations for model performance"""
        print("\n=== Creating Visualizations ===")
        
        # ROC Curves
        plt.figure(figsize=(12, 8))
        
        for model_name, results in model_results.items():
            if 'probabilities' in results:
                fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {results['test_roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Earthquake Detection Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        roc_file = os.path.join(self.output_dir, 'roc_curves.png')
        plt.savefig(roc_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model comparison
        model_names = list(model_results.keys())
        roc_scores = [model_results[name]['test_roc_auc'] for name in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, roc_scores, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.ylabel('ROC-AUC Score')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, roc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3, axis='y')
        comparison_file = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance (for tree-based models)
        if 'RandomForest' in self.models:
            rf_model = self.models['RandomForest']
            feature_importance = rf_model.feature_importances_
            
            # Get top 20 features
            top_features_idx = np.argsort(feature_importance)[-20:]
            top_features = [self.selected_feature_names[i] for i in top_features_idx]
            top_importance = feature_importance[top_features_idx]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Feature Importance (Random Forest)')
            plt.tight_layout()
            
            importance_file = os.path.join(self.output_dir, 'feature_importance.png')
            plt.savefig(importance_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Visualizations saved to {self.output_dir}")
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        print("\n=== Saving Models ===")
        
        for model_name, model in self.models.items():
            model_file = os.path.join(self.output_dir, f'{model_name.lower()}_earthquake_detector.joblib')
            joblib.dump(model, model_file)
            print(f"✓ {model_name} saved to: {model_file}")
        
        # Save scalers and feature selectors
        scaler_file = os.path.join(self.output_dir, 'feature_scaler.joblib')
        joblib.dump(self.scalers['main'], scaler_file)
        
        selector_file = os.path.join(self.output_dir, 'feature_selector.joblib')
        joblib.dump(self.feature_selectors['main'], selector_file)
        
        # Save feature names
        feature_info = {
            'all_feature_names': self.feature_names,
            'selected_feature_names': self.selected_feature_names,
            'n_features': len(self.selected_feature_names)
        }
        
        feature_file = os.path.join(self.output_dir, 'feature_info.json')
        with open(feature_file, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"✓ Preprocessing objects and feature info saved")
    
    def create_training_report(self, model_results):
        """Create comprehensive training report"""
        report = {
            'training_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(self.training_data),
                'features_used': len(self.selected_feature_names),
                'models_trained': list(model_results.keys())
            },
            'model_performance': {},
            'feature_info': {
                'total_features_extracted': len(self.feature_names),
                'features_selected': len(self.selected_feature_names),
                'top_features': self.selected_feature_names[:10]
            },
            'recommendations': []
        }
        
        # Model performance summary
        for model_name, results in model_results.items():
            report['model_performance'][model_name] = {
                'test_roc_auc': results['test_roc_auc'],
                'best_params': results.get('best_params', {}),
                'classification_metrics': results['classification_report']
            }
        
        # Find best model
        best_model = max(model_results.items(), key=lambda x: x[1]['test_roc_auc'])
        report['best_model'] = {
            'name': best_model[0],
            'roc_auc': best_model[1]['test_roc_auc']
        }
        
        # Recommendations
        if best_model[1]['test_roc_auc'] > 0.9:
            report['recommendations'].append("Excellent model performance achieved!")
        elif best_model[1]['test_roc_auc'] > 0.8:
            report['recommendations'].append("Good model performance. Consider collecting more diverse training data.")
        else:
            report['recommendations'].append("Consider feature engineering or collecting more training data.")
        
        report['recommendations'].extend([
            "Test the model with real sensor data before deployment",
            "Implement continuous learning with new earthquake data",
            "Consider ensemble methods for critical applications",
            "Monitor model performance in production"
        ])
        
        # Save report
        report_file = os.path.join(self.output_dir, 'training_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create human-readable summary
        summary_file = os.path.join(self.output_dir, 'TRAINING_SUMMARY.md')
        with open(summary_file, 'w') as f:
            f.write("# Sensor-Based Earthquake Detection - Training Summary\n\n")
            f.write(f"**Training Date:** {report['training_summary']['timestamp']}\n\n")
            f.write(f"**Dataset:** {report['training_summary']['total_samples']} samples\n")
            f.write(f"**Features:** {report['training_summary']['features_used']} selected features\n\n")
            
            f.write("## Model Performance\n\n")
            for model_name, perf in report['model_performance'].items():
                f.write(f"- **{model_name}:** ROC-AUC = {perf['test_roc_auc']:.3f}\n")
            
            f.write(f"\n**Best Model:** {report['best_model']['name']} (ROC-AUC: {report['best_model']['roc_auc']:.3f})\n\n")
            
            f.write("## Top Features\n\n")
            for i, feature in enumerate(report['feature_info']['top_features'], 1):
                f.write(f"{i}. {feature}\n")
            
            f.write("\n## Recommendations\n\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"✓ Training report saved to: {report_file}")
        print(f"✓ Summary saved to: {summary_file}")
    
    def run_complete_pipeline(self, use_synthetic_data=True):
        """Run the complete ML pipeline"""
        print("🌊 SENSOR-BASED EARTHQUAKE DETECTION ML PIPELINE 🌊")
        print("=" * 60)
        
        # Step 1: Load or generate training data
        if use_synthetic_data:
            success = self.generate_synthetic_training_data(normal_samples=1000, earthquake_samples=1000)
        else:
            # For real data, you would call:
            # data_files = [('normal_data1.csv', 0), ('earthquake_data1.csv', 1), ...]
            # success = self.load_training_data(data_files)
            print("⚠ Real data loading not implemented. Using synthetic data.")
            success = self.generate_synthetic_training_data(normal_samples=1000, earthquake_samples=1000)
        
        if not success:
            print("❌ Failed to load training data")
            return False
        
        # Step 2: Prepare features
        feature_data = self.prepare_features()
        if not feature_data:
            print("❌ Failed to prepare features")
            return False
        
        X_train, X_test, y_train, y_test = feature_data
        
        # Step 3: Train models
        model_results, y_test = self.train_models(X_train, X_test, y_train, y_test)
        
        if not model_results:
            print("❌ Failed to train models")
            return False
        
        # Step 4: Create visualizations
        self.create_visualizations(model_results, y_test)
        
        # Step 5: Save models
        self.save_models()
        
        # Step 6: Create report
        self.create_training_report(model_results)
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print(f"📁 All outputs saved to: {self.output_dir}")
        
        # Show best model
        best_model = max(model_results.items(), key=lambda x: x[1]['test_roc_auc'])
        print(f"🏆 Best Model: {best_model[0]} (ROC-AUC: {best_model[1]['test_roc_auc']:.3f})")
        
        return True


def main():
    """Main function to run the sensor-based earthquake detection pipeline"""
    pipeline = SensorEarthquakeMLPipeline()
    
    print("Sensor-Based Earthquake Detection ML Pipeline")
    print("This pipeline trains models to detect earthquakes from accelerometer and gyroscope data\n")
    
    # Run the complete pipeline
    success = pipeline.run_complete_pipeline(use_synthetic_data=True)
    
    if success:
        print("\n✅ Training completed successfully!")
        print(f"🔍 Check '{pipeline.output_dir}' for trained models and reports")
        print("\n📋 Next steps:")
        print("1. Test the trained models with real sensor data")
        print("2. Deploy the best model for real-time earthquake detection")
        print("3. Continuously collect labeled data to improve the model")
    else:
        print("\n❌ Training failed. Check error messages above.")


if __name__ == "__main__":
    main()
