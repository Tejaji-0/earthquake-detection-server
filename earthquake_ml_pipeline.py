#!/usr/bin/env python3
"""
Machine Learning Pipeline for Earthquake Detection
This script creates and trains ML models to detect earthquakes using seismic data features.
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class EarthquakeMLPipeline:
    def __init__(self, data_path='data/earthquake_1995-2023.csv', output_dir='ml_models'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.features = None
        self.labels = None
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_and_preprocess_data(self):
        """Load earthquake data and create features for ML training."""
        print("Loading earthquake data...")
        self.df = pd.read_csv(self.data_path)
        
        # Convert date_time to datetime
        self.df['date_time'] = pd.to_datetime(self.df['date_time'], format='%d-%m-%Y %H:%M')
        
        # Sort by datetime
        self.df = self.df.sort_values('date_time').reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} earthquake records")
        print(f"Date range: {self.df['date_time'].min()} to {self.df['date_time'].max()}")
        
        return self.df
    
    def engineer_features(self):
        """Create features for earthquake detection."""
        print("Engineering features...")
        
        # Basic features from existing data
        features_df = pd.DataFrame()
        
        # Magnitude-based features
        features_df['magnitude'] = self.df['magnitude']
        features_df['magnitude_squared'] = self.df['magnitude'] ** 2
        features_df['magnitude_log'] = np.log(self.df['magnitude'] + 1)
        
        # Depth features
        features_df['depth'] = self.df['depth'].fillna(self.df['depth'].median())
        features_df['depth_log'] = np.log(features_df['depth'] + 1)
        features_df['depth_normalized'] = features_df['depth'] / features_df['depth'].max()
        
        # Geographic features
        features_df['latitude'] = self.df['latitude']
        features_df['longitude'] = self.df['longitude']
        features_df['lat_abs'] = np.abs(self.df['latitude'])
        features_df['lon_abs'] = np.abs(self.df['longitude'])
        
        # Distance from equator and prime meridian
        features_df['distance_from_equator'] = np.abs(self.df['latitude'])
        features_df['distance_from_prime_meridian'] = np.abs(self.df['longitude'])
        
        # Significance and alert features
        features_df['significance'] = self.df['sig'].fillna(0)
        features_df['significance_log'] = np.log(features_df['significance'] + 1)
        
        # Alert level encoding
        alert_encoder = LabelEncoder()
        alert_values = self.df['alert'].fillna('none')
        features_df['alert_encoded'] = alert_encoder.fit_transform(alert_values)
        
        # Tsunami flag
        features_df['tsunami'] = self.df['tsunami'].fillna(0)
        
        # Network and station features
        features_df['num_stations'] = self.df['nst'].fillna(0)
        features_df['gap'] = self.df['gap'].fillna(180)  # Default to 180 degrees
        features_df['dmin'] = self.df['dmin'].fillna(self.df['dmin'].median())
        
        # Temporal features
        features_df['year'] = self.df['date_time'].dt.year
        features_df['month'] = self.df['date_time'].dt.month
        features_df['day'] = self.df['date_time'].dt.day
        features_df['hour'] = self.df['date_time'].dt.hour
        features_df['day_of_year'] = self.df['date_time'].dt.dayofyear
        features_df['day_of_week'] = self.df['date_time'].dt.dayofweek
        
        # Seasonal features
        features_df['sin_month'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['cos_month'] = np.cos(2 * np.pi * features_df['month'] / 12)
        features_df['sin_hour'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['cos_hour'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        
        # Regional activity features (sliding window analysis)
        window_days = 30
        features_df['recent_activity_30d'] = self.calculate_recent_activity(window_days)
        
        window_days = 7
        features_df['recent_activity_7d'] = self.calculate_recent_activity(window_days)
        
        # Magnitude trend features
        features_df['magnitude_trend_7d'] = self.calculate_magnitude_trend(7)
        features_df['magnitude_trend_30d'] = self.calculate_magnitude_trend(30)
        
        # Remove any rows with NaN values
        features_df = features_df.fillna(features_df.median())
        
        self.features = features_df
        print(f"Created {len(features_df.columns)} features")
        return features_df
    
    def calculate_recent_activity(self, window_days):
        """Calculate recent seismic activity in a sliding window."""
        activity = []
        for i, row in self.df.iterrows():
            current_time = row['date_time']
            window_start = current_time - timedelta(days=window_days)
            
            # Count earthquakes in the window (excluding current event)
            recent_events = self.df[
                (self.df['date_time'] >= window_start) & 
                (self.df['date_time'] < current_time) &
                (np.abs(self.df['latitude'] - row['latitude']) < 5) &  # Within ~500km
                (np.abs(self.df['longitude'] - row['longitude']) < 5)
            ]
            
            activity.append(len(recent_events))
        
        return activity
    
    def calculate_magnitude_trend(self, window_days):
        """Calculate magnitude trend in recent earthquakes."""
        trends = []
        for i, row in self.df.iterrows():
            current_time = row['date_time']
            window_start = current_time - timedelta(days=window_days)
            
            # Get recent earthquakes in region
            recent_events = self.df[
                (self.df['date_time'] >= window_start) & 
                (self.df['date_time'] < current_time) &
                (np.abs(self.df['latitude'] - row['latitude']) < 5) &
                (np.abs(self.df['longitude'] - row['longitude']) < 5)
            ]
            
            if len(recent_events) > 1:
                # Calculate linear trend of magnitudes
                x = np.arange(len(recent_events))
                y = recent_events['magnitude'].values
                trend = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
            else:
                trend = 0
            
            trends.append(trend)
        
        return trends
    
    def create_labels(self):
        """Create classification labels for earthquake detection."""
        # Create different classification tasks
        
        # Adjust thresholds based on data distribution
        mag_median = self.df['magnitude'].median()
        mag_75th = self.df['magnitude'].quantile(0.75)
        sig_median = self.df['sig'].median()
        
        print(f"Magnitude stats - Median: {mag_median:.1f}, 75th percentile: {mag_75th:.1f}")
        print(f"Significance median: {sig_median:.0f}")
        
        self.labels = {
            'major_earthquake': (self.df['magnitude'] >= mag_75th).astype(int),
            'significant_earthquake': (self.df['sig'] >= sig_median).astype(int),
            'tsunami_generating': self.df['tsunami'].fillna(0).astype(int),
            'high_alert': (self.df['alert'].isin(['yellow', 'orange', 'red'])).astype(int)
        }
        
        print("Created classification labels:")
        for label_name, label_values in self.labels.items():
            print(f"  {label_name}: {label_values.sum()} positive cases ({label_values.mean()*100:.1f}%)")
        
        return self.labels
    
    def select_features(self, task='major_earthquake', k=20):
        """Select the best features using statistical tests."""
        X = self.features
        y = self.labels[task]
        
        # Select K best features
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        print(f"Selected {len(selected_features)} best features for {task}:")
        scores = self.feature_selector.scores_[self.feature_selector.get_support()]
        for feature, score in zip(selected_features, scores):
            print(f"  {feature}: {score:.2f}")
        
        return X_selected, selected_features
    
    def train_models(self, task='major_earthquake'):
        """Train multiple ML models for earthquake detection."""
        print(f"\nTraining models for task: {task}")
        
        # Select features and prepare data
        X, selected_features = self.select_features(task)
        y = self.labels[task]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Positive class ratio: {y_train.mean():.3f}")
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=6, 
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', 
                probability=True, 
                random_state=42,
                class_weight='balanced'
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=500, 
                random_state=42
            )
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                # Handle both binary and single class cases
                if proba.shape[1] > 1:
                    y_pred_proba = proba[:, 1]
                else:
                    y_pred_proba = proba[:, 0]
            
            # Evaluation metrics
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'roc_auc': None
            }
            
            # Calculate ROC-AUC only if we have binary classification
            if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                try:
                    results[name]['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                except ValueError:
                    results[name]['roc_auc'] = None
            
            # Cross-validation
            try:
                if len(np.unique(y_train)) > 1:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                    results[name]['cv_auc_mean'] = cv_scores.mean()
                    results[name]['cv_auc_std'] = cv_scores.std()
                else:
                    # Single class - use accuracy instead
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    results[name]['cv_auc_mean'] = cv_scores.mean()
                    results[name]['cv_auc_std'] = cv_scores.std()
            except Exception as e:
                print(f"Cross-validation failed for {name}: {e}")
                results[name]['cv_auc_mean'] = 0.5
                results[name]['cv_auc_std'] = 0.0
            
            roc_auc = results[name]['roc_auc']
            cv_mean = results[name]['cv_auc_mean']
            cv_std = results[name]['cv_auc_std']
            
            print(f"Test ROC-AUC: {roc_auc:.3f}" if roc_auc else "Test ROC-AUC: N/A (single class)")
            print(f"CV Score: {cv_mean:.3f} ± {cv_std:.3f}")
        
        # Store results
        self.models[task] = {
            'results': results,
            'test_data': (X_test, y_test),
            'selected_features': selected_features,
            'scaler': self.scaler
        }
        
        return results
    
    def optimize_best_model(self, task='major_earthquake'):
        """Optimize hyperparameters for the best performing model."""
        print(f"\nOptimizing best model for task: {task}")
        
        # Get best model based on CV score
        results = self.models[task]['results']
        best_model_name = max(results.keys(), 
                            key=lambda k: results[k]['cv_auc_mean'] if results[k]['cv_auc_mean'] else 0)
        
        print(f"Best model: {best_model_name}")
        
        # Prepare data
        X, _ = self.select_features(task)
        y = self.labels[task]
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define hyperparameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            }
        }
        
        if best_model_name in param_grids:
            # Get base model
            if best_model_name == 'Random Forest':
                base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
            elif best_model_name == 'Gradient Boosting':
                base_model = GradientBoostingClassifier(random_state=42)
            elif best_model_name == 'SVM':
                base_model = SVC(random_state=42, probability=True, class_weight='balanced')
            
            # Grid search
            grid_search = GridSearchCV(
                base_model,
                param_grids[best_model_name],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            print("Running grid search...")
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.3f}")
            
            # Evaluate optimized model
            optimized_model = grid_search.best_estimator_
            y_pred = optimized_model.predict(X_test)
            y_pred_proba = optimized_model.predict_proba(X_test)[:, 1]
            
            print(f"Optimized test ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
            
            # Store optimized model
            self.models[task]['optimized_model'] = optimized_model
            self.models[task]['best_params'] = grid_search.best_params_
            
            return optimized_model
        
        return None
    
    def analyze_feature_importance(self, task='major_earthquake'):
        """Analyze feature importance for tree-based models."""
        print(f"\nAnalyzing feature importance for task: {task}")
        
        results = self.models[task]['results']
        selected_features = self.models[task]['selected_features']
        
        # Analyze Random Forest and Gradient Boosting feature importance
        for model_name in ['Random Forest', 'Gradient Boosting']:
            if model_name in results:
                model = results[model_name]['model']
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Create importance dataframe
                    importance_df = pd.DataFrame({
                        'feature': selected_features,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    print(f"\nTop 10 features for {model_name}:")
                    for _, row in importance_df.head(10).iterrows():
                        print(f"  {row['feature']}: {row['importance']:.4f}")
                    
                    # Save importance plot
                    plt.figure(figsize=(10, 8))
                    plt.barh(range(len(importance_df.head(15))), 
                            importance_df.head(15)['importance'])
                    plt.yticks(range(len(importance_df.head(15))), 
                              importance_df.head(15)['feature'])
                    plt.xlabel('Feature Importance')
                    plt.title(f'Feature Importance - {model_name} ({task})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f'feature_importance_{model_name}_{task}.png'))
                    plt.close()
    
    def create_evaluation_plots(self, task='major_earthquake'):
        """Create evaluation plots for all models."""
        print(f"\nCreating evaluation plots for task: {task}")
        
        results = self.models[task]['results']
        X_test, y_test = self.models[task]['test_data']
        
        # ROC curves
        plt.figure(figsize=(12, 8))
        
        has_valid_roc = False
        for model_name, result in results.items():
            if result['probabilities'] is not None and result['roc_auc'] is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                    auc_score = result['roc_auc']
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
                    has_valid_roc = True
                except Exception as e:
                    print(f"Could not plot ROC for {model_name}: {e}")
        
        if has_valid_roc:
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {task}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, f'roc_curves_{task}.png'))
        else:
            plt.text(0.5, 0.5, 'No valid ROC curves\n(single class problem)', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'ROC Curves - {task} (No binary classification)')
            plt.savefig(os.path.join(self.output_dir, f'roc_curves_{task}.png'))
        
        plt.close()
        
        # Confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (model_name, result) in enumerate(results.items()):
            if i < 4:  # Only plot first 4 models
                cm = result['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
                axes[i].set_title(f'{model_name} - Confusion Matrix')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'confusion_matrices_{task}.png'))
        plt.close()
    
    def save_models(self, task='major_earthquake'):
        """Save trained models and metadata."""
        print(f"\nSaving models for task: {task}")
        
        task_dir = os.path.join(self.output_dir, task)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        
        # Save each model
        results = self.models[task]['results']
        for model_name, result in results.items():
            model_file = os.path.join(task_dir, f'{model_name.lower().replace(" ", "_")}_model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(result['model'], f)
        
        # Save optimized model if available
        if 'optimized_model' in self.models[task]:
            opt_model_file = os.path.join(task_dir, 'optimized_model.pkl')
            with open(opt_model_file, 'wb') as f:
                pickle.dump(self.models[task]['optimized_model'], f)
        
        # Save scaler and feature selector
        scaler_file = os.path.join(task_dir, 'scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        selector_file = os.path.join(task_dir, 'feature_selector.pkl')
        with open(selector_file, 'wb') as f:
            pickle.dump(self.feature_selector, f)
        
        # Save metadata
        metadata = {
            'task': task,
            'selected_features': self.models[task]['selected_features'],
            'model_performances': {
                name: {
                    'cv_auc_mean': result['cv_auc_mean'],
                    'cv_auc_std': result['cv_auc_std'],
                    'test_roc_auc': result['roc_auc']
                }
                for name, result in results.items()
            },
            'training_date': datetime.now().isoformat(),
            'data_size': len(self.df),
            'feature_count': len(self.features.columns),
            'selected_feature_count': len(self.models[task]['selected_features'])
        }
        
        if 'best_params' in self.models[task]:
            metadata['best_hyperparameters'] = self.models[task]['best_params']
        
        metadata_file = os.path.join(task_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models saved to: {task_dir}")
    
    def run_full_pipeline(self, tasks=None):
        """Run the complete ML pipeline."""
        if tasks is None:
            tasks = ['major_earthquake', 'significant_earthquake', 'tsunami_generating']
        
        print("=== Earthquake ML Pipeline ===")
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Engineer features
        self.engineer_features()
        
        # Create labels
        self.create_labels()
        
        # Run pipeline for each task
        for task in tasks:
            if task in self.labels:
                print(f"\n{'='*50}")
                print(f"Processing task: {task}")
                print('='*50)
                
                # Train models
                self.train_models(task)
                
                # Optimize best model
                self.optimize_best_model(task)
                
                # Analyze feature importance
                self.analyze_feature_importance(task)
                
                # Create evaluation plots
                self.create_evaluation_plots(task)
                
                # Save models
                self.save_models(task)
        
        print(f"\n=== Pipeline Complete ===")
        print(f"Results saved to: {os.path.abspath(self.output_dir)}")

def main():
    """Main function to run the earthquake ML pipeline."""
    # Initialize pipeline
    pipeline = EarthquakeMLPipeline()
    
    # Run full pipeline
    pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()
