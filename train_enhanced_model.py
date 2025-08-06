#!/usr/bin/env python3
"""
Enhanced ML Model Training Script for Earthquake Detection
This script trains the enhanced ensemble model with advanced features
and optimized hyperparameters for better earthquake detection.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import our enhanced feature extractor
from enhanced_realtime_detector import EnhancedFeatureExtractor

warnings.filterwarnings('ignore')

class EnhancedModelTrainer:
    """Enhanced model trainer with advanced feature engineering and hyperparameter optimization"""
    
    def __init__(self, output_dir='enhanced_ml_models'):
        self.output_dir = output_dir
        self.feature_extractor = EnhancedFeatureExtractor()
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Training configuration
        self.test_size = 0.2
        self.random_state = 42
        self.cv_folds = 5
        
        print(f"Enhanced Model Trainer initialized")
        print(f"Output directory: {self.output_dir}")
    
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic sensor data for training"""
        print("Generating synthetic training data...")
        
        np.random.seed(self.random_state)
        data = []
        labels = []
        
        # Generate normal activity samples (80%)
        n_normal = int(n_samples * 0.8)
        for i in range(n_normal):
            # Normal background noise and small movements with more variety
            base_std = np.random.uniform(0.02, 0.08)  # Varying noise levels
            
            acc_x = np.random.normal(0, base_std, 1000)
            acc_y = np.random.normal(0, base_std, 1000)
            acc_z = np.random.normal(1.0, base_std, 1000)
            
            # Add occasional small disturbances (human activity, vehicles, etc.)
            if np.random.random() < 0.4:
                disturbance_type = np.random.choice(['walking', 'vehicle', 'wind', 'machinery'])
                
                if disturbance_type == 'walking':
                    # Walking pattern - regular small peaks
                    for step in range(np.random.randint(3, 8)):
                        step_idx = 150 + step * np.random.randint(80, 120)
                        if step_idx < 950:
                            step_strength = np.random.uniform(0.05, 0.15)
                            acc_z[step_idx:step_idx+20] += step_strength
                            
                elif disturbance_type == 'vehicle':
                    # Vehicle passing - longer duration, medium intensity
                    start_idx = np.random.randint(100, 300)
                    duration = np.random.randint(100, 300)
                    intensity = np.random.uniform(0.03, 0.12)
                    
                    # Gradual increase and decrease
                    for j in range(duration):
                        if start_idx + j < 1000:
                            factor = np.sin(np.pi * j / duration) * intensity
                            acc_x[start_idx + j] += factor * 0.8
                            acc_y[start_idx + j] += factor * 0.6
                            acc_z[start_idx + j] += factor
                            
                elif disturbance_type == 'wind':
                    # Wind - low frequency oscillation
                    freq = np.random.uniform(0.1, 0.5)
                    amplitude = np.random.uniform(0.01, 0.05)
                    for j in range(1000):
                        t = j / 100.0
                        wind_factor = amplitude * np.sin(2 * np.pi * freq * t)
                        acc_x[j] += wind_factor
                        acc_y[j] += wind_factor * 0.8
                        
                elif disturbance_type == 'machinery':
                    # Machinery - higher frequency vibration
                    start_idx = np.random.randint(200, 600)
                    duration = np.random.randint(50, 200)
                    freq = np.random.uniform(5, 15)
                    amplitude = np.random.uniform(0.02, 0.08)
                    
                    for j in range(duration):
                        if start_idx + j < 1000:
                            t = j / 100.0
                            vibration = amplitude * np.sin(2 * np.pi * freq * t)
                            acc_x[start_idx + j] += vibration
                            acc_y[start_idx + j] += vibration * 0.9
                            acc_z[start_idx + j] += vibration * 0.7
            
            sensor_data = {'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z}
            features = self.feature_extractor.extract_advanced_features(sensor_data)
            data.append(features)
            labels.append(0)  # Normal
        
        # Generate earthquake samples (20%)
        n_earthquake = n_samples - n_normal
        for i in range(n_earthquake):
            # Start with normal background
            base_std = np.random.uniform(0.02, 0.06)
            acc_x = np.random.normal(0, base_std, 1000)
            acc_y = np.random.normal(0, base_std, 1000)
            acc_z = np.random.normal(1.0, base_std, 1000)
            
            # Add earthquake characteristics with more variety
            earthquake_type = np.random.choice(['local', 'regional', 'distant', 'micro'])
            magnitude = np.random.uniform(0.1, 3.0)  # Simulated magnitude scale
            
            if earthquake_type == 'local':
                # Local earthquake: high frequency, high amplitude, short duration
                start_idx = np.random.randint(200, 400)
                duration = np.random.randint(50, 200)
                amplitude = magnitude * np.random.uniform(0.2, 0.8)
                
                # P-wave arrival
                p_duration = int(duration * 0.3)
                for j in range(p_duration):
                    if start_idx + j < 1000:
                        acc_x[start_idx + j] += np.random.normal(0, amplitude*0.3)
                        acc_y[start_idx + j] += np.random.normal(0, amplitude*0.3)
                        acc_z[start_idx + j] += np.random.normal(0, amplitude*0.4)
                
                # S-wave arrival (higher amplitude)
                s_start = start_idx + p_duration + np.random.randint(5, 30)
                s_duration = duration - p_duration
                for j in range(s_duration):
                    if s_start + j < 1000:
                        acc_x[s_start + j] += np.random.normal(0, amplitude)
                        acc_y[s_start + j] += np.random.normal(0, amplitude)
                        acc_z[s_start + j] += np.random.normal(0, amplitude*1.2)
                
            elif earthquake_type == 'regional':
                # Regional earthquake: medium frequency, medium amplitude, longer duration
                start_idx = np.random.randint(100, 300)
                duration = np.random.randint(200, 500)
                amplitude = magnitude * np.random.uniform(0.1, 0.4)
                
                # Gradual build-up
                for j in range(duration):
                    if start_idx + j < 1000:
                        factor = np.sin(np.pi * j / duration) * amplitude
                        acc_x[start_idx + j] += np.random.normal(0, factor * 0.8)
                        acc_y[start_idx + j] += np.random.normal(0, factor * 0.8)
                        acc_z[start_idx + j] += np.random.normal(0, factor)
            
            elif earthquake_type == 'distant':
                # Distant earthquake: low frequency, low amplitude, very long duration
                start_idx = np.random.randint(50, 200)
                duration = np.random.randint(300, 700)
                amplitude = magnitude * np.random.uniform(0.05, 0.2)
                
                # Low frequency oscillation
                freq = np.random.uniform(0.5, 3.0)  # Hz
                for j in range(duration):
                    if start_idx + j < 1000:
                        t = j / 100.0  # Convert to seconds
                        factor = amplitude * np.sin(2 * np.pi * freq * t) * np.exp(-t/15)
                        acc_x[start_idx + j] += factor
                        acc_y[start_idx + j] += factor
                        acc_z[start_idx + j] += factor*1.1
                        
            else:  # micro earthquake
                # Micro earthquake: very subtle, short duration
                start_idx = np.random.randint(200, 600)
                duration = np.random.randint(20, 80)
                amplitude = magnitude * np.random.uniform(0.02, 0.1)
                
                for j in range(duration):
                    if start_idx + j < 1000:
                        acc_x[start_idx + j] += np.random.normal(0, amplitude)
                        acc_y[start_idx + j] += np.random.normal(0, amplitude)
                        acc_z[start_idx + j] += np.random.normal(0, amplitude*1.1)
            
            # Add realistic noise that varies with intensity
            noise_factor = 1 + magnitude * 0.1
            acc_x += np.random.normal(0, 0.01 * noise_factor, 1000)
            acc_y += np.random.normal(0, 0.01 * noise_factor, 1000)
            acc_z += np.random.normal(0, 0.01 * noise_factor, 1000)
            
            sensor_data = {'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z}
            features = self.feature_extractor.extract_advanced_features(sensor_data)
            data.append(features)
            labels.append(1)  # Earthquake
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(data)
        feature_df = feature_df.fillna(0)  # Handle any NaN values
        
        print(f"Generated {len(feature_df)} samples with {len(feature_df.columns)} features")
        print(f"Class distribution: Normal={np.sum(np.array(labels)==0)}, Earthquake={np.sum(np.array(labels)==1)}")
        
        return feature_df, np.array(labels)
    
    def train_enhanced_model(self, X, y):
        """Train the enhanced ensemble model with hyperparameter optimization"""
        print("Training enhanced ensemble model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Feature scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        print("Performing feature selection...")
        selector = SelectKBest(f_classif, k=min(50, X_train_scaled.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        print(f"Selected {X_train_selected.shape[1]} best features")
        
        # Define base models with optimized parameters
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=self.random_state
        )
        
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('random_forest', rf_model),
                ('gradient_boosting', gb_model),
                ('neural_network', mlp_model)
            ],
            voting='soft'
        )
        
        # Train ensemble
        print("Training ensemble model...")
        ensemble.fit(X_train_selected, y_train)
        
        # Evaluate model
        train_score = ensemble.score(X_train_selected, y_train)
        test_score = ensemble.score(X_test_selected, y_test)
        
        # Get predictions for detailed evaluation
        y_pred = ensemble.predict(X_test_selected)
        y_pred_proba = ensemble.predict_proba(X_test_selected)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation
        print("Performing cross-validation...")
        cv_scores = cross_val_score(ensemble, X_train_selected, y_train, 
                                  cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                                  scoring='roc_auc')
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model and preprocessing objects
        model_data = {
            'model': ensemble,
            'scaler': scaler,
            'feature_selector': selector,
            'feature_names': list(X.columns),
            'selected_features': selector.get_support(),
            'training_metrics': {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'auc_score': auc_score,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std()
            }
        }
        
        model_path = os.path.join(self.output_dir, 'enhanced_ensemble_model.joblib')
        joblib.dump(model_data, model_path)
        print(f"Enhanced model saved to: {model_path}")
        
        # Save feature importance
        self.save_feature_importance(ensemble, X.columns, selector)
        
        # Create evaluation plots
        self.create_evaluation_plots(y_test, y_pred, y_pred_proba)
        
        return ensemble, scaler, selector
    
    def save_feature_importance(self, ensemble, feature_names, selector):
        """Save feature importance analysis"""
        print("Analyzing feature importance...")
        
        # Get feature importance from Random Forest
        rf_model = ensemble.named_estimators_['random_forest']
        
        # Get selected feature names
        selected_features = np.array(feature_names)[selector.get_support()]
        importance_scores = rf_model.feature_importances_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Save to CSV
        importance_path = os.path.join(self.output_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title('Top 20 Most Important Features for Earthquake Detection')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance saved to: {importance_path}")
        print(f"Feature importance plot saved to: {plot_path}")
    
    def create_evaluation_plots(self, y_true, y_pred, y_pred_proba):
        """Create evaluation plots"""
        print("Creating evaluation plots...")
        
        # Set up the plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Prediction Probability Distribution
        axes[1, 0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Normal', color='blue')
        axes[1, 0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Earthquake', color='red')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Probability Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        axes[1, 1].plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curve')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'model_evaluation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plots saved to: {plot_path}")
    
    def run_training(self, use_synthetic_data=True, n_samples=10000):
        """Run the complete training process"""
        print("="*80)
        print("ENHANCED EARTHQUAKE DETECTION MODEL TRAINING")
        print("="*80)
        
        start_time = datetime.now()
        
        if use_synthetic_data:
            # Generate synthetic data
            X, y = self.generate_synthetic_data(n_samples)
        else:
            # TODO: Load real sensor data
            print("Real data loading not implemented yet. Using synthetic data.")
            X, y = self.generate_synthetic_data(n_samples)
        
        # Train model
        model, scaler, selector = self.train_enhanced_model(X, y)
        
        # Save training summary
        training_summary = {
            'training_date': datetime.now().isoformat(),
            'training_duration': str(datetime.now() - start_time),
            'data_samples': len(X),
            'features_total': len(X.columns),
            'features_selected': selector.n_features_in_ if hasattr(selector, 'n_features_in_') else len(X.columns),
            'model_type': 'Enhanced Ensemble (RF + GB + MLP)',
            'synthetic_data': use_synthetic_data
        }
        
        summary_path = os.path.join(self.output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"\nTraining completed in {datetime.now() - start_time}")
        print(f"Training summary saved to: {summary_path}")
        print("="*80)
        
        return model, scaler, selector


def main():
    """Main training function"""
    print("Starting Enhanced Earthquake Detection Model Training...")
    
    trainer = EnhancedModelTrainer()
    
    # Run training with synthetic data
    model, scaler, selector = trainer.run_training(
        use_synthetic_data=True,
        n_samples=10000
    )
    
    print("\n✓ Enhanced model training completed successfully!")
    print("The enhanced model is now ready for real-time earthquake detection.")


if __name__ == "__main__":
    main()
