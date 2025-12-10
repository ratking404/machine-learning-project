"""
Weather Prediction Machine Learning Model
Author: AI Assistant
Date: 2024
Description: Predicts weather conditions using Seattle weather data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class WeatherPredictor:
    """
    A machine learning model for predicting weather conditions.
    Supports multiple algorithms and provides comprehensive evaluation.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the Weather Predictor
        
        Parameters:
        -----------
        model_type : str
            Type of model to use: 'logistic', 'decision_tree', or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = ['precipitation', 'temp_max', 'temp_min', 'wind']
        self.target_column = 'weather'
        
        # Model mapping
        self.models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
    
    def load_and_preprocess_data(self, file_path='seattle-weather.csv'):
        """
        Load and preprocess the weather data
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
            
        Returns:
        --------
        tuple: (X_train, X_test, y_train, y_test, df)
        """
        print("📊 Loading data...")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Display dataset info
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print("\n📈 Dataset Overview:")
        print(df.head())
        print("\n📋 Dataset Info:")
        print(df.info())
        print("\n📊 Basic Statistics:")
        print(df.describe())
        
        # Check for missing values
        print("\n🔍 Missing values:")
        print(df.isnull().sum())
        
        # Create date features
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Encode target variable
        df['weather_encoded'] = self.label_encoder.fit_transform(df[self.target_column])
        
        # Prepare features
        feature_cols = self.feature_columns + ['month', 'day_of_year']
        X = df[feature_cols]
        y = df['weather_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n✅ Data preprocessing complete!")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Testing samples: {X_test.shape[0]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, df
    
    def train_model(self, X_train, y_train):
        """
        Train the selected machine learning model
        
        Parameters:
        -----------
        X_train : array
            Training features
        y_train : array
            Training labels
        """
        print(f"\n🤖 Training {self.model_type} model...")
        
        # Select model
        if self.model_type not in self.models:
            print(f"⚠️  Unknown model type: {self.model_type}. Using Random Forest.")
            self.model_type = 'random_forest'
        
        self.model = self.models[self.model_type]
        
        # Train model
        self.model.fit(X_train, y_train)
        print(f"✅ Model training complete!")
        
        # Feature importance for tree-based models
        if self.model_type in ['decision_tree', 'random_forest']:
            self._display_feature_importance()
    
    def _display_feature_importance(self):
        """Display feature importance for tree-based models"""
        feature_cols = self.feature_columns + ['month', 'day_of_year']
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n📊 Feature Importance:")
        for i in range(len(feature_cols)):
            print(f"   {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        
        Parameters:
        -----------
        X_test : array
            Testing features
        y_test : array
            Testing labels
            
        Returns:
        --------
        dict: Evaluation metrics
        """
        print("\n📈 Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Decode labels for reporting
        y_test_decoded = self.label_encoder.inverse_transform(y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        
        # Generate classification report
        report = classification_report(
            y_test_decoded, 
            y_pred_decoded,
            output_dict=True
        )
        
        # Display results
        print(f"✅ Model Accuracy: {accuracy:.4f}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test_decoded, y_pred_decoded))
        
        # Return metrics
        metrics = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test_decoded,
            'y_pred': y_pred_decoded
        }
        
        return metrics
    
    def predict_weather(self, features):
        """
        Predict weather for new input
        
        Parameters:
        -----------
        features : list or array
            Input features: [precipitation, temp_max, temp_min, wind, month, day_of_year]
            
        Returns:
        --------
        str: Predicted weather condition
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Convert to numpy array and scale
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Make prediction
        prediction_encoded = self.model.predict(features_scaled)
        prediction = self.label_encoder.inverse_transform(prediction_encoded)
        
        # Get prediction probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            class_probabilities = dict(zip(
                self.label_encoder.classes_, 
                probabilities
            ))
            return prediction[0], class_probabilities
        
        return prediction[0], None
    
    def save_model(self, filepath='weather_model.pkl'):
        """Save the trained model and preprocessing objects"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath='weather_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        print(f"📂 Model loaded from {filepath}")
    
    def plot_results(self, metrics, save_path=None):
        """
        Plot evaluation results
        
        Parameters:
        -----------
        metrics : dict
            Evaluation metrics from evaluate_model
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Confusion Matrix
        cm = metrics['confusion_matrix']
        classes = self.label_encoder.classes_
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes, ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Plot 2: Accuracy by Weather Type
        report_df = pd.DataFrame(metrics['report']).transpose()
        if 'accuracy' in report_df.index:
            report_df = report_df.drop('accuracy')
        if 'macro avg' in report_df.index:
            report_df = report_df.drop('macro avg')
        if 'weighted avg' in report_df.index:
            report_df = report_df.drop('weighted avg')
        
        report_df['precision'].plot(kind='bar', ax=axes[1], color='skyblue')
        axes[1].set_title('Precision by Weather Type')
        axes[1].set_xlabel('Weather Type')
        axes[1].set_ylabel('Precision')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()


def main():
    """
    Main function to run the weather prediction model
    """
    print("=" * 60)
    print("🌤️  WEATHER PREDICTION MACHINE LEARNING MODEL")
    print("=" * 60)
    
    # Initialize predictor
    predictor = WeatherPredictor(model_type='random_forest')
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, df = predictor.load_and_preprocess_data()
    
    # Train model
    predictor.train_model(X_train, y_train)
    
    # Evaluate model
    metrics = predictor.evaluate_model(X_test, y_test)
    
    # Make sample prediction
    print("\n🔮 Sample Prediction:")
    sample_features = [5.0, 15.0, 8.0, 3.0, 6, 150]  # Example: June
    prediction, probabilities = predictor.predict_weather(sample_features)
    print(f"   Input: Precipitation=5.0, Temp Max=15.0, Temp Min=8.0, Wind=3.0, Month=6")
    print(f"   Predicted Weather: {prediction}")
    if probabilities:
        print("   Prediction Probabilities:")
        for weather, prob in probabilities.items():
            print(f"     {weather}: {prob:.2%}")
    
    # Plot results
    predictor.plot_results(metrics, save_path='model_results.png')
    
    # Save model
    predictor.save_model('weather_model.pkl')
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    
    # Display weather distribution
    print("\n🌧️  Weather Distribution in Dataset:")
    weather_counts = df['weather'].value_counts()
    for weather, count in weather_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {weather}: {count} days ({percentage:.1f}%)")
    
    return predictor


if __name__ == "__main__":
    # Run the main function
    trained_predictor = main()