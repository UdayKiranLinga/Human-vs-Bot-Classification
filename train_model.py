"""
Machine Learning Model Training for Behavioral CAPTCHA Detection.
Trains a Random Forest classifier to distinguish between human and bot mouse movements.
"""

import pandas as pd
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

def train_model():
    """Train a Random Forest classifier for bot detection."""
    try:
        # Load feature data
        print("Loading feature data...")
        df = pd.read_csv('data/features.csv')
        print("Dataset shape:", df.shape)
        print("\nFeature statistics:")
        print(df.describe())
        
        # Prepare features and labels (now includes session_duration)
        print("\nPreparing features and labels...")
        X = df.drop(columns=['label', 'session_id'])
        y = df['label']
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        print(f"\nTrain set: {len(X_train)}, Test set: {len(X_test)}")
        print(f"Features: {list(X.columns)}")
        
        # Train Random Forest
        print("\nTraining Random Forest classifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        print("Training completed successfully!")
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Human', 'Bot']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Save model
        print("\nSaving model...")
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/mouse_model.pkl')
        print("Model saved as 'models/mouse_model.pkl'")
        
        # Simple inference time measurement
        print("\nMeasuring inference time...")
        start_time = time.time()
        _ = model.predict(X_test[:1])
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Average inference time: {inference_time:.3f} ms per prediction")
        
        return model, accuracy
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user (Ctrl+C). Exiting gracefully...")
        return None, None
    except FileNotFoundError as e:
        print(f"\nError: Required file not found - {e}")
        print("Please ensure you have run generate_sessions.py and extract_features.py first.")
        return None, None
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        print("Please check your data files and dependencies.")
        return None, None

if __name__ == "__main__":
    print("Starting Behavioral CAPTCHA Model Training...")
    print("=" * 50)
    
    result = train_model()
    
    if result[0] is not None:
        model, accuracy = result
        print("\nTraining completed successfully!")
        print(f"Final accuracy: {accuracy:.3f}")
    else:
        print("\nTraining failed or was interrupted.")
        print("Please check the error messages above.")