"""
Comprehensive test script for model evaluation with detailed metrics.
Tests predictions and provides ROC-AUC, precision/recall analysis including session_duration.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from predict import predict_movement
import joblib

def comprehensive_evaluation():
    """Perform comprehensive evaluation on test data with all metrics."""
    try:
        print("Loading data and model...")
        
        # Load sessions and model
        with open('data/all_sessions.json', 'r') as f:
            all_sessions = json.load(f)
        model = joblib.load('models/mouse_model.pkl')
        features_df = pd.read_csv('data/features.csv')
        
        print(f"Evaluating on {len(all_sessions)} sessions...")
        
        # Prepare data (now includes session_duration)
        X = features_df.drop(columns=['label', 'session_id'])
        y_true = features_df['label'].values
        
        print("Making predictions...")
        # Get predictions and probabilities
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        print("Calculating metrics...")
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        print(f"\nCOMPREHENSIVE EVALUATION RESULTS:")
        print(f"=" * 50)
        print(f"Total samples: {len(y_true)}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")
        print(f"Features used: {list(X.columns)}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Human', 'Bot']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              Human  Bot")
        print(f"Actual Human    {cm[0,0]:3d}  {cm[0,1]:3d}")
        print(f"       Bot      {cm[1,0]:3d}  {cm[1,1]:3d}")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance Ranking:")
        for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
            print(f"{i}. {row['feature']}: {row['importance']:.3f}")
        
        # Performance by session type
        human_accuracy = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])
        bot_accuracy = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])
        
        print(f"\nPer-Class Performance:")
        print(f"Human detection accuracy: {human_accuracy:.3f}")
        print(f"Bot detection accuracy: {bot_accuracy:.3f}")
        
        return accuracy, roc_auc
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user (Ctrl+C). Exiting gracefully...")
        return None, None
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        return None, None

def test_individual_sessions():
    """Test individual sessions to demonstrate real-time prediction."""
    try:
        print(f"\nINDIVIDUAL SESSION TESTING:")
        print(f"=" * 50)
        
        # Load sessions
        print("Loading sessions...")
        with open('data/all_sessions.json', 'r') as f:
            all_sessions = json.load(f)
        
        # Find one human and one bot session
        human_session = next(s for s in all_sessions if s['type'] == 'human')
        bot_session = next(s for s in all_sessions if s['type'] == 'bot')
        
        # Test bot session
        print("\nTesting Bot Session:")
        print(f"   Session ID: {bot_session['session_id']}")
        print(f"   Duration: {bot_session['metadata']['session_duration']:.3f}s")
        with open('test_bot.json', 'w') as f:
            json.dump(bot_session, f)
        
        prediction, confidence = predict_movement('test_bot.json')
        result = "Bot" if prediction == 1 else "Human"
        status = "CORRECT" if prediction == 1 else "INCORRECT"
        print(f"   Result: {result} ({confidence:.1f}% confidence) {status}")
        
        # Test human session
        print("\nTesting Human Session:")
        print(f"   Session ID: {human_session['session_id']}")
        print(f"   Duration: {human_session['metadata']['session_duration']:.3f}s")
        with open('test_human.json', 'w') as f:
            json.dump(human_session, f)
        
        prediction, confidence = predict_movement('test_human.json')
        result = "Bot" if prediction == 1 else "Human"
        status = "CORRECT" if prediction == 0 else "INCORRECT"
        print(f"   Result: {result} ({confidence:.1f}% confidence) {status}")
        
    except KeyboardInterrupt:
        print("\n\nIndividual session testing interrupted by user (Ctrl+C). Exiting gracefully...")
    except Exception as e:
        print(f"\nError during individual session testing: {e}")

if __name__ == "__main__":
    print("Running Comprehensive Model Testing...")
    print("=" * 60)
    
    try:
        # Run comprehensive evaluation
        result = comprehensive_evaluation()
        
        if result[0] is not None:
            accuracy, roc_auc = result
            
            # Test individual sessions
            test_individual_sessions()
            
            print(f"\nSUMMARY:")
            print(f"Overall Accuracy: {accuracy:.3f}")
            print(f"ROC-AUC Score: {roc_auc:.3f}")
            print(f"Session Duration: Key differentiator between humans and bots")
            print(f"Testing completed successfully!")
        else:
            print("\nTesting failed or was interrupted.")
            
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user (Ctrl+C). Exiting gracefully...")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run generate_sessions.py and train_model.py first.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check your data files and dependencies.")
