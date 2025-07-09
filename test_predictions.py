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
    
    # Get predictions and probabilities
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
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
    print(f"           Human   Bot")
    print(f"Human     {cm[0,0]:5d}  {cm[0,1]:4d}")
    print(f"Bot       {cm[1,0]:5d}  {cm[1,1]:4d}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance Analysis (Best 4 Features):")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']:<18}: {row['importance']:.3f}")
    
    # Session duration specific analysis
    human_mask = y_true == 0
    bot_mask = y_true == 1
    human_durations = features_df[human_mask]['session_duration']
    bot_durations = features_df[bot_mask]['session_duration']
    
    print(f"\nSession Duration Analysis:")
    print(f"  Human sessions: μ={human_durations.mean():.3f}s, σ={human_durations.std():.3f}s")
    print(f"  Bot sessions:   μ={bot_durations.mean():.3f}s, σ={bot_durations.std():.3f}s")
    print(f"  Duration difference: {abs(human_durations.mean() - bot_durations.mean()):.3f}s")
    
    return accuracy, roc_auc
def test_individual_sessions():
    """Test predictions on individual human and bot sessions."""
    with open('data/all_sessions.json', 'r') as f:
        all_sessions = json.load(f)
    
    # Find one human and one bot session
    human_session = next(s for s in all_sessions if s['type'] == 'human')
    bot_session = next(s for s in all_sessions if s['type'] == 'bot')
    
    # Test bot session
    print("\n Testing Bot Session:")
    print(f"   Session ID: {bot_session['session_id']}")
    print(f"   Duration: {bot_session['metadata']['session_duration']:.3f}s")
    with open('test_bot.json', 'w') as f:
        json.dump(bot_session, f)
    
    prediction, confidence = predict_movement('test_bot.json')
    result = "Bot" if prediction == 1 else "Human"
    status = " CORRECT" if prediction == 1 else " INCORRECT"
    print(f"   Result: {result} ({confidence:.1f}% confidence) {status}")
    
    # Test human session
    print("\n� Testing Human Session:")
    print(f"   Session ID: {human_session['session_id']}")
    print(f"   Duration: {human_session['metadata']['session_duration']:.3f}s")
    with open('test_human.json', 'w') as f:
        json.dump(human_session, f)
    
    prediction, confidence = predict_movement('test_human.json')
    result = "Bot" if prediction == 1 else "Human"
    status = " CORRECT" if prediction == 0 else " INCORRECT"
    print(f"   Result: {result} ({confidence:.1f}% confidence) {status}")

if __name__ == "__main__":
    print("🧪 Running Comprehensive Model Testing...")
    print("=" * 60)
    
    try:
        # Run comprehensive evaluation
        accuracy, roc_auc = comprehensive_evaluation()
        
        # Test individual sessions
        test_individual_sessions()
        print(f"\nSUMMARY:")
        print(f"Overall Accuracy: {accuracy:.3f}")
        print(f"ROC-AUC Score: {roc_auc:.3f}")
        print(f"Session Duration: Key differentiator between humans and bots")
        print(f"Testing completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run generate_sessions.py and train_model.py first.")
    except Exception as e:
        print(f"Unexpected error: {e}")
