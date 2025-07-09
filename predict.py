"""
Prediction module for behavioral CAPTCHA detector.
Uses trained ML model to classify mouse movements as human or bot.
"""

import joblib
import json
import numpy as np
import os
import warnings
from extract_features import extract_features

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

def predict_movement(data_file, model_file='mouse_model.pkl'):
    """Predict if mouse movement data is from human or bot."""
    # Load trained model
    try:
        model = joblib.load(f'models/{model_file}')
        print(f"Model loaded from models/{model_file}")
    except FileNotFoundError:
        print(f"Error: Model file models/{model_file} not found. Please train the model first.")
        return None, None
    
    # Load test data
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        print(f"Test data loaded from {data_file}")
    except FileNotFoundError:
        print(f"Error: Data file {data_file} not found.")
        return None, None
    
    # Extract features (handle both old and new formats)
    if 'movements' in data:
        movements = data['movements']
        session_duration = data.get('metadata', {}).get('session_duration', None)
    else:
        movements = data
        session_duration = None
    
    # Validate movements data
    if not movements or len(movements) < 2:
        print("Error: Invalid movement data: Need at least 2 movement points")
        return None, None
    
    features = extract_features(movements, session_duration)
    
    # Validate features
    required_features = ['std_speed', 'max_speed', 'num_points', 'session_duration']
    for feature in required_features:
        if feature not in features:
            print(f"Error: Missing required feature: {feature}")
            return None, None
    
    print(f"\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.3f}")
    
    # Prepare feature vector for prediction (best 4 features)
    X = [[features['std_speed'], features['max_speed'], 
          features['num_points'], features['session_duration']]]
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    # Get confidence scores
    human_confidence, bot_confidence = probability[0] * 100, probability[1] * 100
    
    print(f"\nPrediction Results:")
    print(f"  Classification: {'Bot' if prediction == 1 else 'Human'}")
    print(f"  Human Confidence: {human_confidence:.1f}%")
    print(f"  Bot Confidence: {bot_confidence:.1f}%")
    
    return prediction, max(human_confidence, bot_confidence)
if __name__ == "__main__":
    # Create test data file using one of the bot sessions
    test_file = 'test_data.json'
    if not os.path.exists(test_file):
        try:
            with open('data/all_sessions.json', 'r') as f:
                all_sessions = json.load(f)
            
            # Find the first bot session
            bot_session = next((s for s in all_sessions if s['type'] == 'bot'), None)
            
            if bot_session:
                with open(test_file, 'w') as f:
                    json.dump(bot_session, f)
                print(f"Created {test_file} for testing using bot session")
            else:
                print("Error: No bot sessions found in data/all_sessions.json")
                exit(1)
                
        except FileNotFoundError:
            print("Error: data/all_sessions.json not found. Please generate sessions first.")
            exit(1)
    
    # Make prediction
    prediction, confidence = predict_movement(test_file)
    
    if prediction is not None:
        result = "Bot" if prediction == 1 else "Human"
        print(f"\nFinal Result: {result} (Confidence: {confidence:.1f}%)")
    else:
        print("\nPrediction failed. Please check if model is trained and data files exist.")
