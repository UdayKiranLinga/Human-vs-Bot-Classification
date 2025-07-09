"""
Feature extraction module for behavioral CAPTCHA detector.
Converts raw mouse movement data into meaningful features for ML classification.
"""

import json
import numpy as np
import pandas as pd
import os

def extract_features(data, session_duration=None):
    """
    Extract behavioral features from mouse movement data.
    
    Args:
        data: List of mouse movement points with x, y coordinates and timestamps
        session_duration: Optional session duration in seconds
        
    Returns:
        Dictionary containing extracted features for ML classification
    """
    # Extract coordinates and timestamps
    xs, ys, ts = [p['x'] for p in data], [p['y'] for p in data], [p['t'] for p in data]

    # Calculate speeds between consecutive points
    speeds = []
    for i in range(1, len(xs)):
        dx, dy, dt = xs[i] - xs[i-1], ys[i] - ys[i-1], (ts[i] - ts[i-1]) / 1000
        if dt > 0:  # avoid division by zero
            speeds.append(((dx**2 + dy**2)**0.5) / dt)

    # Calculate session duration if not provided
    if session_duration is None:
        session_duration = (ts[-1] - ts[0]) / 1000 if len(ts) > 1 else 0

    return {
        'std_speed': np.std(speeds),            # Human: high variance, Bot: low variance (65% difference)
        'max_speed': np.max(speeds),            # Peak movement speed (65% difference)
        'num_points': len(xs),                  # Total data points collected (58% difference)
        'session_duration': session_duration    # Key differentiator: Bots faster, Humans slower (45% difference)
    }

def load_data(file, label):
    """Load mouse movement data from JSON file and extract features."""
    with open(file, 'r') as f:
        raw = json.load(f)
    features = extract_features(raw)
    features['label'] = label
    return features

# Main execution
if __name__ == "__main__":
    sessions_file = 'data/all_sessions.json'
    
    if not os.path.exists(sessions_file):
        print(f"Error: {sessions_file} not found. Please run generate_sessions.py first.")
        exit(1)
    
    print("Processing sessions to extract features...")
    
    with open(sessions_file, 'r') as f:
        sessions = json.load(f)
    
    features_data = []
    for session in sessions:
        movements = session['movements']
        session_type = session['type']
        session_duration = session['metadata'].get('session_duration', None)
        
        # Extract features including session_duration
        features = extract_features(movements, session_duration)
        features['label'] = 1 if session_type == 'bot' else 0
        features['session_id'] = session['session_id']
        features_data.append(features)
    
    # Create DataFrame and save
    df = pd.DataFrame(features_data)
    df.to_csv('data/features.csv', index=False)
    
    print(f"Features extracted and saved to data/features.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-2])}")
    
    # Feature statistics
    print(f"\nFeature Statistics by Class:")
    feature_cols = ['std_speed', 'max_speed', 'num_points', 'session_duration']
    human_stats = df[df['label'] == 0][feature_cols].mean()
    bot_stats = df[df['label'] == 1][feature_cols].mean()
    
    print("\nHuman vs Bot Feature Comparison (Top 4 Features):")
    for feature in feature_cols:
        print(f"  {feature}: Human={human_stats[feature]:.3f}, Bot={bot_stats[feature]:.3f}")
    
    print(f"\nSample Features (first 5 rows):")
    print(df.head())

