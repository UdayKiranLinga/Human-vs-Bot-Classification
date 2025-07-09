#!/usr/bin/env python3
"""
Behavioral CAPTCHA Detector - Complete Demo
==========================================

This script demonstrates the complete workflow:
1. Generate synthetic mouse movement sessions
2. Extract behavioral features
3. Train Random Forest classifier
4. Evaluate model performance
5. Generate comprehensive visualizations

Run this script to see the entire system in action.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def run_script(script_name, description):
    """Run a script and display progress."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f" {script_name} completed successfully!")
            if result.stdout:
                print(result.stdout)
        else:
            print(f" Error in {script_name}:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f" Failed to run {script_name}: {e}")
        return False
    
    return True

def main():
    """Run the complete behavioral CAPTCHA detector workflow."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║          🤖 BEHAVIORAL CAPTCHA DETECTOR DEMO                 ║
    ║                                                              ║
    ║  This demo shows how mouse movement patterns can distinguish ║
    ║  between human users and automated bots using ML.           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check if required files exist
    required_files = [
        'generate_sessions.py',
        'extract_features.py', 
        'train_model.py',
        'test_predictions.py',
        'visualize_data.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f" Missing required files: {', '.join(missing_files)}")
        return
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("\n📂 Project structure verified - all directories ready!")
    
    # Step 1: Generate session data
    if not run_script('generate_sessions.py', 
                     "Step 1: Generating synthetic mouse movement sessions"):
        return
    
    # Step 2: Extract features
    if not run_script('extract_features.py',
                     "Step 2: Extracting behavioral features from mouse data"):
        return
    
    # Step 3: Train model
    if not run_script('train_model.py',
                     "Step 3: Training Random Forest classifier"):
        return
    
    # Step 4: Test predictions
    if not run_script('test_predictions.py',
                     "Step 4: Evaluating model performance"):
        return
    
    # Step 5: Create visualizations
    if not run_script('visualize_data.py',
                     "Step 5: Creating comprehensive visualizations"):
        return
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 BEHAVIORAL CAPTCHA DETECTOR DEMO COMPLETED!")
    print(f"{'='*60}")
    
    print("\n📊 Generated Files:")
    print("  • data/all_sessions.json - Combined session data")
    print("  • data/features.csv - ML-ready feature matrix")
    print("  • models/mouse_model.pkl - Trained Random Forest model")
    print("  • plots/*.png - Comprehensive visualizations")
    
    print("\n🔍 Key Results:")
    
    # Check if model file exists and show basic info
    if os.path.exists('models/mouse_model.pkl'):
        print("   Model trained successfully")
        
    if os.path.exists('data/features.csv'):
        import pandas as pd
        df = pd.read_csv('data/features.csv')
        print(f"   Dataset: {df.shape[0]} sessions, {df.shape[1]-2} features")
        print(f"   Features: {', '.join(df.columns[:-2])}")
    
    if os.path.exists('plots'):
        plot_files = [f for f in os.listdir('plots') if f.endswith('.png')]
        print(f"   Visualizations: {len(plot_files)} plots generated")
    
    print("\n Next Steps:")
    print("  • Open plots/*.png to view analysis results")
    print("  • Use predict.py to test on new mouse movement data")
    print("  • Modify parameters in train_model.py to experiment")
    print("  • Check web/index.html for data collection interface")
    
    print("\n This demo showcases how behavioral biometrics can")
    print("   distinguish human users from automated bots through")
    print("   natural mouse movement analysis!")

if __name__ == "__main__":
    main()
