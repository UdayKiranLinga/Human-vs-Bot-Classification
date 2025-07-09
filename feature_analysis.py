"""
Feature Analysis Visualization for Behavioral CAPTCHA Detector.
Creates comprehensive analysis of the 4 selected features and their discrimination power.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_feature_analysis():
    """Create comprehensive feature analysis showing why these 4 features work."""
    
    # Load data
    df = pd.read_csv('data/features.csv')
    
    # Create optimized figure layout (3x2 instead of 2x3)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Feature Analysis: Why These 4 Features Detect Bots', fontsize=16, fontweight='bold')
    
    features = ['std_speed', 'max_speed', 'num_points', 'session_duration']
    colors = ['#3498db', '#e74c3c']  # Blue for human, red for bot
    
    # Feature distributions (first 4 subplots)
    for i, feature in enumerate(features):
        row, col = i // 2, i % 2
        
        human_data = df[df['label'] == 0][feature]
        bot_data = df[df['label'] == 1][feature]
        
        # Histogram
        axes[row, col].hist(human_data, bins=15, alpha=0.7, label='Human', color=colors[0], density=True)
        axes[row, col].hist(bot_data, bins=15, alpha=0.7, label='Bot', color=colors[1], density=True)
        
        # Statistics
        human_mean, human_std = np.mean(human_data), np.std(human_data)
        bot_mean, bot_std = np.mean(bot_data), np.std(bot_data)
        
        axes[row, col].axvline(human_mean, color=colors[0], linestyle='--', linewidth=2, alpha=0.8)
        axes[row, col].axvline(bot_mean, color=colors[1], linestyle='--', linewidth=2, alpha=0.8)
        
        # Clean title without discrimination percentage
        axes[row, col].set_title(f'{feature.replace("_", " ").title()}\nHuman vs Bot Distribution', fontweight='bold')
        axes[row, col].set_xlabel(feature.replace('_', ' ').title())
        axes[row, col].set_ylabel('Density')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    # Feature importance bar chart (bottom left)
    importance_data = {
        'num_points': 54.4,
        'max_speed': 23.4,
        'std_speed': 13.3,
        'session_duration': 8.9
    }
    
    ax = axes[2, 0]
    bars = ax.bar(range(len(importance_data)), list(importance_data.values()), 
                  color=['#2ecc71', '#f39c12', '#9b59b6', '#e67e22'])
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance (%)')
    ax.set_title('Random Forest\nFeature Importance')
    ax.set_xticks(range(len(importance_data)))
    ax.set_xticklabels([k.replace('_', '\n') for k in importance_data.keys()], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, importance_data.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Summary statistics table (bottom right)
    ax_summary = axes[2, 1]
    ax_summary.axis('off')
    
    # Calculate discrimination summary
    discrimination_scores = []
    for feature in features:
        human_data = df[df['label'] == 0][feature]
        bot_data = df[df['label'] == 1][feature]
        human_mean, human_std = np.mean(human_data), np.std(human_data)
        bot_mean, bot_std = np.mean(bot_data), np.std(bot_data)
        discrimination = abs(human_mean - bot_mean) / (human_std + bot_std) * 100
        discrimination_scores.append(discrimination)
    
    # Create summary table
    summary_text = "DISCRIMINATION POWER SUMMARY\n\n"
    for i, (feature, score) in enumerate(zip(features, discrimination_scores)):
        rank = i + 1
        summary_text += f"{rank}. {feature.replace('_', ' ').title()}: {score:.1f}%\n"
    
    summary_text += f"\nAverage Discrimination: {np.mean(discrimination_scores):.1f}%\n"
    summary_text += f"Best Feature: {features[np.argmax(discrimination_scores)].replace('_', ' ').title()}\n"
    summary_text += f"Dataset: {len(df)} sessions\n"
    summary_text += f"Human sessions: {len(df[df['label'] == 0])}\n"
    summary_text += f"Bot sessions: {len(df[df['label'] == 1])}"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/feature_analysis.png', dpi=300, bbox_inches='tight')
    print(" Feature analysis saved to plots/feature_analysis.png")
    plt.show()
    
    # Print insights
    print("\n Feature Analysis Insights:")
    print("=" * 50)
    
    for feature in features:
        human_data = df[df['label'] == 0][feature]
        bot_data = df[df['label'] == 1][feature]
        
        human_mean, human_std = np.mean(human_data), np.std(human_data)
        bot_mean, bot_std = np.mean(bot_data), np.std(bot_data)
        
        discrimination = abs(human_mean - bot_mean) / (human_std + bot_std) * 100
        
        print(f"\n{feature.replace('_', ' ').title()}:")
        print(f"  Human: μ={human_mean:.1f}, σ={human_std:.1f}")
        print(f"  Bot:   μ={bot_mean:.1f}, σ={bot_std:.1f}")
        print(f"  Discrimination: {discrimination:.1f}%")

if __name__ == "__main__":
    create_feature_analysis()
