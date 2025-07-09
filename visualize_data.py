"""
Comprehensive visualization suite for behavioral CAPTCHA detection.
Creates professional plots for model evaluation, movement patterns, and feature analysis.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os



def plot_session_duration_analysis(sessions_file='data/all_sessions.json', save_path='plots/session_duration_analysis.png'):
    """Create comprehensive session duration analysis with larger, clearer visualization."""
    with open(sessions_file, 'r') as f:
        sessions = json.load(f)
    
    human_durations = [s['metadata']['session_duration'] for s in sessions if s['type'] == 'human']
    bot_durations = [s['metadata']['session_duration'] for s in sessions if s['type'] == 'bot']
    
    # Create larger figure with better layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Large histogram comparison
    ax1 = fig.add_subplot(gs[0, :2])
    bins = np.linspace(0, max(max(human_durations), max(bot_durations)), 20)
    ax1.hist(human_durations, bins=bins, alpha=0.7, label='Human Sessions', color='#3498db', edgecolor='black')
    ax1.hist(bot_durations, bins=bins, alpha=0.7, label='Bot Sessions', color='#e74c3c', edgecolor='black')
    ax1.set_xlabel('Session Duration (seconds)', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.set_title('Session Duration Distribution\nBots Complete Tasks 1.9x Faster', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = fig.add_subplot(gs[0, 2])
    bp = ax2.boxplot([human_durations, bot_durations], labels=['Human', 'Bot'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax2.set_ylabel('Session Duration (seconds)', fontsize=14)
    ax2.set_title('Duration Variability\nComparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Statistical summary
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    human_mean, human_std = np.mean(human_durations), np.std(human_durations)
    bot_mean, bot_std = np.mean(bot_durations), np.std(bot_durations)
    
    stats_text = f"""
    STATISTICAL SUMMARY
    
    Human Sessions:
    • Mean: {human_mean:.2f} seconds
    • Std Dev: {human_std:.2f} seconds
    • Count: {len(human_durations)} sessions
    
    Bot Sessions:
    • Mean: {bot_mean:.2f} seconds
    • Std Dev: {bot_std:.2f} seconds
    • Count: {len(bot_durations)} sessions
    
    Key Insights:
    • Bots are {human_mean/bot_mean:.1f}x faster
    • Humans show {human_std/bot_std:.1f}x more variability
    • Discrimination: {abs(human_mean-bot_mean)/(human_std+bot_std)*100:.1f}% power
    """
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 4. Violin plot
    ax4 = fig.add_subplot(gs[1, 1])
    parts = ax4.violinplot([human_durations, bot_durations], positions=[1, 2], showmeans=True)
    parts['bodies'][0].set_facecolor('#3498db')
    parts['bodies'][1].set_facecolor('#e74c3c')
    ax4.set_xticks([1, 2])
    ax4.set_xticklabels(['Human', 'Bot'])
    ax4.set_ylabel('Session Duration (seconds)', fontsize=14)
    ax4.set_title('Distribution Shape\nAnalysis', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Timeline scatter
    ax5 = fig.add_subplot(gs[1, 2])
    human_x = range(len(human_durations))
    bot_x = range(len(human_durations), len(human_durations) + len(bot_durations))
    
    ax5.scatter(human_x, human_durations, color='#3498db', alpha=0.7, s=40, label='Human')
    ax5.scatter(bot_x, bot_durations, color='#e74c3c', alpha=0.7, s=40, label='Bot')
    ax5.axhline(y=human_mean, color='#3498db', linestyle='--', alpha=0.8)
    ax5.axhline(y=bot_mean, color='#e74c3c', linestyle='--', alpha=0.8)
    ax5.set_xlabel('Session Index', fontsize=14)
    ax5.set_ylabel('Duration (seconds)', fontsize=14)
    ax5.set_title('Session Timeline\nPatterns', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    fig.suptitle('Session Duration Analysis: Critical Bot Detection Feature', fontsize=20, fontweight='bold')
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Session duration analysis saved to {save_path}")
    plt.show()

def plot_movement_patterns(sessions_file='data/all_sessions.json', save_path='plots/movement_patterns.png'):
    """Plot movement patterns comparing human vs bot behavior."""
    with open(sessions_file, 'r') as f:
        sessions = json.load(f)
    
    # Select first 3 sessions of each type
    human_sessions = [s for s in sessions if s['type'] == 'human'][:3]
    bot_sessions = [s for s in sessions if s['type'] == 'bot'][:3]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Mouse Movement Behavioral Analysis: Human vs Bot Detection', fontsize=18, fontweight='bold')
    
    for i, session in enumerate(human_sessions + bot_sessions):
        row, col = i // 3, i % 3
        movements = session['movements']
        xs, ys = [m['x'] for m in movements], [m['y'] for m in movements]
        
        # Plot with time-based color gradient
        colors = range(len(xs))
        axes[row, col].scatter(xs, ys, c=colors, cmap='viridis', alpha=0.7, s=20)
        axes[row, col].plot(xs, ys, alpha=0.3, linewidth=1, color='gray')
        
        # Mark start/end points
        axes[row, col].scatter(xs[0], ys[0], color='green', s=100, marker='o', label='Start', zorder=5)
        axes[row, col].scatter(xs[-1], ys[-1], color='red', s=100, marker='x', label='End', zorder=5)
        
        # Calculate average speed
        avg_speed = sum((((movements[j+1]['x'] - movements[j]['x'])**2 + (movements[j+1]['y'] - movements[j]['y'])**2)**0.5) / 
                       max(1, movements[j+1]['t'] - movements[j]['t']) for j in range(len(movements)-1)) / max(1, len(movements)-1)
        
        session_type = session['type'].title()
        duration = session['metadata'].get('session_duration', 0)
        axes[row, col].set_title(f"{session_type} Pattern\n{len(movements)} pts • {duration:.2f}s • Speed: {avg_speed:.1f}", 
                               fontweight='bold' if session_type == 'Human' else 'normal')
        axes[row, col].set_xlabel('X Coordinate')
        axes[row, col].set_ylabel('Y Coordinate')
        axes[row, col].legend(fontsize=8)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_xlim(0, 800)
        axes[row, col].set_ylim(0, 600)
        
        # Row labels
        if col == 0:
            row_label = "Human Patterns" if row == 0 else "Bot Patterns"
            axes[row, col].text(-0.15, 0.5, row_label, transform=axes[row, col].transAxes,
                              fontsize=14, fontweight='bold', rotation=90, verticalalignment='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, left=0.08)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Movement patterns saved to {save_path}")
    plt.show()

def plot_session_duration_analysis(sessions_file='data/all_sessions.json', save_path='plots/session_duration_analysis.png'):
    """Plot session duration comparison between humans and bots."""
    with open(sessions_file, 'r') as f:
        sessions = json.load(f)
    
    human_durations = [s['metadata']['session_duration'] for s in sessions if s['type'] == 'human']
    bot_durations = [s['metadata']['session_duration'] for s in sessions if s['type'] == 'bot']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram comparison
    ax1.hist(human_durations, bins=15, alpha=0.7, label='Human', color='blue', edgecolor='black')
    ax1.hist(bot_durations, bins=15, alpha=0.7, label='Bot', color='red', edgecolor='black')
    ax1.set_xlabel('Session Duration (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Session Duration Distribution\nKey Differentiator: Bots are Faster & More Consistent')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax2.boxplot([human_durations, bot_durations], labels=['Human', 'Bot'], patch_artist=True,
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Session Duration (seconds)')
    ax2.set_title('Duration Variability\nHumans: More Variable, Bots: Consistent')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    human_mean, human_std = np.mean(human_durations), np.std(human_durations)
    bot_mean, bot_std = np.mean(bot_durations), np.std(bot_durations)
    
    stats_text = f"Human: μ={human_mean:.2f}s, σ={human_std:.2f}s\nBot: μ={bot_mean:.2f}s, σ={bot_std:.2f}s"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle('Session Duration Analysis: Critical Bot Detection Feature', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Session duration analysis saved to {save_path}")
    plt.show()

def create_essential_visualizations():
    """Create essential visualizations for the project."""
    try:
        print(" Creating essential visualizations...")
        
        # 1. Movement patterns
        print("1. Creating movement patterns comparison...")
        plot_movement_patterns()
        
        # 2. Session duration analysis
        print("2. Creating session duration analysis...")
        plot_session_duration_analysis()
        
        print("Essential visualizations completed!")
        
    except FileNotFoundError as e:
        print(f"Error creating visualizations: {e}")
    except ImportError as e:
        print(f" Missing required library: {e}")

if __name__ == "__main__":
    create_essential_visualizations()
