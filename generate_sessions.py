"""
Generate comprehensive mouse movement sessions for behavioral CAPTCHA detection.
Creates 50 bot sessions and 50 human sessions with realistic patterns and metadata.
"""

import json
import random
import math
import time
from datetime import datetime, timedelta

def generate_human_session(session_id, metadata=None):
    """Generate realistic human mouse movement data with natural variations."""
    data = []
    x, y = random.randint(50, 300), random.randint(50, 300)
    start_time = int(time.time() * 1000) + random.randint(-10000, 10000)
    
    # Human characteristics: variable movement, pauses, direction changes
    num_points = random.randint(80, 200)
    pause_prob, direction_prob = 0.1, 0.15
    current_time = start_time
    velocity_x, velocity_y = random.uniform(-2, 2), random.uniform(-2, 2)
    
    for i in range(num_points):
        # Natural variations and tremor
        if random.random() < direction_prob:
            velocity_x += random.uniform(-1, 1)
            velocity_y += random.uniform(-1, 1)
        
        jitter_x, jitter_y = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
        x += velocity_x + jitter_x
        y += velocity_y + jitter_y
        x, y = max(0, min(800, x)), max(0, min(600, y))
        
        # Human-like timing with pauses
        time_delta = random.randint(200, 800) if random.random() < pause_prob else random.randint(8, 50)
        current_time += time_delta
        
        data.append({'x': round(x), 'y': round(y), 't': current_time})
    
    # Calculate session duration based on actual movement times
    session_duration = (current_time - start_time) / 1000  # Convert to seconds
    
    session = {
        'session_id': f'human_{session_id:03d}',
        'type': 'human',
        'metadata': metadata or generate_session_metadata('human', session_duration),
        'movements': data
    }
    
    return session

def generate_bot_session(session_id, metadata=None):
    """Generate bot-like mouse movement data with mechanical patterns."""
    data = []
    pattern_type = random.choice(['linear', 'curve', 'step', 'zigzag'])
    x, y = random.randint(100, 200), random.randint(100, 200)
    start_time = int(time.time() * 1000) + random.randint(-10000, 10000)
    
    # Bot characteristics: shorter paths, consistent timing
    num_points = random.randint(30, 80)
    time_interval = random.randint(50, 150)
    
    if pattern_type == 'linear':
        end_x, end_y = random.randint(300, 600), random.randint(300, 500)
        for i in range(num_points):
            progress = i / (num_points - 1)
            curr_x = int(x + (end_x - x) * progress)
            curr_y = int(y + (end_y - y) * progress)
            data.append({'x': curr_x, 'y': curr_y, 't': start_time + i * time_interval})
    
    elif pattern_type == 'curve':
        control_x, control_y = random.randint(200, 400), random.randint(200, 400)
        end_x, end_y = random.randint(400, 600), random.randint(300, 500)
        for i in range(num_points):
            t = i / (num_points - 1)
            curve_x = (1-t)**2 * x + 2*(1-t)*t * control_x + t**2 * end_x
            curve_y = (1-t)**2 * y + 2*(1-t)*t * control_y + t**2 * end_y
            data.append({'x': int(curve_x), 'y': int(curve_y), 't': start_time + i * time_interval})
    
    elif pattern_type == 'step':
        for i in range(num_points):
            if i % 10 == 0:
                x += random.choice([20, -20, 0])
                y += random.choice([20, -20, 0])
            data.append({'x': x, 'y': y, 't': start_time + i * time_interval})
    
    else:  # zigzag
        amplitude, frequency = random.randint(20, 50), random.uniform(0.1, 0.3)
        for i in range(num_points):
            offset = amplitude * math.sin(i * frequency)
            data.append({'x': int(x + i * 5), 'y': int(y + offset), 't': start_time + i * time_interval})
    
    # Calculate session duration - bots are faster and more consistent
    session_duration = (data[-1]['t'] - start_time) / 1000 if data else 0.5
    
    session = {
        'session_id': f'bot_{session_id:03d}',
        'type': 'bot',
        'metadata': metadata or generate_session_metadata('bot', session_duration),
        'movements': data
    }
    
    return session

def generate_session_metadata(session_type, session_duration):
    """Generate realistic metadata for a session including calculated session_duration."""
    
    # Time distribution weights
    hour_weights = [1,1,1,1,1,2,3,5,8,10,12,15,15,15,12,10,8,12,15,18,15,10,5,2] if session_type == 'human' else [8,10,12,10,8,5,3,2,3,5,7,8,8,8,8,8,8,8,10,8,8,8,10,10]
    
    hour = random.choices(range(24), weights=hour_weights)[0]
    minute, second = random.randint(0, 59), random.randint(0, 59)
    
    # Generate date within last 30 days
    base_date = datetime.now() - timedelta(days=random.randint(0, 30))
    session_time = base_date.replace(hour=hour, minute=minute, second=second)
    
    # Device configurations
    browsers = ['Chrome', 'Firefox', 'Safari', 'Edge']
    os_list = ['Windows 10', 'Windows 11', 'macOS', 'Linux', 'Android', 'iOS']
    
    if session_type == 'human':
        screen_resolutions = ['1920x1080', '1366x768', '1440x900', '1536x864', '1280x720']
        devices = ['Desktop', 'Laptop', 'Tablet', 'Mobile']
        user_agent_entropy = random.uniform(2.0, 8.0)
    else:
        screen_resolutions = ['1920x1080', '1366x768', '1024x768']
        devices = ['Desktop', 'Virtual Machine']
        user_agent_entropy = random.uniform(1.0, 3.0)
    
    return {
        'timestamp': session_time.isoformat(),
        'time_of_day': f"{hour:02d}:{minute:02d}:{second:02d}",
        'day_of_week': session_time.strftime('%A'),
        'browser': random.choice(browsers),
        'os': random.choice(os_list),
        'screen_resolution': random.choice(screen_resolutions),
        'device_type': random.choice(devices),
        'session_duration': round(session_duration, 3),  # Actual calculated duration in seconds
        'ip_region': random.choice(['US-East', 'US-West', 'EU-West', 'Asia-Pacific']),
        'user_agent_entropy': round(user_agent_entropy, 2)
    }

def main():
    """Generate all sessions and save to files."""
    print("Generating mouse movement sessions...")
    
    # Generate sessions
    print("Generating 50 human sessions...")
    human_sessions = [generate_human_session(i + 1) for i in range(50)]
    print("Generating 50 bot sessions...")
    bot_sessions = [generate_bot_session(i + 1) for i in range(50)]
    
    # Save sessions
    print("Saving sessions to files...")
    with open('data/human_sessions.json', 'w') as f:
        json.dump(human_sessions, f, indent=2)
    with open('data/bot_sessions.json', 'w') as f:
        json.dump(bot_sessions, f, indent=2)
    
    # Combined and shuffled dataset
    all_sessions = human_sessions + bot_sessions
    random.shuffle(all_sessions)
    with open('data/all_sessions.json', 'w') as f:
        json.dump(all_sessions, f, indent=2)
    
    # Statistics
    human_durations = [s['metadata']['session_duration'] for s in human_sessions]
    bot_durations = [s['metadata']['session_duration'] for s in bot_sessions]
    human_movements = [len(s['movements']) for s in human_sessions]
    bot_movements = [len(s['movements']) for s in bot_sessions]
    
    print(f"\nSession Generation Summary:")
    print(f"  Total sessions: {len(all_sessions)} (50 human + 50 bot)")
    print(f"\nSession Duration Analysis:")
    print(f"  Human avg: {sum(human_durations)/len(human_durations):.2f}s (range: {min(human_durations):.2f}-{max(human_durations):.2f}s)")
    print(f"  Bot avg: {sum(bot_durations)/len(bot_durations):.2f}s (range: {min(bot_durations):.2f}-{max(bot_durations):.2f}s)")
    print(f"\nMovement Points:")
    print(f"  Human avg: {sum(human_movements)/len(human_movements):.1f}")
    print(f"  Bot avg: {sum(bot_movements)/len(bot_movements):.1f}")
    print(f"\nFiles created: data/human_sessions.json, data/bot_sessions.json, data/all_sessions.json")

if __name__ == "__main__":
    main()
