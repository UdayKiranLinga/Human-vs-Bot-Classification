# 🤖 Behavioral CAPTCHA Detector

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning system that distinguishes between human and bot behavior using mouse movement analysis. This project demonstrates how behavioral biometrics can be used for next-generation CAPTCHA systems.

## 🎯 How It Works

Instead of traditional CAPTCHAs that ask users to solve puzzles, this system **invisibly analyzes mouse movements** to detect bots. It works by:

1. **Recording mouse movements** during normal web interactions
2. **Extracting 4 key behavioral features** that distinguish humans from bots
3. **Using a Random Forest classifier** to make real-time predictions
4. **Visualizing patterns** to validate and improve the system

## 📁 Project Structure

```
behavioral-captcha-detector/
├── README.md                    # This documentation
├── requirements.txt             # Python dependencies
├── demo.py                     # Complete workflow demonstration
│
├── generate_sessions.py         # Session data generation (human + bot)
├── extract_features.py          # Feature extraction engine
├── train_model.py              # Model training pipeline
├── predict.py                  # Real-time prediction system
├── test_predictions.py         # Model evaluation & testing
├── visualize_data.py           # Comprehensive visualizations
├── feature_analysis.py         # Feature discrimination analysis
│
├── data/                       # Generated datasets
├── models/                     # Trained models
├── plots/                      # Analysis visualizations
```

## 🔍 Why These 4 Features?

I selected these features because they show the **biggest differences** between human and bot behavior:

| Feature | Why It Works | Human Pattern | Bot Pattern |
|---------|-------------|---------------|-------------|
| **`num_points`** | Humans create micro-movements | 142 ± 34 points | 54 ± 14 points |
| **`max_speed`** | Humans have natural speed limits | 434 ± 156 px/s | 135 ± 81 px/s |
| **`std_speed`** | Humans vary their movement speed | 78 ± 25 px/s | 23 ± 23 px/s |
| **`session_duration`** | Humans take time to think | 10.7 ± 2.7 sec | 5.8 ± 2.1 sec |

**Key Insight**: Bots move too perfectly - they lack the natural imperfections that make humans... human.

## 🎲 Why Random Forest Classifier?

I chose Random Forest because it's **perfect for this problem**:

- ✅ **Works with small datasets** (my 100 sessions)
- ✅ **Handles behavioral data well** (robust to outliers)
- ✅ **Fast predictions** (<10ms for real-time use)
- ✅ **Explainable results** (shows which features matter most)
- ✅ **No overfitting** (ensemble method is naturally robust)

## 📊 Why I Use Visualizations?

Visualizations are **essential** for this security system:

1. **🔍 Validate Features**: Prove my 4 features actually distinguish humans from bots
2. **🛡️ Security Analysis**: Identify patterns that bots might try to exploit
3. **📈 Model Debugging**: Understand why the classifier makes certain decisions
4. **💼 Stakeholder Communication**: Show non-technical users how the system works

## 🚀 Quick Start

### Option 1: One-Command Demo
```bash
python demo.py
```

### Option 2: Step-by-Step
```bash
# Install dependencies
pip install -r requirements.txt

# Generate data and train model
python generate_sessions.py
python extract_features.py
python train_model.py

# Evaluate and visualize
python test_predictions.py
python visualize_data.py
python feature_analysis.py
```

## 📈 Performance

- **Accuracy**: 100% on my test dataset
- **Speed**: <10ms prediction time
- **Features**: 4 behavioral features
- **Dataset**: 100 sessions (50 human + 50 bot)

## 🔮 Future Improvements

- **More Features**: Add click patterns, acceleration analysis
- **Larger Dataset**: Collect more real-world data
- **Deep Learning**: Try neural networks for complex patterns
- **Real-time Integration**: Deploy to actual websites

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Key Innovation**: This system makes bot detection **invisible to users** while being **highly effective** at catching automated scripts through natural behavioral analysis.
