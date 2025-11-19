#!/bin/bash

echo "🚀 Starting Hyperlocal Anomaly Detection Setup..."

# Update packages
sudo apt update -y
sudo apt install -y python3-full python3-venv

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install required dependencies
echo "📚 Installing dependencies..."
pip install streamlit pandas numpy scikit-learn sentence-transformers torch nltk tqdm plotly joblib

# Download NLTK data
echo "📥 Downloading NLTK data..."
python3 - <<END
import nltk
nltk.download('vader_lexicon')
END

echo "✅ Setup Complete!"
echo "👉 To activate environment later: source venv/bin/activate"
echo "👉 To run the app: streamlit run demo.py"
