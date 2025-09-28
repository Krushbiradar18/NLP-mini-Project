# Development Environment Setup Script
# Run this script to set up the complete development environment

echo "🔧 Setting up NLP MCQ Generator Development Environment..."
echo "============================================================"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Install spaCy model
echo "🚀 Installing spaCy English model..."
python -m spacy download en_core_web_sm

# Test installation
echo "🧪 Testing installation..."
python test_installation.py

echo ""
echo "🎉 Setup complete!"
echo "💡 To get started:"
echo "   1. Activate the virtual environment: source venv/bin/activate"
echo "   2. Run the quick start script: python quick_start.py"
echo "   3. Or directly run: streamlit run streamlit_app.py"