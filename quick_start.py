"""
Quick start script for the NLP MCQ Generator
This script helps you get started quickly with the application.
"""

import os
import sys
import subprocess
import webbrowser
import time

def print_banner():
    """Print welcome banner"""
    print("🎓" + "="*60 + "🎓")
    print("  NLP-Based Multiple Choice Question Generator")
    print("  📚 Semester 7 Mini Project - Natural Language Processing")
    print("🎓" + "="*60 + "🎓")
    print()

def check_environment():
    """Check if virtual environment is activated"""
    if sys.prefix == sys.base_prefix:
        print("❌ Virtual environment not activated!")
        print("Please run:")
        print("  source venv/bin/activate")
        print("  python quick_start.py")
        return False
    else:
        print("✅ Virtual environment activated")
        return True

def check_dependencies():
    """Quick dependency check"""
    try:
        import streamlit
        import nltk
        import spacy
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def get_api_key_info():
    """Display API key information"""
    print("\n🔑 Google Gemini API Key Required")
    print("To use this application, you need a Google Gemini API key.")
    print("📋 Steps to get your API key:")
    print("1. Visit: https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated key")
    print("5. Enter it in the Streamlit app sidebar")
    print()

def show_features():
    """Display key features"""
    print("🚀 Key Features:")
    print("├── 📄 PDF text extraction and processing")
    print("├── 🔤 Advanced NLP analysis (tokenization, stemming, lemmatization)")
    print("├── 🏷️ Named Entity Recognition and POS tagging")
    print("├── 📊 TF-IDF keyword extraction")
    print("├── 🤖 AI-powered MCQ generation using Google Gemini")
    print("├── 📈 Question statistics and analysis")
    print("├── 💾 Export options (JSON, CSV, TXT)")
    print("└── 🌐 User-friendly web interface")
    print()

def start_streamlit():
    """Start the Streamlit application"""
    print("🚀 Starting Streamlit application...")
    print("📱 The app will open in your default web browser")
    print("🔗 If it doesn't open automatically, visit: http://localhost:8501")
    print()
    print("⏹️  To stop the application, press Ctrl+C in this terminal")
    print("="*70)
    
    try:
        # Start Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped. Thank you for using NLP MCQ Generator!")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

def show_demo_instructions():
    """Show demo instructions"""
    print("📖 Demo Instructions:")
    print("1. Enter your Google Gemini API key in the sidebar")
    print("2. Upload a PDF file (academic papers, textbooks, etc.)")
    print("3. Choose the number of questions (start with 3-5)")
    print("4. Click 'Generate MCQ Questions'")
    print("5. View the generated questions and NLP analysis")
    print("6. Export questions in your preferred format")
    print()

def main():
    """Main function"""
    print_banner()
    
    # Check environment
    if not check_environment():
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Show features
    show_features()
    
    # API key info
    get_api_key_info()
    
    # Demo instructions
    show_demo_instructions()
    
    # Ask user if they want to start
    response = input("🚀 Ready to start the application? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        start_streamlit()
    else:
        print("👍 No problem! You can start the app anytime by running:")
        print("   streamlit run streamlit_app.py")
        print()
        print("📚 Or run the test script to verify installation:")
        print("   python test_installation.py")

if __name__ == "__main__":
    main()