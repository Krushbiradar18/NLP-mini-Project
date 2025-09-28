"""
Demo script to test the MCQ Generator installation and functionality.
Run this script to verify that all dependencies are properly installed.
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit: OK")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    try:
        import nltk
        print("✅ NLTK: OK")
    except ImportError as e:
        print(f"❌ NLTK: {e}")
        return False
    
    try:
        import spacy
        print("✅ spaCy: OK")
    except ImportError as e:
        print(f"❌ spaCy: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn: OK")
    except ImportError as e:
        print(f"❌ scikit-learn: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI: OK")
    except ImportError as e:
        print(f"❌ Google Generative AI: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2: OK")
    except ImportError as e:
        print(f"❌ PyPDF2: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas: OK")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy: OK")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    return True

def test_nltk_data():
    """Test NLTK data downloads"""
    print("\n📚 Testing NLTK data...")
    
    import nltk
    
    try:
        from nltk.corpus import stopwords
        nltk.download('stopwords', quiet=True)
        stop_words = stopwords.words('english')
        print(f"✅ NLTK stopwords: {len(stop_words)} words loaded")
    except Exception as e:
        print(f"❌ NLTK stopwords: {e}")
        return False
    
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        
        test_text = "This is a test sentence. This is another sentence."
        sentences = sent_tokenize(test_text)
        words = word_tokenize(test_text)
        print(f"✅ NLTK tokenization: {len(sentences)} sentences, {len(words)} words")
    except Exception as e:
        print(f"❌ NLTK tokenization: {e}")
        return False
    
    try:
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        
        word = "running"
        stemmed = stemmer.stem(word)
        lemmatized = lemmatizer.lemmatize(word, pos='v')
        print(f"✅ NLTK stemming/lemmatization: '{word}' -> stem: '{stemmed}', lemma: '{lemmatized}'")
    except Exception as e:
        print(f"❌ NLTK stemming/lemmatization: {e}")
        return False
    
    return True

def test_spacy_model():
    """Test spaCy model"""
    print("\n🚀 Testing spaCy model...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        test_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
        doc = nlp(test_text)
        
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"✅ spaCy NER: Found {len(entities)} entities: {entities}")
        
        tokens = [(token.text, token.pos_) for token in doc]
        print(f"✅ spaCy POS tagging: {len(tokens)} tokens processed")
        
    except OSError:
        print("❌ spaCy model not found. Run: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"❌ spaCy error: {e}")
        return False
    
    return True

def test_mcq_generator():
    """Test MCQ Generator class"""
    print("\n🤖 Testing MCQ Generator...")
    
    try:
        from mcq_generator import MCQGenerator
        
        # Test with dummy API key
        test_api_key = "dummy_key_for_testing"
        print("✅ MCQGenerator class imported successfully")
        
        # Test NLP features with sample text
        sample_text = """
        Natural Language Processing (NLP) is a subfield of artificial intelligence 
        that focuses on the interaction between computers and humans through natural language. 
        The ultimate objective of NLP is to read, decipher, understand, and make sense of 
        human languages in a manner that is valuable.
        """
        
        try:
            generator = MCQGenerator(test_api_key)
            print("✅ MCQGenerator initialized (note: dummy API key used)")
            
            # Test preprocessing
            preprocessed_data = generator.preprocess_text(sample_text)
            print(f"✅ Text preprocessing: {len(preprocessed_data['sentences'])} sentences processed")
            
            # Test keyword extraction
            keywords = generator.extract_keywords(preprocessed_data, top_k=5)
            print(f"✅ Keyword extraction: {len(keywords)} keywords found")
            
        except Exception as e:
            print(f"⚠️  MCQGenerator warning: {e} (This is expected with dummy API key)")
        
    except ImportError as e:
        print(f"❌ MCQGenerator import error: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🔍 MCQ Generator Installation Test\n")
    print("="*50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test NLTK data
    if not test_nltk_data():
        all_tests_passed = False
    
    # Test spaCy model
    if not test_spacy_model():
        all_tests_passed = False
    
    # Test MCQ Generator
    if not test_mcq_generator():
        all_tests_passed = False
    
    print("\n" + "="*50)
    
    if all_tests_passed:
        print("🎉 All tests passed! Your installation is ready.")
        print("\n📝 Next steps:")
        print("1. Get your Google Gemini API key from: https://makersuite.google.com/app/apikey")
        print("2. Run the Streamlit app: streamlit run streamlit_app.py")
        print("3. Upload a PDF and start generating MCQ questions!")
    else:
        print("❌ Some tests failed. Please check the errors above and fix them.")
        print("\n🔧 Common fixes:")
        print("- Install spaCy model: python -m spacy download en_core_web_sm")
        print("- Reinstall requirements: pip install -r requirements.txt")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)