# NLP-Based Multiple Choice Question Generator

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.50+-red.svg)
![License](https://img.shields.io/badge/license-Educational-green.svg)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20spaCy-orange.svg)

## 🎯 Overview
This project generates intelligent multiple choice questions (MCQs) from PDF documents using advanced Natural Language Processing (NLP) techniques combined with Google Gemini AI. The system demonstrates practical applications of various NLP concepts in an educational context.

### 🚀 Live Demo
- **Repository**: [GitHub](https://github.com/Krushbiradar18/NLP-mini-Project)
- **Clone**: `git clone https://github.com/Krushbiradar18/NLP-mini-Project.git`

## NLP Features Implemented

### 1. **Text Preprocessing**
- **Tokenization**: Breaking text into sentences and words using NLTK
- **Text Cleaning**: Removing special characters and normalizing text
- **Stop Word Removal**: Filtering out common words for better analysis

### 2. **Morphological Analysis**
- **Stemming**: Using Porter Stemmer to reduce words to their root forms
- **Lemmatization**: Converting words to their base/dictionary forms using WordNet

### 3. **Linguistic Analysis**
- **POS Tagging**: Identifying parts of speech for word classification
- **Named Entity Recognition (NER)**: Using spaCy to identify persons, organizations, locations, etc.
- **Important Word Extraction**: Focusing on nouns and verbs as key concepts

### 4. **Information Retrieval**
- **TF-IDF Vectorization**: Extracting important keywords and ranking text chunks
- **Keyword Extraction**: Identifying the most relevant terms from documents
- **Text Chunk Selection**: Intelligently selecting the best passages for question generation

### 5. **AI Integration**
- **Google Gemini API**: Generating contextual and intelligent MCQ questions
- **Prompt Engineering**: Structured prompts for consistent question formatting

## Project Structure
```
NLP-MINI/
├── venv/                      # Virtual environment
├── mcq_generator.py          # Main MCQ generator class
├── streamlit_app.py          # Streamlit web interface
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── NLP_MCQ_Generator.ipynb   # Original Jupyter notebook
```

## 💻 Installation & Setup

### 1. Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### 2. Quick Setup
```bash
# Clone the repository
git clone https://github.com/Krushbiradar18/NLP-mini-Project.git
cd NLP-mini-Project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Test installation
python test_installation.py
```

### 3. Alternative Setup (Automated)
```bash
# Use the setup script (macOS/Linux)
chmod +x setup.sh
./setup.sh
```

### 3. Running the Application

#### Option 1: Streamlit Web Interface (Recommended)
```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run streamlit_app.py
```

#### Option 2: Python Script
```bash
# Activate virtual environment
source venv/bin/activate

# Edit mcq_generator.py to add your API key
# Then run
python mcq_generator.py
```

## Usage Instructions

### Web Interface Features
1. **API Key Configuration**: Enter your Google Gemini API key in the sidebar
2. **PDF Upload**: Upload any PDF document for question generation
3. **Question Settings**: Adjust the number of questions (1-20)
4. **NLP Analysis**: View detailed linguistic analysis of the text
5. **Export Options**: Download questions in JSON, CSV, or TXT format

### NLP Analysis Dashboard
The application provides detailed insights into:
- **Tokenization Results**: Sentence and word counts
- **Morphological Analysis**: Stemming vs lemmatization comparison
- **POS Tagging**: Parts of speech identification
- **Named Entity Recognition**: Extracted entities with types
- **Keyword Extraction**: Top TF-IDF weighted terms
- **Question Statistics**: Analysis of generated questions

## API Configuration

### Getting Google Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and paste it in the Streamlit sidebar

### Security Note
Never commit your API key to version control. The application uses environment variables and secure input methods.

## Technical Implementation

### Core NLP Pipeline
```python
# 1. Text Extraction
text = extract_text_from_pdf(pdf_file)

# 2. Preprocessing
preprocessed_data = preprocess_text(text)
# - Tokenization (sentences, words)
# - Stop word removal
# - Stemming and lemmatization
# - POS tagging
# - Named entity recognition

# 3. Feature Extraction
keywords = extract_keywords(preprocessed_data)  # TF-IDF
concepts = identify_key_concepts(preprocessed_data)

# 4. Content Selection
chunks = select_best_chunks(sentences)  # TF-IDF ranking

# 5. Question Generation
questions = generate_mcq_with_gemini(chunks)
```

### Key Libraries Used
- **NLTK**: Core NLP operations (tokenization, POS tagging, stemming)
- **spaCy**: Advanced NLP (NER, lemmatization)
- **scikit-learn**: TF-IDF vectorization and text analysis
- **PyPDF2**: PDF text extraction
- **Streamlit**: Web interface
- **Google Generativeai**: AI-powered question generation

## Educational Value

### NLP Concepts Demonstrated
1. **Text Preprocessing Pipeline**: Complete workflow from raw text to structured data
2. **Feature Engineering**: Converting text to numerical representations
3. **Information Retrieval**: Ranking and selecting relevant content
4. **Linguistic Analysis**: Understanding text structure and meaning
5. **AI Integration**: Combining traditional NLP with modern generative AI

### Use Cases
- **Educational Assessment**: Automated quiz generation for teachers
- **Content Analysis**: Understanding document structure and key concepts
- **Study Aid**: Creating practice questions from textbooks
- **Research Tool**: Analyzing academic papers and generating comprehension questions

## Troubleshooting

### Common Issues
1. **API Key Error**: Ensure your Gemini API key is valid and has sufficient quota
2. **PDF Reading Error**: Check that the PDF contains extractable text (not images)
3. **Model Loading Error**: Ensure spaCy English model is installed:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. **Memory Issues**: For large PDFs, consider reducing the number of questions

### Performance Tips
- Start with smaller PDFs (< 50 pages) for faster processing
- Use 5-10 questions initially to test the system
- Enable NLP analysis only when needed (it adds processing time)

## Future Enhancements
- Support for multiple document formats (DOCX, TXT)
- Multiple choice question difficulty levels
- Integration with learning management systems
- Support for multiple languages
- Question quality assessment metrics
- Batch processing capabilities

## Contributing
This project is part of an academic assignment for Natural Language Processing. Contributions and suggestions are welcome for educational purposes.

## License
Educational use only. Please respect API usage limits and terms of service.

---

**Subject**: Natural Language Processing  
**Project Type**: Mini Project - Semester 7  
**Technologies**: Python, NLTK, spaCy, scikit-learn, Streamlit, Google Gemini AI