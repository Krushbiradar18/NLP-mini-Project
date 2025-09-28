"""
Streamlit UI for NLP-Based Multiple Choice Question Generator

This Streamlit application provides a user-friendly interface for generating
MCQ questions from PDF files using NLP techniques and Google Gemini API.

Features:
- PDF file upload
- MCQ generation with customizable number of questions
- Display of generated questions
- Export functionality (JSON, CSV, TXT)
- NLP analysis demonstration
- Question statistics and analysis

Author: MCQ Generator Team
Subject: Natural Language Processing
"""

import streamlit as st
import json
import os
from io import BytesIO
import pandas as pd
import numpy as np
from collections import Counter

# Import our MCQ Generator
from mcq_generator import MCQGenerator

# Page configuration
st.set_page_config(
    page_title="NLP MCQ Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .option-text {
        margin: 0.5rem 0;
        padding: 0.25rem;
    }
    .correct-answer {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.5rem;
        color: #155724;
    }
    .nlp-demo {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

def display_nlp_features(generator, text):
    """Display NLP features demonstration"""
    st.markdown('<div class="sub-header">🔍 NLP Features Analysis</div>', unsafe_allow_html=True)
    
    with st.expander("Click to see NLP Analysis Details", expanded=False):
        # Preprocess the text
        preprocessed_data = generator.preprocess_text(text[:1000])  # Limit for demo
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📝 Tokenization:**")
            st.write(f"- Sentences: {len(preprocessed_data['sentences'])}")
            st.write(f"- Words: {len(preprocessed_data['words'])}")
            st.write(f"- Words (no stop words): {len(preprocessed_data['words_no_stop'])}")
            
            st.markdown("**🏷️ POS Tagging (first 10):**")
            pos_tags_sample = preprocessed_data['pos_tags'][:10]
            for word, pos in pos_tags_sample:
                st.write(f"- {word}: {pos}")
        
        with col2:
            st.markdown("**🔤 Stemming vs Lemmatization (first 10):**")
            original_words = preprocessed_data['words_no_stop'][:10]
            stemmed_words = preprocessed_data['stemmed_words'][:10]
            lemmatized_words = preprocessed_data['lemmatized_words'][:10]
            
            comparison_df = pd.DataFrame({
                'Original': original_words,
                'Stemmed': stemmed_words,
                'Lemmatized': lemmatized_words
            })
            st.dataframe(comparison_df)
        
        # Keywords extraction
        st.markdown("**🔑 TF-IDF Keywords:**")
        keywords = generator.extract_keywords(preprocessed_data, top_k=15)
        if keywords:
            keyword_str = ", ".join(keywords[:15])
            st.write(keyword_str)
        
        # Named entities
        if preprocessed_data['named_entities']:
            st.markdown("**👤 Named Entities:**")
            entities_df = pd.DataFrame(preprocessed_data['named_entities'], 
                                     columns=['Entity', 'Type'])
            st.dataframe(entities_df)

def display_questions(questions):
    """Display generated questions in a formatted way"""
    st.markdown('<div class="sub-header">📋 Generated Questions</div>', unsafe_allow_html=True)
    
    for i, question in enumerate(questions, 1):
        with st.container():
            st.markdown(f'<div class="question-box">', unsafe_allow_html=True)
            st.markdown(f"**Question {i}:** {question['question']}")
            
            # Display options
            for j, option in enumerate(question['options']):
                letter = chr(65 + j)  # A, B, C, D
                option_class = "correct-answer" if letter == question['correct_answer'] else "option-text"
                prefix = "✅ " if letter == question['correct_answer'] else ""
                st.markdown(f'<div class="{option_class}">{prefix}{letter}) {option}</div>', 
                           unsafe_allow_html=True)
            
            # Show explanation
            with st.expander(f"💡 Explanation for Question {i}"):
                st.write(question['explanation'])
                if 'keywords' in question and question['keywords']:
                    st.write(f"**Related Keywords:** {', '.join(question['keywords'])}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

def display_question_statistics(questions):
    """Display statistics about generated questions"""
    st.markdown('<div class="sub-header">📊 Question Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Questions", len(questions))
        
        # Question lengths
        question_lengths = [len(q['question'].split()) for q in questions]
        avg_length = np.mean(question_lengths) if question_lengths else 0
        st.metric("Avg Question Length", f"{avg_length:.1f} words")
    
    with col2:
        # Answer distribution
        answer_distribution = Counter([q['correct_answer'] for q in questions])
        st.write("**Answer Distribution:**")
        for answer, count in sorted(answer_distribution.items()):
            st.write(f"Option {answer}: {count}")
    
    with col3:
        if question_lengths:
            st.write("**Question Length Range:**")
            st.write(f"Min: {min(question_lengths)} words")
            st.write(f"Max: {max(question_lengths)} words")
    
    # Keywords analysis
    if questions and any('keywords' in q for q in questions):
        st.markdown("**🏷️ Most Common Keywords:**")
        all_keywords = []
        for q in questions:
            if 'keywords' in q:
                all_keywords.extend(q['keywords'])
        
        if all_keywords:
            keyword_freq = Counter(all_keywords)
            keyword_df = pd.DataFrame(keyword_freq.most_common(10), 
                                    columns=['Keyword', 'Frequency'])
            st.bar_chart(keyword_df.set_index('Keyword'))

def export_questions(questions, format_type):
    """Export questions to different formats"""
    if not questions:
        st.error("No questions to export!")
        return None
    
    if format_type == "JSON":
        export_data = json.dumps(questions, indent=2, ensure_ascii=False)
        return export_data.encode('utf-8'), "questions.json", "application/json"
    
    elif format_type == "CSV":
        df_data = []
        for i, q in enumerate(questions, 1):
            df_data.append({
                'Question_Number': i,
                'Question': q['question'],
                'Option_A': q['options'][0],
                'Option_B': q['options'][1],
                'Option_C': q['options'][2],
                'Option_D': q['options'][3],
                'Correct_Answer': q['correct_answer'],
                'Explanation': q['explanation']
            })
        df = pd.DataFrame(df_data)
        csv_data = df.to_csv(index=False)
        return csv_data.encode('utf-8'), "questions.csv", "text/csv"
    
    elif format_type == "TXT":
        formatted_text = ""
        for i, q in enumerate(questions, 1):
            formatted_text += f"Question {i}: {q['question']}\n\n"
            for j, option in enumerate(q['options']):
                letter = chr(65 + j)
                formatted_text += f"{letter}) {option}\n"
            formatted_text += f"\nCorrect Answer: {q['correct_answer']}\n"
            formatted_text += f"Explanation: {q['explanation']}\n"
            formatted_text += "\n" + "-"*50 + "\n\n"
        
        return formatted_text.encode('utf-8'), "questions.txt", "text/plain"

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📚 NLP-Based MCQ Generator</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        Generate intelligent multiple choice questions from PDF documents using advanced NLP techniques and Google Gemini AI
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Google Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key to enable question generation"
        )
        
        if api_key:
            if st.session_state.generator is None or st.session_state.get('api_key') != api_key:
                with st.spinner("Initializing MCQ Generator..."):
                    try:
                        st.session_state.generator = MCQGenerator(api_key)
                        st.session_state.api_key = api_key
                        st.success("✅ Generator initialized!")
                    except Exception as e:
                        st.error(f"❌ Error initializing generator: {str(e)}")
                        st.session_state.generator = None
        
        st.markdown("---")
        
        # Settings
        st.markdown("### 📋 Question Settings")
        num_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=20,
            value=5,
            help="Select how many questions to generate"
        )
        
        show_nlp_analysis = st.checkbox(
            "Show NLP Analysis",
            value=True,
            help="Display detailed NLP processing information"
        )
        
        st.markdown("---")
        
        # Export options
        if st.session_state.questions:
            st.markdown("### 💾 Export Options")
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "CSV", "TXT"],
                help="Choose format for exporting questions"
            )
            
            if st.button("📥 Export Questions"):
                export_data, filename, mime_type = export_questions(
                    st.session_state.questions, export_format
                )
                if export_data:
                    st.download_button(
                        label=f"⬇️ Download {export_format}",
                        data=export_data,
                        file_name=filename,
                        mime=mime_type
                    )
    
    # Main content
    if not api_key:
        st.warning("🔑 Please enter your Google Gemini API key in the sidebar to get started.")
        st.markdown("""
        ### 🚀 How to get started:
        1. Get your Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enter the API key in the sidebar
        3. Upload a PDF file
        4. Generate MCQ questions!
        
        ### 🔧 NLP Features Used:
        - **Tokenization**: Breaking text into sentences and words
        - **Stemming & Lemmatization**: Word normalization techniques
        - **POS Tagging**: Identifying parts of speech
        - **Named Entity Recognition**: Extracting important entities
        - **TF-IDF**: Keyword extraction and text ranking
        - **Stop Word Removal**: Filtering common words
        """)
        return
    
    if not st.session_state.generator:
        st.error("❌ Please check your API key and try again.")
        return
    
    # File upload
    st.markdown('<div class="sub-header">📤 Upload PDF Document</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to generate MCQ questions from"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📄 **File:** {uploaded_file.name}")
        with col2:
            st.info(f"📊 **Size:** {uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.info(f"📋 **Questions:** {num_questions}")
        
        # Generate questions button
        if st.button("🤖 Generate MCQ Questions", type="primary"):
            with st.spinner("🔄 Processing PDF and generating questions..."):
                try:
                    # Generate questions
                    questions = st.session_state.generator.generate_mcq_questions(
                        uploaded_file, num_questions=num_questions
                    )
                    
                    if questions:
                        st.session_state.questions = questions
                        st.success(f"✅ Successfully generated {len(questions)} questions!")
                        
                        # Show NLP analysis if requested
                        if show_nlp_analysis:
                            # Extract some text for analysis
                            text = st.session_state.generator.extract_text_from_pdf(uploaded_file)
                            if text:
                                display_nlp_features(st.session_state.generator, text)
                        
                        # Display questions
                        display_questions(questions)
                        
                        # Show statistics
                        display_question_statistics(questions)
                        
                    else:
                        st.error("❌ Failed to generate questions. Please check your PDF file and API key.")
                        
                except Exception as e:
                    st.error(f"❌ Error generating questions: {str(e)}")
    
    # Display existing questions if available
    elif st.session_state.questions:
        display_questions(st.session_state.questions)
        display_question_statistics(st.session_state.questions)
    
    # Demo section
    with st.expander("🧪 Try NLP Demo with Sample Text", expanded=False):
        st.markdown("Test the NLP features with your own text:")
        
        sample_text = st.text_area(
            "Enter text for NLP analysis:",
            value="Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language. Machine learning algorithms are used to analyze and understand human language.",
            height=100
        )
        
        if st.button("🔍 Analyze Text"):
            if st.session_state.generator:
                display_nlp_features(st.session_state.generator, sample_text)
            else:
                st.warning("Please initialize the generator with your API key first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>📚 NLP-Based MCQ Generator | Built with Streamlit & Google Gemini AI</p>
        <p>🔬 Demonstrates: Tokenization, Stemming, Lemmatization, POS Tagging, NER, TF-IDF</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()