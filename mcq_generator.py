"""
NLP-Based Multiple Choice Question Generator

This module provides a comprehensive MCQ generator using NLP techniques and Google Gemini API.
It extracts text from PDF files, processes it using various NLP methods, and generates
intelligent multiple choice questions.

Author: MCQ Generator Team
Subject: Natural Language Processing
"""

import re
import json
import random
import warnings
from typing import List, Dict, Tuple, Optional
from collections import Counter
import io
import os

import numpy as np
import pandas as pd
import PyPDF2
from io import BytesIO

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai

warnings.filterwarnings('ignore')


class MCQGenerator:
    """
    A comprehensive Multiple Choice Question Generator using NLP and GenAI
    """

    def __init__(self, gemini_api_key: str):
        """
        Initialize the MCQ Generator with Gemini API key

        Args:
            gemini_api_key (str): Google Gemini API key
        """
        # Initialize Gemini API
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

        # Initialize NLP tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # Download required NLTK data
        self._download_nltk_data()

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model not found. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Initialize stop words
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("NLTK stopwords not found. Downloading...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

        print("MCQ Generator initialized successfully!")

    def _download_nltk_data(self):
        """Download required NLTK data"""
        nltk_downloads = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'omw-1.4', 'punkt_tab',
            'averaged_perceptron_tagger_eng'
        ]

        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {item}: {e}")

    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from PDF file

        Args:
            pdf_file: PDF file object or path

        Returns:
            str: Extracted text from PDF
        """
        try:
            if isinstance(pdf_file, str):
                # If it's a file path
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            else:
                # If it's a file object (for Streamlit uploads)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

            print(f"Successfully extracted {len(text)} characters from PDF")
            return text

        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""

    def preprocess_text(self, text: str) -> Dict:
        """
        Comprehensive text preprocessing using NLP techniques

        Args:
            text (str): Input text

        Returns:
            Dict: Preprocessed text data
        """
        # Clean text
        clean_text = re.sub(r'[^\w\s\.\?\!]', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Tokenization
        sentences = sent_tokenize(clean_text)
        words = word_tokenize(clean_text.lower())

        # Remove stop words
        words_no_stop = [word for word in words if word not in self.stop_words and len(word) > 2]

        # Stemming
        stemmed_words = [self.stemmer.stem(word) for word in words_no_stop]

        # Lemmatization
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words_no_stop]

        # POS tagging
        pos_tags = pos_tag(words_no_stop)

        # Named Entity Recognition (if spaCy is available)
        named_entities = []
        if self.nlp:
            doc = self.nlp(clean_text)
            named_entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Extract nouns and verbs (important concepts)
        important_words = [word for word, pos in pos_tags if pos.startswith(('NN', 'VB'))]

        preprocessed_data = {
            'original_text': text,
            'clean_text': clean_text,
            'sentences': sentences,
            'words': words,
            'words_no_stop': words_no_stop,
            'stemmed_words': stemmed_words,
            'lemmatized_words': lemmatized_words,
            'pos_tags': pos_tags,
            'named_entities': named_entities,
            'important_words': important_words
        }

        print(f"Text preprocessing completed:")
        print(f"- Sentences: {len(sentences)}")
        print(f"- Words (no stop words): {len(words_no_stop)}")
        print(f"- Named entities: {len(named_entities)}")

        return preprocessed_data

    def extract_keywords(self, preprocessed_data: Dict, top_k: int = 20) -> List[str]:
        """
        Extract important keywords using TF-IDF

        Args:
            preprocessed_data (Dict): Preprocessed text data
            top_k (int): Number of top keywords to return

        Returns:
            List[str]: List of important keywords
        """
        sentences = preprocessed_data['sentences']

        if not sentences:
            return []

        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()

            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

            # Get top keywords
            top_indices = mean_scores.argsort()[-top_k:][::-1]
            keywords = [feature_names[i] for i in top_indices]

            print(f"Extracted {len(keywords)} keywords: {keywords[:10]}...")
            return keywords
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []

    def identify_key_concepts(self, preprocessed_data: Dict) -> List[str]:
        """
        Identify key concepts from the text

        Args:
            preprocessed_data (Dict): Preprocessed text data

        Returns:
            List[str]: List of key concepts
        """
        concepts = []

        # Add named entities
        entities = [entity[0] for entity in preprocessed_data['named_entities']]
        concepts.extend(entities)

        # Add frequent important words
        important_words = preprocessed_data['important_words']
        word_freq = Counter(important_words)
        frequent_concepts = [word for word, freq in word_freq.most_common(15)]
        concepts.extend(frequent_concepts)

        # Remove duplicates and filter
        concepts = list(set(concepts))
        concepts = [concept for concept in concepts if len(concept) > 3]

        print(f"Identified {len(concepts)} key concepts")
        return concepts

    def generate_mcq_with_gemini(self, text_chunk: str, num_questions: int = 1) -> List[Dict]:
        """
        Generate MCQ questions using Google Gemini

        Args:
            text_chunk (str): Text chunk to generate questions from
            num_questions (int): Number of questions to generate

        Returns:
            List[Dict]: List of generated MCQ questions
        """
        prompt = f"""
        Based on the following text, generate {num_questions} multiple choice question(s) with 4 options each.
        The questions should test comprehension and understanding of key concepts.

        Format the response as a JSON array where each question has:
        - "question": the question text
        - "options": array of 4 options (A, B, C, D)
        - "correct_answer": the correct option letter (A, B, C, or D)
        - "explanation": brief explanation of why the answer is correct

        Text:
        {text_chunk}

        Provide only the JSON array response, no additional text.
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Clean the response to extract JSON
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            questions = json.loads(response_text)
            return questions if isinstance(questions, list) else [questions]

        except Exception as e:
            print(f"Error generating questions with Gemini: {str(e)}")
            return []

    def select_best_chunks(self, sentences: List[str], chunk_size: int = 3, num_chunks: int = 5) -> List[str]:
        """
        Select the best text chunks for question generation

        Args:
            sentences (List[str]): List of sentences
            chunk_size (int): Size of each chunk in sentences
            num_chunks (int): Number of chunks to select

        Returns:
            List[str]: Selected text chunks
        """
        # Create chunks
        chunks = []
        for i in range(0, len(sentences) - chunk_size + 1, chunk_size // 2):
            chunk = ' '.join(sentences[i:i + chunk_size])
            if len(chunk.strip()) > 100:  # Minimum chunk length
                chunks.append(chunk)

        if len(chunks) <= num_chunks:
            return chunks

        # Use TF-IDF to score chunks
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(chunks)

            # Calculate chunk importance (sum of TF-IDF scores)
            chunk_scores = np.sum(tfidf_matrix.toarray(), axis=1)

            # Select top chunks
            top_indices = chunk_scores.argsort()[-num_chunks:][::-1]
            selected_chunks = [chunks[i] for i in top_indices]

            print(f"Selected {len(selected_chunks)} chunks for question generation")
            return selected_chunks
        except Exception as e:
            print(f"Error selecting chunks: {e}")
            return chunks[:num_chunks]

    def generate_mcq_questions(self, pdf_file, num_questions: int = 10) -> List[Dict]:
        """
        Main function to generate MCQ questions from PDF

        Args:
            pdf_file: PDF file object or path
            num_questions (int): Number of questions to generate

        Returns:
            List[Dict]: Generated MCQ questions
        """
        print("Starting MCQ generation process...")

        # Step 1: Extract text from PDF
        text = self.extract_text_from_pdf(pdf_file)
        if not text:
            return []

        # Step 2: Preprocess text
        preprocessed_data = self.preprocess_text(text)

        # Step 3: Extract keywords and concepts
        keywords = self.extract_keywords(preprocessed_data)
        concepts = self.identify_key_concepts(preprocessed_data)

        # Step 4: Select best text chunks
        sentences = preprocessed_data['sentences']
        selected_chunks = self.select_best_chunks(sentences, num_chunks=min(num_questions, 8))

        # Step 5: Generate questions
        all_questions = []
        questions_per_chunk = max(1, num_questions // len(selected_chunks))

        for i, chunk in enumerate(selected_chunks):
            if len(all_questions) >= num_questions:
                break

            questions_needed = min(questions_per_chunk, num_questions - len(all_questions))
            chunk_questions = self.generate_mcq_with_gemini(chunk, questions_needed)

            for question in chunk_questions:
                question['source_chunk'] = f"Chunk {i+1}"
                question['keywords'] = keywords[:5]  # Add relevant keywords

            all_questions.extend(chunk_questions)
            print(f"Generated {len(chunk_questions)} questions from chunk {i+1}")

        # Shuffle questions
        random.shuffle(all_questions)

        print(f"Successfully generated {len(all_questions)} MCQ questions!")
        return all_questions[:num_questions]

    def format_questions_for_display(self, questions: List[Dict]) -> str:
        """
        Format questions for display

        Args:
            questions (List[Dict]): List of questions

        Returns:
            str: Formatted questions
        """
        formatted_output = "="*60 + "\n"
        formatted_output += "MULTIPLE CHOICE QUESTIONS\n"
        formatted_output += "="*60 + "\n\n"

        for i, q in enumerate(questions, 1):
            formatted_output += f"Question {i}:\n"
            formatted_output += f"{q['question']}\n\n"

            for j, option in enumerate(q['options']):
                letter = chr(65 + j)  # A, B, C, D
                formatted_output += f"{letter}) {option}\n"

            formatted_output += f"\nCorrect Answer: {q['correct_answer']}\n"
            formatted_output += f"Explanation: {q['explanation']}\n"

            if 'keywords' in q:
                formatted_output += f"Related Keywords: {', '.join(q['keywords'])}\n"

            formatted_output += "\n" + "-"*40 + "\n\n"

        return formatted_output

    def demonstrate_nlp_features(self, text: str):
        """
        Demonstrate NLP features used in the project

        Args:
            text (str): Sample text to analyze
        """
        print("=== NLP FEATURES DEMONSTRATION ===")
        print(f"Original text: {text[:200]}...\n")

        # Tokenization
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        print(f"1. TOKENIZATION:")
        print(f"   - Sentences: {len(sentences)}")
        print(f"   - Words: {len(words)}")
        print(f"   - First 10 words: {words[:10]}\n")

        # Stop word removal
        words_no_stop = [word for word in words if word not in self.stop_words and len(word) > 2]
        print(f"2. STOP WORD REMOVAL:")
        print(f"   - Words after removal: {len(words_no_stop)}")
        print(f"   - Sample: {words_no_stop[:10]}\n")

        # Stemming
        stemmed_words = [self.stemmer.stem(word) for word in words_no_stop[:10]]
        print(f"3. STEMMING (Porter Stemmer):")
        print(f"   - Original: {words_no_stop[:10]}")
        print(f"   - Stemmed:  {stemmed_words}\n")

        # Lemmatization
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words_no_stop[:10]]
        print(f"4. LEMMATIZATION:")
        print(f"   - Original:     {words_no_stop[:10]}")
        print(f"   - Lemmatized:   {lemmatized_words}\n")

        # POS Tagging
        pos_tags = pos_tag(words_no_stop[:10])
        print(f"5. POS TAGGING:")
        print(f"   - Tags: {pos_tags}\n")

        # Named Entity Recognition
        if self.nlp:
            doc = self.nlp(text[:500])  # Limit text for demo
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            print(f"6. NAMED ENTITY RECOGNITION:")
            print(f"   - Entities found: {entities[:10]}\n")

        # TF-IDF Keywords
        preprocessed_data = self.preprocess_text(text)
        keywords = self.extract_keywords(preprocessed_data, top_k=10)
        print(f"7. TF-IDF KEYWORD EXTRACTION:")
        print(f"   - Top keywords: {keywords}\n")

    def export_questions(self, questions: List[Dict], format_type: str = "json", filename: str = "mcq_questions") -> str:
        """
        Export questions to different formats

        Args:
            questions (List[Dict]): List of questions
            format_type (str): Export format ('json', 'csv', 'txt')
            filename (str): Output filename

        Returns:
            str: Path to exported file
        """
        if format_type == "json":
            filepath = f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)
        
        elif format_type == "csv":
            filepath = f"{filename}.csv"
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
            df.to_csv(filepath, index=False, encoding='utf-8')
        
        elif format_type == "txt":
            filepath = f"{filename}.txt"
            formatted_text = self.format_questions_for_display(questions)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
        
        return filepath


if __name__ == "__main__":
    # Example usage
    API_KEY = "YOUR_GEMINI_API_KEY_HERE"
    
    if API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        generator = MCQGenerator(API_KEY)
        
        # Example with sample text
        sample_text = """
        Natural Language Processing (NLP) is a subfield of artificial intelligence 
        that focuses on the interaction between computers and humans through natural language. 
        The ultimate objective of NLP is to read, decipher, understand, and make sense of 
        human languages in a manner that is valuable.
        """
        
        # Demonstrate NLP features
        generator.demonstrate_nlp_features(sample_text)
    else:
        print("Please set your Gemini API key to use the MCQ Generator")