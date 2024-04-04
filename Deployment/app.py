# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:53:43 2024

@author: dipen
"""

import streamlit as st
import pandas as pd
import string
from nltk.tokenize import RegexpTokenizer
import docx2txt
import pdfplumber
import re
import PyPDF2
import pickle as pk
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

#----------------------------------------------------------

st.title('RESUME CLASSIFICATION')

# Resize the image and display it
st.image("curriculum-vitae.png")

# Center-align the text
st.markdown('<div style="text-align: center;"><h1 style="color: Red;">Welcome to Resume Classification App</h1></div>', unsafe_allow_html=True)


# FUNCTIONS
def extract_skills(resume_text):
    stop_words = set(stopwords.words('english'))
    
    # Tokenize the resume text into sentences
    sentences = sent_tokenize(resume_text)
    
    # Tokenize each sentence into words and filter out stop words
    tokens = [word_tokenize(sentence) for sentence in sentences]
    filtered_tokens = [
        [token for token in sentence if token.lower() not in stop_words]
        for sentence in tokens
    ]
    
    # Perform part-of-speech tagging on the filtered tokens
    tagged_tokens = [
        nltk.pos_tag(sentence) for sentence in filtered_tokens
    ]
    
    # Extract noun phrases (chunking)
    noun_phrases = []
    for tagged_sentence in tagged_tokens:
        chunk_tree = nltk.ne_chunk(tagged_sentence, binary=False)
        for subtree in chunk_tree.subtrees(filter=lambda t: t.label() == 'NP'):
            noun_phrases.append(' '.join(word for word, pos in subtree.leaves()))
    
    # Return the extracted noun phrases (skillset)
    return noun_phrases


def getText(filename):
    fullText = '' # Create empty string 
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        for para in doc:
            fullText = fullText + para      
    else:  
        with pdfplumber.open(filename) as pdf_file:
            pdoc = PyPDF2.PdfFileReader(filename)
            number_of_pages = pdoc.getNumPages()
            page = pdoc.pages[0]
            page_content = page.extractText()
        for paragraph in page_content:
            fullText =  fullText + paragraph       
    return (fullText)

def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else:
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())  
    return resume[0]

def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words) 

file_type=pd.DataFrame([], columns=['Uploaded File',  'Predicted Profile'])
filename = []
predicted = []


# MAIN CODE

predicted = []

# Load the trained model and vectorizer
with open('mnb_model.pkl', 'rb') as f:
    model = pk.load(f)

with open('bow_vectorizer.pkl', 'rb') as f:
    cv = pk.load(f)

upload_file = st.file_uploader('Upload Your Resumes', type=['docx', 'pdf'], accept_multiple_files=True)

try:
    all_resume_text = ""

    for doc_file in upload_file:
        if doc_file is not None:
            filename.append(doc_file.name)
            cleaned = preprocess(display(doc_file))
            prediction = model.predict(cv.transform([cleaned]))[0]
            if prediction == 0:
                predicted.append("PeopleSoft")
            elif prediction == 1:
                predicted.append("React Developer")
            elif prediction == 2:
                predicted.append("SQL Developer")
            elif prediction == 3:
                predicted.append("Workday")
            else:
                predicted.append("Other")

            # Concatenate the cleaned resume text to the variable
            all_resume_text += cleaned + " "

    if len(predicted) > 0:
        file_type['Uploaded File'] = filename
        file_type['Predicted Profile'] = predicted
        st.table(file_type)

        # Generate and display Word Cloud
        st.subheader("Word Cloud")

        if all_resume_text != "":
            # Create Word Cloud
            wordcloud = WordCloud(width=3000, height=2000, background_color='black', max_words=100,
                                  colormap='Set2').generate(all_resume_text)

            # Display the Word Cloud
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)


except Exception as e: 
    st.error(f"An error occurred: {e}") 