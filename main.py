import streamlit as st
import pickle as pkl
import re
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import spacy
from nltk.corpus import stopwords


# load model and vectorizer
with open('svm.pkl', 'rb') as model_file: 
    model = pkl.load(model_file)
    
with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pkl.load(vec_file)


st.title('Text Sentiment Analysis')

text = st.text_area(
        'Enter the text to analyze: ',
)


if st.button('Analyze!'):
    if text:
        # convert the text into string and lower case
        text = text.lower()
        
        #remove urls, user mentions, hashtags, numbers and special character
        text = re.sub(r'https?://\S+|@\w+|#|\d|[^\w\s]', '', text) 
        
        # tokenization
        token = word_tokenize(text)
                
        # lemmitization
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text) 
        token = [token.lemma_ for token in doc]

        # remove stopwords
        stop_words = set(stopwords.words('english'))
        token = [word for word in token if word not in stop_words]
        
        # Convert back to text for vectorization
        processed_text = ' '.join(token)
        
        print(type(vectorizer))

        # Vectorize the text
        text_vectorized = vectorizer.transform([processed_text])
      
        # prediction
        prediction = model.predict(text_vectorized)[0]
        
        if prediction == 0:
            st.error('Negative')
        elif prediction == 1:
            st.info('Neutral')
        else:
            st.success('Positive')
            
    else:
        st.warning('Please enter some text to analyze')
