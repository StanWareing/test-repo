import spacy  # importing spacy
import pandas as pd
from textblob import TextBlob
from detokenize.detokenizer import detokenize

nlp = spacy.load('en_core_web_sm')

def analyze_polarity(text):
    # Preprocess the text with spaCy
    doc = nlp(text)

    clean_token = []
    for token in doc:
      if not (token.is_stop):   
         clean_token.append((token.text).lower().strip('.\n,: '))
         #clean_token.append((token.text).lower().strip(':'))
    sentence = detokenize(clean_token)

    # Analyze sentiment with TextBlob
    print(sentence)
    blob = TextBlob(sentence)
    polarity = blob.sentiment.polarity
    
    return polarity

# Read in text data
df = pd.read_csv("amazon_product_reviews.csv")

# Retrieve 'reviews.text' column
reviews_data = df['reviews.text']
print(reviews_data.head(100))

# Remove missing values from column
clean_data = reviews_data.dropna()

test_data = clean_data[0]
polarity_score = analyze_polarity(test_data)

if polarity_score > 0:
    sentiment = 'positive'
elif polarity_score < 0:
    sentiment = 'negative'
else:
    sentiment = 'neutral'
 
print(f"Text: {test_data}\nPolarity score: {polarity_score}\nSentiment: {sentiment}")


 