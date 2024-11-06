# Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template

# Load dataset
data = pd.read_csv('java_dataset.csv')

# Initialize TF-IDF Vectorizer and compute term-document matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Question'])

# Function to find the most similar question
def find_answer(user_question, threshold=0.1):
    user_question_tfidf = vectorizer.transform([user_question])
    cosine_similarities = cosine_similarity(user_question_tfidf, tfidf_matrix).flatten()
    
    # Get the best match
    best_match_index = cosine_similarities.argmax()
    best_similarity_score = cosine_similarities[best_match_index]

    # If the similarity score is too low, return a default response
    if best_similarity_score < threshold:
        return "Sorry, I couldn't find a relevant answer.", "Can you please try asking a Java-related question?"

    best_match_question = data['Question'][best_match_index]
    best_match_answer = data['Answer'][best_match_index]
    return best_match_question, best_match_answer


# Initialize Flask application
app = Flask(__name__)

# Route to serve the HTML template
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the chatbot
@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get('question')
    best_question, best_answer = find_answer(user_question)
    return jsonify({'question': best_question, 'answer': best_answer})

if __name__ == '__main__':
    app.run(debug=True)
