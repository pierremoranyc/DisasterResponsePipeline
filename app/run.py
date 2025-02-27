import json
import plotly
import pandas as pd
import joblib  # Corrected import

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import os
import gdown

app = Flask(__name__) 

# Google Drive File ID of classifier.pkl
file_id = "1vmyMRimyM24MbFFhvNUvl2N6uRGbhTR4"  # Replace with actual file ID
model_path = "models/classifier.pkl"

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Check if model exists, if not, download it
if not os.path.exists(model_path):
    print("Downloading classifier model from Google Drive...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Tokenization function
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

# Load data
db_path = os.path.abspath("../data/DisasterResponse.db")  # Ensure correct database path
engine = create_engine(f'sqlite:///{db_path}')
df = pd.read_sql_table('MessagesCategories', engine)

# Load model from correct path
model = joblib.load(model_path)


# Home page - Displays visuals
@app.route('/')
@app.route('/index')
def index():
    # Extract data for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Define visuals
    graphs = [
        {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        }
    ]
    
    # Encode Plotly graphs in JSON
    ids = [f"graph-{i}" for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Handle user query and display classification results
@app.route('/go')
def go():
    query = request.args.get('query', '')  # Get user input

    # Predict classification for the input query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return render_template('go.html', query=query, classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()