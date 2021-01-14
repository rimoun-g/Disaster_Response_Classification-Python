import json
import plotly
import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_clean', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # count most frequent words in messages
    text = ""
    for i in df['message'].values:
        text += i + " "
    
    all_words = word_count(text.lower())
    sorted_words = {k: v for k, v in sorted(all_words.items(), key=lambda item: item[1], reverse=True)}
    
    final = {}
    for i in sorted_words.keys():
        if i not in stopwords.words('english') and len(final) <= 10:
            print(i)
            final[i] = sorted_words[i]
            if len(final) > 10:
                break;
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {'data': [
                Bar(
                    x=list(final.keys()),
                    y=list(final.values())
                )
            ],

            'layout': {
                'title': 'Distribution of most frequent words in Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if re.match("\w+",word):
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

    return counts

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

    

if __name__ == '__main__':
    main()