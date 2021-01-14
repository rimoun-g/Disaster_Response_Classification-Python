# import the necessary libs
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
    """This function tokenizes text to be ready for processing"""
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
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
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    

    # count most frequent words in messages
    text = ""
    # store messages in one paragraph
    for i in df['message'].values:
        text += i + " "
    
    # count words
    all_words = word_count(text.lower())
    
    # sort the counted words
    sorted_words = {k: v for k, v in sorted(all_words.items(), key=lambda item: item[1], reverse=True)}
    
    final = {}
    # get the words that are not in the stop words
    for i in sorted_words.keys():
        if i not in stopwords.words('english') and len(final) <= 10:
            print(i)
            final[i] = sorted_words[i]
            if len(final) > 10:
                break;
    
    # create the graphs
    
    graphs = [
        { # first graph the count of each genre
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
        }, # the second graph is the frequency of the top 10 words
        {'data': [
                Bar(
                    x=list(final.keys()),
                    y=list(final.values())
                )
            ],

            'layout': {
                'title': 'Distribution of the top 10 most frequent words in Messages',
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
    """This function returns the prediction of the user input"""
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
    """This function counts the number of times each word appeared in the given text"""
    
    #save the counts in dictinary
    
    counts = dict()
    words = str.split()
    
    # only get the words 
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