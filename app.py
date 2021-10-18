#utf-
'''

            Laboratorio 9.
        Visualización dinámica
        Visualizaciones interactivaS

Creado por:

Andrea Elias
Diego Estrada
Saul Contreras
Juan Fernando De Leon Quezada
'''

from _plotly_utils.basevalidators import TitleValidator
from dash.dcc.Graph import Graph
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from nltk.util import ngrams
import re
import string
import emoji
import numpy as np
from dash import Dash, dcc, html, Input, Output
import random
import warnings
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def normalize_data(data):
    _max = 0
    for i in data:
        if(_max<len(i)):
            _max=len(i)
    for i in data:
        while (_max>len(i)):
            i.append(" ") 
    return (data)


app = Dash(__name__)

# Import and clean data

tweet_frequency = pd.read_csv('tweet_count.csv')
blogs_frequency = pd.read_csv('blogs_count.csv')
news_frequency = pd.read_csv('news_count.csv')

# Open TXT files
# en_us_test = open('test.txt', "r")
en_us_blogs = open('test.txt', "r", encoding="utf8") # CHANGE THE NAME OF TXT FILES FOR THE CORRECT ONE
en_us_news = open('test.txt', "r", encoding="utf8")
en_us_twitter = open('test.txt', "r", encoding="utf8")

# Read TXT files
# en_us_test_text = en_us_test.readlines()
en_us_blogs_text = en_us_blogs.readlines()
en_us_news_text = en_us_news.readlines()
en_us_twitter_text = en_us_twitter.readlines()

# Close TXT Files
# en_us_test.close()
en_us_blogs.close()
en_us_news.close()
en_us_twitter.close()

normalized = normalize_data([en_us_blogs_text, en_us_news_text, en_us_twitter_text])
# Turn data into Dataframe
data = {
    "blogs": normalized[0],
    "news": normalized[1],
    "twitter": normalized[2],
}
df = pd.DataFrame(data=data)

# Eliminar signos de puntuación, url y números
def remove_characters(text):
    '''Remove all signs from a string'''
    return text.translate(text.maketrans('', '', string.punctuation))

def remove_url(text):
    '''Remove url from a string'''
    return re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

def remove_num(text):
    '''Remove num'''
    return re.sub('^\d+\s|\s\d+\s|\s\d+$','',text)

# Se quitan vacíos
df = df.dropna(subset=['blogs'])
df = df.dropna(subset=['news'])
df = df.dropna(subset=['twitter'])

# Lowercasing
df['blogs'] = df['blogs'].apply(lambda line: str(line).lower())
df['news'] = df['news'].apply(lambda line: str(line).lower())
df['twitter'] = df['twitter'].apply(lambda line: str(line).lower())

# Se quitan signos de puntuación
df['blogs'] = df['blogs'].apply(lambda line: remove_characters(str(line)))
df['news'] = df['news'].apply(lambda line: remove_characters(str(line)))
df['twitter'] = df['twitter'].apply(lambda line: remove_characters(str(line)))

# Se quitan enlaces URL
df['blogs'] = df['blogs'].apply(lambda line: remove_url(str(line)))
df['news'] = df['news'].apply(lambda line: remove_url(str(line)))
df['twitter'] = df['twitter'].apply(lambda line: remove_url(str(line)))

# Se quitan los emojis
df['blogs'] = df['blogs'].apply(lambda line: emoji.demojize(str(line)))
df['news'] = df['news'].apply(lambda line: emoji.demojize(str(line)))
df['twitter'] = df['twitter'].apply(lambda line: emoji.demojize(str(line)))

# Se quitan números
df['blogs'] = df['blogs'].apply(lambda line: remove_num(str(line)))
df['news'] = df['news'].apply(lambda line: remove_num(str(line)))
df['twitter'] = df['twitter'].apply(lambda line: remove_num(str(line)))

expresiones = []
for i in expresiones:
    stopwords.add(i)

clean_tweets = []
for tweet in df['twitter']:
    word_list = []
    for word in tweet.split():
        word_list.append(word)
    clean_tweets.append(' '.join(word_list))

clean_blogs = []
for blog in df['blogs']:
    word_listo = []
    for word in blog.split():
        word_listo.append(word)
    clean_blogs.append(' '.join(word_listo))

clean_news = []
for new in df['news']:
    list_words = []
    for word in new.split():
        list_words.append(word)
    clean_news.append(' '.join(list_words))

global_data = clean_tweets+clean_news+clean_blogs
val=round(len(global_data)*0.1,0)
random_sample=random.sample(global_data,int(val))
warnings.filterwarnings('ignore')

# Digrama
digrama=[]
size=2
def ngram(ngrama, size):
    for word in range(len(random_sample)):
        try:
            for item in ngrams(random_sample[word].split(),size):
                ngrama.append(item)
        except Exception as e:
            print(e)
            return ngrama
    return ngrama
digrama = ngram(digrama, size)

#Trigrama
trigrama=[]
size=3
trigrama = ngram(trigrama, size)

#Tetragrama
tetragrama=[]
size=4
tetragrama = ngram(tetragrama, size)

#Pentagrama
pentgrama=[]
size=5
pentgrama = ngram(pentgrama, size)

#MODELO 1: PENTAGRAMA
model = defaultdict(lambda: defaultdict(lambda: 0))
for i,j,k,l,m in pentgrama:
    model[(i,j)][k,l,m] += 1
for i,j in model:
    total=float(sum(model[(i,j)].values()))
    for k,l,m in model[(i,j)]:
        model[(i,j)][k,l,m] /= total

#MODELO 2: DIGRAMA
model2 = defaultdict(lambda: defaultdict(lambda: 0))
for i,j in digrama:
    model2[(i)][j] += 1
for i in model2:
    total=float(sum(model2[(i)].values()))
    for j in model2[(i)]:
        model2[(i)][j] /= total


#MODELO 3: TRIAGRAMA
model3 = defaultdict(lambda: defaultdict(lambda: 0))
for i,j,k in trigrama:
    model3[(i)][j,k] += 1
for i in model3:
    total=float(sum(model3[(i)].values()))
    for j,k in model3[(i)]:
        model3[(i)][j,k] /= total

#MODELO 4: TETRAGRAMA
model4 = defaultdict(lambda: defaultdict(lambda: 0))
for i,j,k,l in tetragrama:
    model4[(i,j)][k,l] += 1
for i,j in model4:
    total=float(sum(model4[(i,j)].values()))
    for k,l in model4[(i,j)]:
        model4[(i,j)][k,l] /= total

#MODELO DE PREDICCION 1
def prediction(input1,input2):
  predictions = {}
  for word1, word2, word3 in model[(input1,input2)].keys():
    predictions[word1] = 1 
    predictions[word2] = 1 
    predictions[word3] = 1
  for word1, word2 in model4[(input1,input2)].keys():
    predictions[word1] = predictions[word1] + 1 if word1 in predictions else 1
    predictions[word2] = predictions[word2] + 1 if word2 in predictions else 1
    predictions[word3] = predictions[word3] + 1 if word3 in predictions else 1
  for word1, word2 in model3[(input1)].keys():
    predictions[word1] = predictions[word1] + 1 if word1 in predictions else 1
    predictions[word2] = predictions[word2] + 1 if word2 in predictions else 1
  for word1, word2 in model3[(input2)].keys():
    predictions[word1] = predictions[word1] + 1 if word1 in predictions else 1
    predictions[word2] = predictions[word2] + 1 if word2 in predictions else 1
  for word1 in model2[(input1)].keys():
    predictions[word1] = predictions[word1] + 1 if word1 in predictions else 1
  for word1 in model2[(input2)].keys():
    predictions[word1] = predictions[word1] + 1 if word1 in predictions else 1
  return predictions

#MODELO DE PREDICCION 2
def prediction2(input1,input2):
  predictions = {}
  for word1, word2, word3 in model[(input1,input2)].keys():
    predictions[word1] = 1 
    predictions[word2] = 1 
    predictions[word3] = 1
  for word1, word2 in model4[(input1,input2)].keys():
    predictions[word1] = predictions[word1] + 1 if word1 in predictions else 1
    predictions[word2] = predictions[word2] + 1 if word2 in predictions else 1
    predictions[word3] = predictions[word3] + 1 if word3 in predictions else 1
  for word1, word2 in model3[(input1)].keys():
    predictions[word1] = predictions[word1] + 1 if word1 in predictions else 1
    predictions[word2] = predictions[word2] + 1 if word2 in predictions else 1
  return predictions

#MODELO DE PREDICCION 3
def prediction3(input1,input2):
  predictions = {}
  for word1, word2, word3 in model[(input1,input2)].keys():
    predictions[word1] = 1 
    predictions[word2] = 1 
    predictions[word3] = 1
  for word1, word2 in model3[(input2)].keys():
    predictions[word1] = predictions[word1] + 1 if word1 in predictions else 1
    predictions[word2] = predictions[word2] + 1 if word2 in predictions else 1
  for word1 in model2[(input1)].keys():
    predictions[word1] = predictions[word1] + 1 if word1 in predictions else 1
  for word1 in model2[(input2)].keys():
    predictions[word1] = predictions[word1] + 1 if word1 in predictions else 1
  return predictions
    

# App Layout
app.layout = html.Div([

    html.H1("Laboratorio 9", style={'text-align': 'center'}),

    html.H2("Explore data", style={'text-align': 'center'}),
    html.Button("NAVIGATE TO NEXT",id="next",n_clicks=0, style={'text-align': 'center', 'margin-left': 50, 'height': 50, 'width': 150, 'color': 'white', 'background': '#004AAD'}),
    html.H3(random.choice(normalized[0] + normalized[1] + normalized[2]),id="data",style={'background':'#55ADEE','min-height':300, 'display': 'flex', 'align-items':'center','justify-content':'center', 'padding': 16, 'text-align': 'center','border':'double 2px black', 'color': 'white', 'margin-right': 50, 'margin-left': 50, 'align-self': 'center'}),

    html.H2("Modelos de predicción",style={'text-align': 'center'}),

    dcc.Input(
        id="model",
        type="text",
        style={'align-self': 'center','width': 500, 'height': 50, 'margin-left': 50,},
        placeholder="Escribe las palabras a predecir... ",
    ),

    html.H5(id="model_catch",style={'color': 'red', 'margin-left': 50,}),

    dcc.Graph(id="model1"),
    dcc.Graph(id="model2"),
    dcc.Graph(id="model3"),

    dcc.Dropdown(
        id='freq-dropdown',
        options=[
            {'label': 'Tweets', 'value': 'tweets'},
            {'label': 'Blogs', 'value': 'blogs'},
            {'label': 'Noticias', 'value': 'news'}
        ],
        value='tweets'
    ),
    dcc.Graph(id="freq-hist"),

    html.Br(),

])

@app.callback(
    Output("data", "children"),
    Input('next', 'n_clicks'),
)
def update_result(n_clicks):
    return random.choice(normalized[0] + normalized[1] + normalized[2])

@app.callback(
    Output("model_catch", "children"),
    Input("model", "value"),
)
def update_result(model):
    words = str(model).split(" ")
    if (len(words)!=2):
        return "Write exactly two words."
    else:
        return ""

@app.callback(
    Output("model1", "figure"),
    [Input("model", "value")],
)
def update_result(model):
    words = str(model).split(" ")
    if (len(words)==2):
        predictions_1 = prediction(words[0],words[1])
        my_predictions1 = dict(sorted(predictions_1.items(), key=lambda item: item[1], reverse=True))
        prediction_ = pd.DataFrame(list(zip(my_predictions1.keys(), [item/10 for item in my_predictions1.values()])),columns =['Word', 'Probabilty'])
        fig = px.bar(prediction_, x="Word", y="Probabilty", title="Model 1 prediction")
        return fig
    else:
        prediction_ = pd.DataFrame(list(zip([], [])),columns =['Word', 'Probabilty'])
        fig = px.bar(prediction_, x="Word", y="Probabilty", title="Model 1 prediction")
        return fig

@app.callback(
    Output("model2", "figure"),
    [Input("model", "value")],
)
def update_result(model):
    words = str(model).split(" ")
    if (len(words)==2):
        predictions_1 = prediction2(words[0],words[1])
        my_predictions1 = dict(sorted(predictions_1.items(), key=lambda item: item[1], reverse=True))
        prediction_ = pd.DataFrame(list(zip(my_predictions1.keys(), [item/10 for item in my_predictions1.values()])),columns =['Word', 'Probabilty'])
        fig = px.bar(prediction_, x="Word", y="Probabilty", title="Model 2 prediction")
        return fig
    else:
        prediction_ = pd.DataFrame(list(zip([], [])),columns =['Word', 'Probabilty'])
        fig = px.bar(prediction_, x="Word", y="Probabilty", title="Model 2 prediction")
        return fig

@app.callback(
    Output("model3", "figure"),
    [Input("model", "value")],
)
def update_result(model):
    words = str(model).split(" ")
    if (len(words)==2):
        predictions_1 = prediction3(words[0],words[1])
        my_predictions1 = dict(sorted(predictions_1.items(), key=lambda item: item[1], reverse=True))
        prediction_ = pd.DataFrame(list(zip(my_predictions1.keys(), [item/10 for item in my_predictions1.values()])),columns =['Word', 'Probabilty'])
        fig = px.bar(prediction_, x="Word", y="Probabilty", title="Model 3 prediction")
        return fig
    else:
        prediction_ = pd.DataFrame(list(zip([], [])),columns =['Word', 'Probabilty'])
        fig = px.bar(prediction_, x="Word", y="Probabilty", title="Model 3 prediction")
        return fig

@app.callback(
    Output("freq-hist", "figure"), 
    Input("freq-dropdown", "value"))
def display_color(value):
    if(value == "tweets"):
        fig = px.histogram(tweet_frequency[:5], x="Word", y="Frequency", title="Palabras más frecuentes en tweets")
        return fig
    elif(value == "blogs"):
        fig = px.histogram(blogs_frequency[:5], x="Word", y="Frequency", title="Palabras más frecuentes en Blogs")
        return fig
    elif(value == "news"):
        fig = px.histogram(news_frequency[:5], x="Word", y="Frequency", title="Palabras más frecuentes en noticias")
        return fig
    else:
        freq = pd.DataFrame(list(zip([], [])),columns =['Word', 'Frequency'])
        fig = px.bar(freq, x="Word", y="Frequency", title="Seleccione un medio")
        return fig
        
    
    
    


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)