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

import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import re
import string
import emoji
import numpy as np
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

# Import and clean data

# Open TXT files
# en_us_test = open('test.txt', "r")
en_us_blogs = open('en_US.blogs.txt', "r")
en_us_news = open('en_US.news.txt', "r")
en_us_twitter = open('en_US.twitter.txt', "r")

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

# Turn data into Dataframe
data = {
    "blogs": en_us_blogs,
    "news": en_us_news,
    "twitter": en_us_twitter,
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



# App Layout
app.layout = html.Div([

    html.H1("Laboratorio 9", style={'text-align': 'center'}),

    dcc.Dropdown(id="slct_year",
                 options=[
                     {"label": "2015", "value": 2015},
                     {"label": "2016", "value": 2016},
                     {"label": "2017", "value": 2017},
                     {"label": "2018", "value": 2018}],
                 multi=False,
                 value=2015,
                 style={'width': "40%"}
                 ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='my_bee_map', figure={})

])


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)