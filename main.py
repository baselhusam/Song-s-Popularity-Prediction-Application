# Import Packages
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import plotly.express as px
from PIL import Image
import joblib

# Make Containers for each section
header = st.container()
features = st.container()
data = st.container()
model_prediction = st.container()
feature_importances = st.container()
model_accuracy = st.container()


@st.cache(allow_output_mutation=True)
def once():
    
    df = pd.read_csv('song_data_cleaned.csv')
    X = df.drop(columns='song_popularity')
    y = df['song_popularity']

    model = RandomForestRegressor(100)
    model.fit(X, y)
    return model

with header:
    
    # The Header Section: Title + Brief Intro
    st.title("Song Popularity Prediction App")
    st.write("#### This app helps you to predict the popularity for the song based on its features!")
    st.markdown("<h5> Tune the features from the sliders on the left </h5>",unsafe_allow_html=True)
    
    
with features:
    
    # The Featues Section: 
    acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 0.23)
    danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.7)
    energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.5)
    instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.0)
    key = st.sidebar.slider('Key', 0, 11, 3)
    liveness = st.sidebar.slider('Liveness', 0.0, 1.0, 0.5)
    loudness = st.sidebar.slider('Loudness', -54.0, 3.2, -4.0)
    mode = st.sidebar.slider('Mode', 0, 1, 0)
    speechiness = st.sidebar.slider('Speechiness', 0.0, 1.0, 0.03)
    tempo = (st.sidebar.slider('Tempo', 0.0, 220.0, 92.0) - 67.32) / (199.921 - 67.32)
    valence = st.sidebar.slider('Valence', 0.0, 1.0, 0.7)
    time_signature = st.sidebar.slider('time signature',0,5,1)
    song_duration_m = st.sidebar.slider('Song Duration Per Minutes', 0.0,10.0,3.0)
    
    
    features = {
        "acousticness":acousticness,
        "danceability":danceability,
        "energy":energy,
        "instrumentalness":instrumentalness,
        "key":key, 
        "liveness":liveness, 
        "loudness":loudness, 
        "mode":mode, 
        "speechiness":speechiness, 
        "tempo":tempo, 
        "valence":valence, 
        "time_signature":time_signature, 
        "song_duration_m":song_duration_m
           }
    
    st.subheader("User Input Parameters:")
    data = pd.DataFrame(features, index=[0])
    
     
    new_features = {
        "acousticness":acousticness,
        "danceability":danceability,
        "energy":energy,
        "instrumentalness":instrumentalness,
        "liveness":liveness,
        "speechiness":speechiness,
        "valence":valence
    }
    
    del_feat = ['mode','key','loudness','time_signature','song_duration_m']
    new_features = pd.Series(new_features, index=[col for col in new_features.keys() if col not in del_feat])
    fet = px.bar(new_features, orientation='h')
    fet.update_layout(showlegend=False)
    st.plotly_chart(fet, use_container_width=True)
    
    st.markdown("---")
    
    
with model_prediction:
    
    # model = joblib.load(open("model.sav", 'rb'))
    model = once()
    prediction = model.predict(data)
    output_prediction = np.round(prediction[0],2)
    st.markdown(""" <h3 align = 'center'> Song Popularity: </h3>""", unsafe_allow_html=True)
    st.markdown(f" <h3 align='center'> {output_prediction} %</h3>", unsafe_allow_html=True)

    st.markdown('---')
    
    
with feature_importances:
    
    feat = data.columns
    feat_imp = model.feature_importances_
    feature_importance = pd.Series(feat_imp, index= feat)
    
    feature_importance = feature_importance.sort_values(ascending=False)
    fig = px.bar(feature_importance)
    fig.update_layout(showlegend=False)
    
    st.markdown(""" <h3 align = 'center'> Feature Importances</h3>""", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    
with model_accuracy:
    
    st.subheader("Model Performance:")
    
    st.markdown("* **RMSE: 16.6**")
    st.text("So the predictions from this model can be 16 higher or 16 lower in the worst case.")
    
    st.markdown("* **R2 Score: 0.42**")
    st.text("This means we can predict about 42% of the data by this model.")    

