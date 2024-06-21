import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import asyncio

app = Flask("__name__")

async def fetch(url):
    response = requests.get(url)
    return response.json()

async def fetch_data_mountain(user_uuid):
    mountains_url = "https://highking.cloud/api/mountains/get-mountains/ml"
    logs_url = f"https://highking.cloud/api/logs/get-mountain-logs/{user_uuid}"
    tasks = [fetch(mountains_url), fetch(logs_url)]
    return await asyncio.gather(*tasks)

@app.route('/api/recommend-mountain/<user_uuid>', methods=['GET'])
async def predict_mountain(user_uuid):
    try:
        # Load the model
        model_mountain = tf.keras.models.load_model('mountain_recommender_model.keras')
        # Load the preprocessor
        preprocessor_mountain = joblib.load('preprocessor_mountain.joblib')
        # Load the TF-IDF vectorizer
        tfidf_mountain = joblib.load('tfidf_vectorizer_mountain.joblib')
        
        mountains_data, logs_data = await fetch_data_mountain(user_uuid)
    except Exception as e:
        return jsonify({'error': f"Failed to fetch data: {str(e)}"}), 500

    mountains_df = pd.DataFrame(mountains_data["data"])
    mountains_df = mountains_df.rename(columns={
                    'height': 'elevation'
                })
    logs_df = pd.DataFrame(logs_data["data"])

    if not logs_df.empty:
        mountain_uuid_max = logs_df["mountain_uuid"].value_counts().idxmax()
        
        X_old = mountains_df.drop(columns=['name'])
        x_preprocess = preprocessor_mountain.transform(X_old)
        x_tfidf = tfidf_mountain.transform(mountains_df["name"])
        X_features = np.hstack([x_preprocess, x_tfidf.toarray()])  
        mountain_idx = mountains_df[mountains_df['mountain_uuid'] == mountain_uuid_max].index[0]
        mountain_features = mountains_df.drop(columns=['name']).iloc[mountain_idx]
        mountain_features_processed = preprocessor_mountain.transform(pd.DataFrame([mountain_features]))
        mountain_name_tfidf = tfidf_mountain.transform([mountains_df.iloc[mountain_idx]["name"]])
        mountain_input = np.hstack([mountain_features_processed, mountain_name_tfidf.toarray()])


        recommendations = model_mountain.predict(mountain_input)
        cosine_similarities = cosine_similarity(recommendations, X_features).flatten()
        top_indices = cosine_similarities.argsort()[-10:][::-1]
        mountains_df = mountains_df.rename(columns={
                'elevation': 'height'
            })
        recommended_mountain = [mountains_df.iloc[i].to_dict() for i in top_indices if mountains_df.iloc[i]['mountain_uuid'] != mountain_uuid_max]
        return jsonify({'data': recommended_mountain}), 200
    else:
        return jsonify({'data': mountains_data["data"]}), 200


async def fetch_data_open_trips(user_uuid):
    opentrip_url = "https://highking.cloud/api/open-trips/get-open-trip/rec"
    logs_url = f"https://highking.cloud/api/logs/get-opentrip-logs/{user_uuid}"
    tasks = [fetch(opentrip_url), fetch(logs_url)]
    return await asyncio.gather(*tasks)

@app.route('/api/recommend-opentrip/<user_uuid>', methods=['GET'])
async def predict_opentrip(user_uuid):
    try:
        # Load the model
        model_trips = tf.keras.models.load_model('model_trip.keras')
        # Load the preprocessor
        preprocessor_trips = joblib.load('preprocessor_trip.joblib')
        # Load the TF-IDF vectorizer
        tfidf_trips = joblib.load('tfidf_vectorizer_trip.joblib')

        opentrip_data, logs_data = await fetch_data_open_trips(user_uuid)
    except Exception as e:
        return jsonify({'error': f"Failed to fetch data: {str(e)}"}), 500

    opentrips_df = pd.DataFrame(opentrip_data["data"])
    df_inference = opentrips_df.copy()
    logs_df = pd.DataFrame(logs_data["data"])

    if opentrips_df.empty:
        return jsonify({'data': []}), 200

    if not logs_df.empty :
        opentrip_uuid_max = logs_df["open_trip_uuid"].value_counts().idxmax()

        df_inference['start_date'] = pd.to_datetime(df_inference['start_date'])
        df_inference['end_date'] = pd.to_datetime(df_inference['end_date'])
        df_inference['include'] = df_inference['include'].str.lower()

        # Extracting numerical features from date columns
        df_inference['Berangkat_Year'] = df_inference['start_date'].dt.year
        df_inference['Berangkat_Month'] = df_inference['start_date'].dt.month
        df_inference['Berangkat_Day'] = df_inference['start_date'].dt.day
        df_inference['Pulang_Year'] = df_inference['end_date'].dt.year
        df_inference['Pulang_Month'] = df_inference['end_date'].dt.month
        df_inference['Pulang_Day'] = df_inference['end_date'].dt.day

        boolean_features = [
            'Transportasi PP','Simaksi','Guide', 'Rumah singgah/Homestay', 'Makan (sesudah atau Sebelum pendakian)',
            'Makan (selama pendakian)', 'Air mineral', 'P3K standard', 'Alat Masak',
            'Tenda Toilet', 'Porter Team', 'Tenda', 'alat makan', 'Dokumentasi',
            'Private', 'Logistik', 'Coffe Break/Buah', 'BBM', 'Tiket', 'Jeep', 'Asuransi'
        ]
        # Iterate over each feature and create new columns
        for feature in boolean_features:
            # Convert the feature name to lowercase for comparison
            feature_lower = feature.lower()
            
            # Create a new column with 1 if the feature is present, 0 otherwise
            df_inference[feature] = df_inference['include'].str.contains(feature_lower).astype(int)

        X = df_inference.drop(columns=['name', 'start_date', 'end_date'])
        X_processed = preprocessor_trips.transform(X)
        # Create TF-IDF matrix for 'Gunung' column
        tfidf_matrix = tfidf_trips.transform(df_inference['mountain_name'])
        # Concatenate all features
        trips_features = np.hstack([X_processed, tfidf_matrix.toarray()])
        
        opentrip_idx = df_inference[df_inference['open_trip_uuid'] == opentrip_uuid_max].index[0]
        opentrip_features = df_inference.drop(columns=['name']).iloc[opentrip_idx]
        opentrip_features_processed = preprocessor_trips.transform(pd.DataFrame([opentrip_features]))
        opentrip_name_tfidf = tfidf_trips.transform([df_inference.iloc[opentrip_idx]["mountain_name"]])
        opentrip_input = np.hstack([opentrip_features_processed, opentrip_name_tfidf.toarray()])         

        recommendations = model_trips.predict(opentrip_input)
        cosine_similarities = cosine_similarity(recommendations, trips_features).flatten()
        top_indices = cosine_similarities.argsort()[-10:][::-1]
        recommended_trips = [opentrips_df.iloc[i].to_dict() for i in top_indices if opentrips_df.iloc[i]['mountain_uuid'] != opentrip_uuid_max]
        return jsonify({'data': recommended_trips}), 200
    else:
        return jsonify({'data': opentrip_data["data"]}), 200

if __name__ == '__main__':
    app.run(threaded=True)