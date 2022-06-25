#!/usr/bin/env python
# coding: utf-8

from calendar import month
import pickle
import pandas as pd
#import pyarrow
from flask import Flask, request, jsonify

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    
    df = pd.read_parquet(filename)
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def predict(inpt):


    year = inpt['year']
    month = inpt['month']
    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')
    dicts = df[categorical].to_dict(orient='records')
    
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    #print(y_pred.mean())


    #df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    #df_result = pd.DataFrame(y_pred, columns = ['Prediction'])
    #df_result['ride_id'] = df['ride_id']

    #df_result.to_parquet(
    #    'OutputPred.parquet',
    #    engine='pyarrow',
    #    compression=None,
    #    index=False
    #)

    return y_pred.mean()



app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    pred = predict(ride)

    result = {
        'duration': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    #ride = {'year':2021, 'month':3}
    #print(predict_endpoint(ride))
