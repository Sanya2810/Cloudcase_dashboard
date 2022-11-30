import pandas as pd
from datetime import datetime
import pickle
import time
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def apply_model(model,list_names):
    df = pd.DataFrame(list_names, columns=[ 'dob','category', 'job', 'time','amount(usd)'])
    df["dob"] = pd.to_datetime(df["dob"], infer_datetime_format=True)  # changing it to pandas dateformat
    now = pd.Timestamp('now')
    df["age"] = (now - df['dob']).astype('<m8[Y]')
    converted_time = time.strptime(df.iloc[0, 3], '%H:%M:%S')
    df['hour'] = converted_time.tm_hour
    features = ['category', 'amount(usd)', 'job', 'hour', 'age']
    data = df[features]
    enc = OrdinalEncoder(dtype=np.int64)
    enc.fit(data.loc[:, ['category', 'job']])
    data.loc[:, ['category', 'job']] = enc.transform(data[['category', 'job']])
    value = model.predict_proba(data)
    return value[:,1]

def fraud_classification(value):
    if float(value) > 0.1:
        print('fraudulent')
    else:
        print('not fraudulent')