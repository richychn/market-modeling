from math import sqrt
from numpy import concatenate
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import numpy as np

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    #print(cols)
#     # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def predict(num=1):
    data = pd.read_csv(filepath_or_buffer="./static/data.csv", index_col="date")
    data['spindx'].replace(0, np.nan, inplace=True)
    data['spindx'].fillna(method='ffill', inplace=True)

    number_of_variables = 8
    values = data[['spindx'] + ['TCMNOM_Y2'] + ['TCMNOM_Y10'] + ['DCOILBRENTEU'] + ['GOLDPMGBD228NLBM'] + ['exalus'] + ['exjpus'] + ['exukus']].values
    values = values.astype('float32')

    historic_data = np.array([])
    for day in values[-3:]:
        #historic_data = np.concatenate((values[-3], values[-2], values[-1]), axis=None)
        historic_data = np.concatenate((historic_data, day), axis=None)
    historic_data = np.append (historic_data, historic_data[0]) #add a dummy for prediction

    look_back = 3
    time_steps = 1
    series_to_supervised(values, look_back, time_steps)
    reframed = series_to_supervised(values, look_back, time_steps)

    number_of_variables = 8
    #keeping first varible in first period
    reframed.drop(reframed.columns[-1 * number_of_variables + 1:], axis=1, inplace=True)
    reframed.drop(reframed.columns[look_back*number_of_variables:-1], axis=1, inplace=True)

    reframed = reframed.append(dict(zip(reframed.columns, historic_data)), ignore_index=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_pred = scaler.fit_transform(reframed)

    scaled_pred.shape

    pred_para = scaled_pred[-1][:-1]
    pred_para = pred_para.reshape(1,1,pred_para.shape[0])

    multi_model = load_model("./static/lstm.hdf5")
    yhat = multi_model.predict(pred_para)

    pred_para = pred_para.reshape((1,24))
    pred = concatenate((pred_para[:, :], yhat), axis=1)
    inv_pred = scaler.inverse_transform(pred)
    inv_pred = inv_pred[:,-1]
    graph = []
    for row in values[-30:]:
        graph.append(row[0])
    graph.append(inv_pred[0])
    return graph
