"""
  Takes a model and data, modifies the data for the model, and runs the model
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras import backend as K
from numpy import concatenate
from numpy import append
from numpy import array
from numpy import nan
from numpy import multiply

def sign_ae(x, y):
    """
    Checks the sign of x and y and returns a function of the difference
    """
    sign_x = K.sign(x)
    sign_y = K.sign(y)
    delta = x - y
    return sign_x * sign_y * K.abs(delta)


def linex_loss(delta, a=-1, b=1):
    """
    Calculates Linex loss, a financial return loss function
    """
    if a!= 0 and b > 0:
        loss = b * (K.exp(a * delta) - a * delta - 1)
        return loss
    else:
        raise ValueError


def linex_loss_val(y_true, y_pred):
    """
    Returns linex loss function without transforming data
    """
    delta = sign_ae(y_true, y_pred)
    res = linex_loss(delta)
    return K.mean(res)


def linex_loss_ret(y_true, y_pred):
    """
    Returns linex loss function calculating the change for the data,
    since linex only considers change
    """
    diff_true = y_true[1:] - y_true[:-1]
    diff_pred = y_pred[1:] - y_pred[:-1]

    delta = sign_ae(diff_true, diff_pred)
    res = linex_loss(delta)
    return K.mean(res)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
      Creates batches of data for the model
    """
    n_vars = 1 if isinstance(data, list) else data.shape[1]
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
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def setup_levels():
    """
      Gets and cleans the data for levels
    """
    data = pd.read_csv(filepath_or_buffer="./static/data/levels.csv", index_col="date")
    data = data.apply(pd.to_numeric, errors = "coerce")
    data['spindx'].replace(0, nan, inplace=True)
    data['spindx'].fillna(method='ffill', inplace=True)
    values = data[['spindx'] + ['TCMNOM_Y2'] + ['TCMNOM_Y10'] + ['DCOILBRENTEU']
                  + ['GOLDPMGBD228NLBM'] + ['exalus'] + ['exjpus'] + ['exukus']].values
    values = values.astype('float32')
    return values

def predict_levels(num=1):
    """
      Uses predict growth to create the equivalent levels
    """
    values = setup_levels()
    growth_predictions = predict_growth(num)

    graph = []
    for row in values[-30:]:
        graph.append(row)

    for i, _ in enumerate(growth_predictions[0][30:]):
        last_level = graph[-1]
        growth = []
        for asset in range(8):
            growth.append(1 + growth_predictions[asset][i])
        graph.append(multiply(last_level, growth))

    chartist = [[],[],[],[],[],[],[],[]]
    for row in graph:
        chartist[0].append(row[0])
        chartist[4].append(row[1])
        chartist[5].append(row[2])
        chartist[7].append(row[3])
        chartist[6].append(row[4])
        chartist[3].append(row[5])
        chartist[1].append(row[6])
        chartist[2].append(row[7])
    return chartist

def setup_growth():
    """
      Modifies data for growth data
      Output: Historical values, shaped data for model
    """
    data = pd.read_csv(filepath_or_buffer="./static/data/growth_rate.csv", index_col="date")
    data = data.apply(pd.to_numeric, errors = "coerce")
    data['spindx'].replace(0, nan, inplace=True)
    data['spindx'].fillna(method='ffill', inplace=True)

    values = data[['spindx'] + ['TCMNOM_Y2'] + ['TCMNOM_Y10'] + ['DCOILBRENTEU']
                  + ['GOLDPMGBD228NLBM'] + ['exalus'] + ['exjpus'] + ['exukus']].values
    values = values.astype('float32')

    historic_data = array([])
    timestep = 10
    for day in values[-1 * timestep:]:
        historic_data = concatenate((historic_data, day), axis=None)

    return values, historic_data.reshape(1,timestep,8), timestep

def predict_growth(num=1):
    """
      Runs the model, and returns predictions
      Input: num represents the number of days to predict
      Output: An array of last 30 days of historical data plus predictions
    """
    num = int(num)
    values, pred_para, timestep = setup_growth()

    K.clear_session()
    multi_model = load_model("./static/models/8-8-LSTM-G.hdf5",
                             custom_objects={'linex_loss_val': linex_loss_val})

    graph = []
    for row in values[-30:]:
        graph.append(row)

    while num > 0:
        yhat = multi_model.predict(pred_para)
        yhat_pred = yhat.reshape(timestep,8)

        for row in yhat_pred:
            graph.append(row)

        pred_para = yhat_pred.reshape(1, timestep, 8)
        num -= 10

    if num < 0:
        graph = graph[:num]

    chartist = [[],[],[],[],[],[],[],[]]
    for row in graph:
        chartist[0].append(row[0])
        chartist[4].append(row[1])
        chartist[5].append(row[2])
        chartist[7].append(row[3])
        chartist[6].append(row[4])
        chartist[3].append(row[5])
        chartist[1].append(row[6])
        chartist[2].append(row[7])
    return chartist
