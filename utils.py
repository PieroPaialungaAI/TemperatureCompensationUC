from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_absolute_error
import pandas as pd 
import numpy as np 

def import_data(file_name = 'Field_Data.xlsx', sheet_name = 'For Piero'):
    data = pd.read_excel(file_name, sheet_name = sheet_name)
    data = data.drop([0,1]).reset_index().drop('index',axis=1).dropna()
    data = data.reset_index().drop('index',axis=1).astype('float')
    return data

def preprocess_data(data):
    X = np.array(data[[c for c in data.columns if c.split('.')[0]!='Resistance']])
    X_columns = [c for c in data.columns if c.split('.')[0]!='Resistance']
    min_max = MinMaxScaler()
    Y_columns = [c for c in data.columns if c.split('.')[0]=='Resistance']
    Y = np.array(data[Y_columns])
    index_list = np.arange(0,len(X))
    split_value = int(len(index_list)*0.8)
    train_list = index_list[0:split_value]
    test_list = index_list[split_value:]
    min_max.fit(X[train_list])
    X_raw = X.copy()
    X = min_max.transform(X)
    return {'train_list':train_list,'test_list':test_list,'X':X,'Y':Y,'X_columns':X_columns,'Y_columns':Y_columns,'X_raw':X_raw}

def model_data(data_dict):
    train_list = data_dict['train_list']
    test_list = data_dict['test_list']
    Y = data_dict['Y']
    X = data_dict['X']
    X_raw = data_dict['X_raw']
    target_columns = data_dict['Y_columns']
    X_columns = data_dict['X_columns']
    pred_list = []
    real_list = []
    X_train = X[train_list]
    num = len(Y.T)
    for i in range(num):
        rf = RandomForestRegressor()
        rf = rf.fit(X_train,Y[train_list,i])
        pred = rf.predict(X[test_list])
        real_i = Y[test_list,i]
        real_list.append(real_i)
        pred_list.append(pred)
    real_list = np.array(real_list).T
    df = pd.DataFrame(real_list,columns = target_columns)
    i=0
    k=0
    for x in X_columns:
        df[x] = X_raw[test_list,i]
        i=i+1
    for x in target_columns:
        df[x+'_pred'] = pred_list[k]
        k=k+1
    df.to_csv('result.csv')
    return df 
