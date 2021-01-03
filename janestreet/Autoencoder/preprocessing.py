import numpy as np
import pandas as pd

def get_df():
    print('unziping data...')
    df = pd.read_csv('input/train.csv.zip')
    print('done!\ndataframe head:\n')
    print(df.head())
    return df

def get_train_data():
    train = get_df()
    train = train.query('date > 85').reset_index(drop = True)
    train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns}) #limit memory use
    train.fillna(train.mean(),inplace=True)
    train = train.query('weight > 0').reset_index(drop = True)
    #train['action'] = (train['resp'] > 0).astype('int')
    train['action'] =  (  (train['resp_1'] > 0.00001 ) & (train['resp_2'] > 0.00001 ) & (train['resp_3'] > 0.00001 ) & (train['resp_4'] > 0.00001 ) &  (train['resp'] > 0.00001 )   ).astype('int')
    features = [c for c in train.columns if 'feature' in c]

    resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

    X = train[features].values
    y = np.stack([(train[c] > 0.000001).astype('int') for c in resp_cols]).T #Multitarget

    f_mean = np.mean(train[features[1:]].values,axis=0)

    return X, y

if __name__ == "__main__":
    # load traning data
    get_train_data()

