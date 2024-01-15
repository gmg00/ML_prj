import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import initializers

def get_data(filename):
    df = pd.read_csv(filename, names = ['all'])
    df['all'] = df['all'].str.strip()
    df['all1']  = df['all'].apply(lambda x: np.array(x.split(' ')))
    df[['target','attr1','attr2','attr3','attr4','attr5','attr6','id']] = pd.DataFrame(df.all1.tolist(), index= df.index)
    df = df.sample(frac=1).reset_index(drop=True)
    return df.drop(columns=['all','all1'])

if __name__ == "__main__":
    df = get_data('/mnt/c/Users/HP/Desktop/UNI/LM_1/MachineLearning/MONK/monks-1.train')
    df_test = get_data('/mnt/c/Users/HP/Desktop/UNI/LM_1/MachineLearning/MONK/monks-1.test')

    X_train, y_train = df.drop(columns=['target','id']), df['target'].apply(lambda x: int(x))
    X_test, y_test = df_test.drop(columns=['target','id']), df_test['target'].apply(lambda x: int(x))

    ohe = OneHotEncoder()
    ohe.fit(X_train, y_train)
    X_train = ohe.transform(X_train).toarray()
    X_test = ohe.transform(X_test).toarray()

    input_data = Input(shape=(17,))
    hidden = Dense(3,activation='tanh',kernel_initializer='random_normal',bias_initializer='zeros')(input_data)
    outputs = Dense(1, activation='tanh',kernel_initializer='random_normal',bias_initializer='zeros')(hidden)

    model = Model(inputs=input_data, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3))
    model.summary()

    history = model.fit(X_train, y_train, validation_split=0.5, epochs = 1000,
                        callbacks = [EarlyStopping(monitor='val_loss',
                                    patience=50,
                                    verbose=1),
                                    ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.25,
                                        patience=10,
                                        verbose=1)],
                        use_multiprocessing=True)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(loss,'-')
    plt.plot(val_loss,'--')
    plt.show()

    pred = model.predict(X_test)

    soglia = 0 
    pred[pred <= soglia] = 0
    pred[pred > soglia] = 1

    df_sol = pd.DataFrame()
    df_sol['true'] = y_test
    df_sol['pred'] = pred

    print((df_sol['true'] == df_sol['pred']).value_counts())