from keras.layers import BatchNormalization, GaussianNoise, Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam


def create_autoencoder(input_dim,output_dim,noise=0.05):
    i = Input((input_dim,))
    encoded = BatchNormalization()(i)
    encoded = GaussianNoise(noise)(encoded)
    encoded = Dense(640,activation='relu')(encoded)
    decoded = Dropout(0.2)(encoded)
    decoded = Dense(input_dim,name='decoded')(decoded)
    x = Dense(320,activation='relu')(decoded)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(output_dim,activation='sigmoid',name='label_output')(x)

    encoder = Model(inputs=i,outputs=encoded)
    autoencoder = Model(inputs=i,outputs=[decoded,x])

    autoencoder.compile(optimizer=Adam(0.001),loss={'decoded':'mse','label_output':'binary_crossentropy'})
    return autoencoder, encoder
