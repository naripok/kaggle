from keras.layers import BatchNormalization, GaussianNoise, Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.backend import sigmoid


def swish(x, beta = 1):
    return (x * sigmoid(beta * x))


def create_model(hp,input_dim,output_dim,encoder):
    inputs = Input((input_dim,))

    x = encoder(inputs)
    x = Concatenate()([x,inputs]) #use both raw and encoded features
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('init_dropout',0.0,0.5))(x)

    for i in range(hp.Int('num_layers',1,5)):
        x = Dense(hp.Int('num_units_{i}',128,256))(x)
        x = BatchNormalization()(x)
        x = Lambda(swish)(x)
        x = Dropout(hp.Float(f'dropout_{i}',0.0,0.5))(x)
    x = Dense(output_dim,activation='sigmoid')(x)
    model = Model(inputs=inputs,outputs=x)
    model.compile(optimizer=Adam(hp.Float('lr',0.00001,0.1,default=0.001)),loss=BinaryCrossentropy(label_smoothing=hp.Float('label_smoothing',0.0,0.1)),metrics=[tf.keras.metrics.AUC(name = 'auc')])
    return model
