if __name__ == "__main__":
    from .create_autoencoder import create_autoencoder
    from keras.callbacks import EarlyStopping
    from .preprocessing import get_train_data

    TRAINING = True

    # load traning data
    X, y = get_train_data()

    autoencoder, encoder = create_autoencoder(X.shape[-1],y.shape[-1],noise=0.1)
    if TRAINING:
        autoencoder.fit(X,(X,y),
                        epochs=1002,
                        batch_size=16384,
                        validation_split=0.1,
                        callbacks=[EarlyStopping('val_loss',patience=10,restore_best_weights=True)])
        encoder.save_weights('./input/encoder.hdf5')
    else:
        encoder.load_weights('../input/encoder.hdf5')
    encoder.trainable = False
