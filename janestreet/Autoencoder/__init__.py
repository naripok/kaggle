#!/usr/bin/env python

import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

if __name__ == "__main__":
    import time
    import numpy as np
    import keras.applications as kapp
    from keras.datasets import cifar10

    def demo():
        (x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
        batch_size = 8
        x_train = x_train[:batch_size]
        x_train = np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)
        model = kapp.VGG19()
        model.compile(optimizer='sgd', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("Running initial batch (compiling tile program)")
        y = model.predict(x=x_train, batch_size=batch_size)

        # Now start the clock and run 10 batches
        print("Timing inference...")
        start = time.time()
        for i in range(10):
            y = model.predict(x=x_train, batch_size=batch_size)
        print("Ran in {} seconds".format(time.time() - start))

    demo()

