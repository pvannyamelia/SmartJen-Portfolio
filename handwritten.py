# =============================================================================
# Handwritten problem
#
# Here, I try to create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Desired accuracy AND validation_accuracy > 95%
# =============================================================================

import tensorflow as tf


def handwritten():
    mnist = tf.keras.datasets.mnist

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')> 0.95 and logs.get('val_accuracy')> 0.95):
                print("\nReached desired accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, callbacks=[callbacks],
              validation_data=(x_test, y_test), verbose=1)
    return model


# The code below is to save the model as a .h5 file.
if __name__ == '__main__':
    model = handwritten()
    model.save("handwritten.h5")