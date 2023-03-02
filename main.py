import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10
image_vector_size = 28*28

x_train = tf.keras.utils.normalize(x_train, 1)
x_test = tf.keras.utils.normalize(x_test, 1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=True, validation_split=.1)
loss, accuracy  = model.evaluate(x_test, y_test, verbose=True)
print("loss: ", loss)
print("accuracy: ", accuracy)