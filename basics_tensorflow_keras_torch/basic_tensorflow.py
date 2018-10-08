import tensorflow as tf

mnist = tf.keras.datasets.mnist

print(mnist)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# We can use the model below which will be more precise.
'''
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
'''

# However, even the simplest model gives 90% accuracy:
model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
print(model.evaluate(x_test, y_test))
