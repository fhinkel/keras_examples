from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

# Load the data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(train_images.shape)

model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Preprocess the data
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32") / 255

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)
              
              
              

