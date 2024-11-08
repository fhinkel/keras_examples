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

# configures the model for training
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
              
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(predictions[0]) 
print(predictions[0][7]) 

print(test_labels[0]) 

# evaluate the model on new data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


