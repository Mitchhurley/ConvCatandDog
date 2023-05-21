import keras.layers
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

#
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "cats-and-dogs",
    labels="inferred",
    label_mode="binary",
    batch_size=32,
    image_size=(100, 100),
    shuffle=True,
    seed=123,
    validation_split=0.2,  # Split the data into train and validation sets
    subset="training",
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "cats-and-dogs",
    labels="inferred",
    label_mode="binary",
    batch_size=32,
    image_size=(100, 100),
    shuffle=True,
    seed=123,
    validation_split=0.2,  # Split the data into train and validation sets
    subset="validation",
)

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])


model = models.Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255), #does what the flatten method would do
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='tanh'),
    layers.Dense(1, activation='sigmoid')
])
op = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=None)

model.compile(optimizer=op,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=50
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "cats-and-dogs",
    labels="inferred",
    label_mode="binary",
    batch_size=32,
    image_size=(100, 100),
    shuffle=True,
    seed=123,
)

pd.DataFrame(history.history)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_loss, test_acc = model.evaluate(test_dataset)
print("Test accuracy:", test_acc)

model.save("model.h5", include_optimizer = False)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
