import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2 as cv
import h5py
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras.models import load_model

base_dir = 'imgs'
train_dir = os.path.join(base_dir,'training')
validation_dir = os.path.join(base_dir,'validation')
CATEGORIES = ["5-euro","10-euro","20-euro","50-euro"]

train_5_dir = os.path.join(train_dir, '5-euro')
train_10_dir = os.path.join(train_dir, '10-euro')
train_20_dir = os.path.join(train_dir, '20-euro')
train_50_dir = os.path.join(train_dir, '50-euro')
val_5_dir = os.path.join(validation_dir, '5-euro')
val_10_dir = os.path.join(validation_dir,'10-euro')
val_20_dir = os.path.join(validation_dir, '20-euro')
val_50_dir = os.path.join(validation_dir,'50-euro')

print('total training 5 images:', len(os.listdir(train_5_dir)))
print('total training 10 images:', len(os.listdir(train_10_dir)))
print('total training 20 images:', len(os.listdir(train_20_dir)))
print('total training 50 images:', len(os.listdir(train_50_dir)))
print('total validation 5 images:', len(os.listdir(val_5_dir)))
print('total validation 10 images:', len(os.listdir(val_10_dir)))
print('total validation 20 images:', len(os.listdir(val_20_dir)))
print('total validation 50 images:', len(os.listdir(val_50_dir)))

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

# Note that the validation data should not be augmented!
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        classes=['5-euro','10-euro','20-euro','50-euro'],
        target_size=(250, 250),  # All images will be resized to 150x150
        batch_size=50,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

# Flow validation images in batches of 20 using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        classes=['5-euro','10-euro','20-euro','50-euro'],
        target_size=(250, 250),
        batch_size=25,
        class_mode='categorical')

num_classes= len(train_generator.class_indices)
train_labels = train_generator.classes
print(train_labels)
# convert the training labels to categorical vectors 
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
print("number of classes:",num_classes)
print("number of labels:",train_labels)
num_classes_val = len(validation_generator.class_indices)
validation_labels = validation_generator.classes
print(validation_labels)
# convert the training labels to categorical vectors 
validation_labels = tf.keras.utils.to_categorical(validation_labels, num_classes=num_classes_val)
print("number of classes validation:",num_classes_val)
print("number of labels validation:",validation_labels)

# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(250, 250, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 128 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(128, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

#Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)

# Create output layer with four nodes and softmax activation
output = layers.Dense(num_classes, activation='softmax')(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully 
# connected layer + softmax output layer
model = Model(img_input, output)

# or categorical_crossentropy
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

history = model.fit(
      train_generator,
      steps_per_epoch=100,  # 5000 images = batch_size * steps
      epochs=10,
      validation_data=validation_generator,
      validation_steps=80,  # 2000 images = batch_size * steps
      verbose=2)

model.save('euro-CNN.h5')

# evaluating and printing results 
score = model.evaluate(validation_generator, verbose = 0) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.show()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

plt.show()
