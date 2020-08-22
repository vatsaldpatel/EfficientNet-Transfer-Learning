# -*- coding: utf-8 -*-
"""
@author: Vatsal patel
"""

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import the Desired Version of EfficientNet
from tensorflow.keras.applications import EfficientNetB0

# Variables
NUM_CLASSES = 2

train_path = "Dataset/training_set/training_set/"
valid_path = "Dataset/validation_set/validation_set/"
test_path = "Dataset/test_set/test_set/"

epochs = 5

model_save_location = "Model/EfficientNet"


img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def build_model(NUM_CLASSES):
    inputs = layers.Input(shape=(224, 224, 3))
    x = img_augmentation(inputs)

    #Using the imported version of EfficientNet
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    
def test_model(model,test_batches):
    #Testing the Model
    test_labels = test_batches.classes
    print("Test Labels",test_labels)
    print(test_batches.class_indices)
    
    predictions = model.predict(test_batches,steps=len(test_batches),verbose=0)
    
    acc = 0
    for i in range(len(test_labels)):
        actual_class = test_labels[i]
        if predictions[i][actual_class] > 0.5 : 
            acc += 1
    print("Accuarcy:",(acc/len(test_labels))*100,"%")

if __name__ == "__main__":
    model = build_model(NUM_CLASSES)
    unfreeze_model(model)
    
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
        directory=train_path, target_size=(224,224), batch_size=10)
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
        directory=valid_path, target_size=(224,224), batch_size=10)
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
        directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)
    
    _ = model.fit(train_batches, epochs=epochs, validation_data=valid_batches, verbose=1)
    
    #Testing the Model
    test_model(model,test_batches)
    
    # Save the tensorflow Model
    model.save(model_save_location)
