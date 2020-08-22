# EfficientNet-Transfer-Learning

In this tutorial we will be doing transfer learning on the EfficientNet B0 CNN model with the imagenet weights. We are going to re-train the model to distinguish between cat and dog.

### What is Transfer Learning?
In simple terms transfer learning is the method where we can reuse a pre-trained model as a starting point of our own object classification model.

### Pre-requisite
1. Tensorflow 2.3.0

You can easily upgrade your tensorflow by running the following command.
~~~~
pip install tensorflow --upgrade
~~~~

### Dataset
* The dataset that I'm using here can be downloaded from Kaggle with this link.(https://www.kaggle.com/tongpython/cat-and-dog)
* The dataset is divided into training and testing set, but for transfer learning we also need a validation set. For validation set divide the testing set in roughly two equal parts. i.e. testing set and validation set.

## Walking Through The Code

### 1. Imports
~~~~python
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import the Desired Version of EfficientNet
from tensorflow.keras.applications import EfficientNetB0
~~~~
* You can import the desired version of EfficientNet (B0 to B7) according to your need. If you are training this model for an edge or mobile device, then use the B0 version.

### 2. Variables
~~~~python
NUM_CLASSES = 2

train_path = "Dataset/training_set/training_set/"
valid_path = "Dataset/validation_set/validation_set/"
test_path = "Dataset/test_set/test_set/"

epochs = 5

model_save_location = "Model/EfficientNet"
~~~~
* **NUM_CLASSES** is the different object the model will be distinguishing. In our case 2 i.e. cat and dog.
* The next are the paths to the training, validation and testing dataset directory.
* **epochs** are the number of times the training batches will be fed to the model. The number of epochs will change according to the dataset.
* **model_save_location** is the location where you want to save the trained model.

### 3. Build Model
~~~~python
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
~~~~
**build_model and unfreeze_model**
1. We will load the EfficientNet B0 version with the imagenet weights.
2. We will freeze the pre-trained weights of the model so that they do not change while training our model.
3. We will add some custom layers on the top of the model and add our output layer with *NUM_CLASSES* and activation function as *softmax*.
4. We will unfreeze the top 20 layers except the BatchNormalization layers, so that the model can learn from our dataset.

### 4. Train Model
~~~~python
model = build_model(NUM_CLASSES)
unfreeze_model(model)

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

_ = model.fit(train_batches, epochs=epochs, validation_data=valid_batches, verbose=1)
~~~~
* We will get our model by calling **build_model**, after that we will call **unfreeze_model**.
* We will preprocess our images and make their batches by using the *ImageDataGenerator* function on the training, testing and validation images.
* We will provide the **train_batches** and **valid_batches** to *model.fit()* function.

### 5. Test Model
~~~~python
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
~~~~
* We will provide the **test_batches** to the *model.predict()* function which will return the probabilities of each images.
* Finally we will print the accuracy as predicted by the model and save the model to the **model_save_location**.

##### THANK YOU !!
