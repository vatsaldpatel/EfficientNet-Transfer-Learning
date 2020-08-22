# EfficientNet-Transfer-Learning

In this tutorial we will be doing transfer learning on the EfficientNet B0 CNN model with the imagenet weights. We are going to train the model to distinguish between cat and dog.

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
* **epochs** are the number of times the training batches will be fed to the model.
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
1. We will load the EfficientNet B0 version with the imagenet weights.
2. We will freeze the pre-trained weights of the model so that they do not change while training our model.
3. We will 
### 4. Train Model

### 5. Test Model
