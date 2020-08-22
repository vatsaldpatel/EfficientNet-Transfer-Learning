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
