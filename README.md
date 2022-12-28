# Objective

In this project we take a database of images, that some of them are corrupted, and trying to classify them 
using multilayered CNN model.

# Process

* Resize image quality to image thumbnails
* Transform thumbnails to numpy 4-d array (#instances, #Hlength, #Vlength, #RGB)
* Check images sizes are equal
* Create multilayered CNN model
* Split the train and test
* Check accuracy

# Tools

PIL, Tensorflow Keras, Sklearn

# Results

0.96 accuracy

# Other notes

* db size 370
* There are examples of corrupted photos and good photos
* The trained model is saved, you can use it. Try to run 'runExamples.py' to see the predictions
* There is a numpy file containing the data, and the target variable (of the predicted model and of the example)   
