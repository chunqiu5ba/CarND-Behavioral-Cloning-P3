#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Nvidia.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/left.jpg "Left Image"
[image4]: ./examples/center.jpg "Center Image"
[image5]: ./examples/right.jpg "Right Image"
[image6]: ./examples/sample.jpg "Normal Image"
[image7]: ./examples/sample2.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode, modified driving speed and image size
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* track1.mp4 and track2.mp4 for autonomous driving 

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model implements Nvidia network, which consists 11 layers:
* 1 Normalization layer
* 3 of convolution layer with 5x5 filter sizes and depths between 24 and 48
* 2 of convolution layer with 3x3 filter sizes and depths 64
* Dropout, Flatten, and 3 fully-connected layers(model.py lines 77-98).

The model includes RELU layers to introduce nonlinearity (code line 82), and the data is normalized in the model using a Keras lambda layer (code line 80). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 92). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 54-57). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 101).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the images from all three cameras, and flipped the images as the augmented data. Dataset from Udacity and simulate training from track two were kept as the final source data.
For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement Nvidia network.

My first step was to use the Nvidia neural network model, the only change is adding a dropout layer before flatten. I thought this model might be appropriate because it is a working model tested in real life.

In order to gauge how well the model was working, I split Udacity image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on both training set and the validation set. 
Model tested working on simulator track one, but failed on track two.

Then I collected extra data: one lap clockwise driving on track one and one lap normal driving on track two, simulator driving on track two was imporved but it started failing on track one. More data were collected on both tracks where the car started off track until it finally can completed both tracks.

I am not satisfied with the situation though, as I realised the failling points were mostly introduced by the "bad driving" from my simulation training. I started from Udacity Data again, add only one more lap simultor driving on track two, the best one from several recordings.

At the end of the process, the vehicle is able to drive autonomously around the both tracks with fewer data.

####2. Final Model Architecture

The final model architecture (model.py lines 77-98) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture

![alt text][image1]

####3. Creation of the Training Set & Training Process

Udacity data set with one lap simulation drive on track two were used in traning. I understand data collection will be more helpful if I can record more recovery driving, but it difficult to switch on/off record while control the steering. I decide to concentrate on center lane driving and leave the recovery task to left and right camera images with steering offset value 0.27.

Images from all three camers were used as this will triple the traning data, even though there are only two laps center lane driving.

Image from left, center, and right camers:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the dataset, I also flipped images and angles thinking that this would generalize the steering data from anti-clockwise driving. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 11722x3x2 = 70332 images/steering angles. Preprocessing is cropping top 50 and bottom 20 pixels, and resizing to 200x66, which is the input shape for Nvidia network.

I finally randomly shuffled the data set and put 20% of the data into a validation set. Traing is done by generator with 64 samples per batch. Epoches is 5 as the loss is close to 0.03 

Training Result:

Epoch 1/5
56265/56265 [==============================] - 1112s - loss: 0.0617 - val_loss: 0.0526

Epoch 2/5
56265/56265 [==============================] - 1144s - loss: 0.0503 - val_loss: 0.0486

Epoch 3/5
56265/56265 [==============================] - 1198s - loss: 0.0430 - val_loss: 0.0434

Epoch 4/5
56265/56265 [==============================] - 1137s - loss: 0.0372 - val_loss: 0.0365

Epoch 5/5
56265/56265 [==============================] - 1120s - loss: 0.0323 - val_loss: 0.0327


Loss
[0.06170803876619365, 0.050340962642766544, 0.042953372616405264, 0.037248355853965971, 0.032260528891586947]

Validation Loss
[0.052595226897472695, 0.048618499031514849, 0.04343254362425842, 0.036487317177901823, 0.03272922731935448]