

# ** Behavioral Cloning Project **

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/image_augmentation.png "Augmented Image"

## Rubric Points

### Here are the [rubric points](https://review.udacity.com/#!/rubrics/432/view)  

---
### Files Submitted & Code Quality

Submission has the following files at the [Github repository](https://github.com/gvogety/udacity-carnd-behavioral-cloning)
* model.py Contains generator function to load large amount of data, model definition and training code.
* drive.py For driving the car in autonomous mode. This is unmodified.
* model.h5 Contains a trained convolution neural network.
* video.mp4 Video of sample run on with the trained model on Track 1
* writeup_report.md This file on [GitHub](https://github.com/gvogety/udacity-carnd-behavioral-cloning/blob/master/writeup_report.md)

#### 2. Submission includes functional code

Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on NVIDIA's convolution neural network.

I first reduced the size of the images by using a Cropping layer to reduce the number of features in the input.

I normalized the data using Keras lambda layer.

For the convolutions, I use 3 layers with 5x5 kernels with 24, 36 and 48 depths. Following that, I used two layers with 3x3 kernels and 64 depths each. After flattening, I used 4 Dense layers with widths of 1164, 100, 50 and 10 to output the steering angle. I experimented with various combinations of kernel sizes, depths and widths of the Dense layers. This set seems to give the most repeatable performance on the simulator in the Autonomous mode. Performance is measured by the number of loops that the car can make on Track 1 (minimum required is one loop). I effectively did a grid search on kernel, depths, widths, dropouts and the number of layers. Unfortunately, improvements in validation accuracy did not result in keeping the car on the track. So I abandoned relying on the validation accuracy to predict final performance. Always take it for a spin.


The model includes RELU layers to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

I started with no dropouts. With the training data I have, the laps were almost perfect with the speed around 10, never veering into shoulders, even around the curves. But to reduce overfitting, I started introducing dropouts. Initially I had 2 Dropout layers, one in Convolution layers and another in Dense layers. Then I increased to 4, 2 in Convolution layers and 2 in Dense layers. With this architecture, the car was traveling a bit more erratically, going into shoulders at speed around 10. When I increase the speed, sometimes it completely veers off the road.

So I decided to finalize my model with just two Dropout layers (not exceeding 50%)

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track for 10-15 laps with multiple launches. I noticed each launch was different, possibly due to random variables used in the simulator.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, doing multiple runs around corners approaching from different angles, doing reverse laps etc. I focused on the corners at different speeds. However, I never tried to recover from a totally-off-the-road position. As a result, the model would not recover from any such situation in the simulator (possible at high speeds)

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use NVIDIA model and try to reduce the number of parameters and hence the model size as a result but still achieve a functional model.

My first step was to use a convolution neural network model similar to the NVIDIA model.. but only 2 or 3 convolution layers and 3 Dense layers. Even though the car was able to stay on the road most of the time, navigating the curves was not reliable. Moreover, number of variables/weights trained was reaching 8-10 million and the corresponding model size was exceeding 100MB. As I introduced more Convolution layers and Dense layers, there was better performance on the road, w.r.t navigating the curve and reduced the number of trained variables to 1.7M and model size to 20MB.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

To combat overfitting, I introduced Dropout layers.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Here is the [Video](https://github.com/gvogety/udacity-carnd-behavioral-cloning/blob/master/video.mp4). Additional videos are available [here](https://github.com/gvogety/udacity-carnd-behavioral-cloning/tree/master/video). Driving at high speeds (more than 20) is not reliable. With more training data or training with speed also (as opposed to just steering angle), better results can be achieved at higher speeds.

#### 2. Final Model Architecture

The final model architecture (model.py lines 124-170) consisted of a convolution neural network with the following layers and layer sizes.

Here is a summary of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded couple of laps on track one using center lane driving.

I then recorded couple of laps in the reverse direction. Then I recorded several runs around the curves while approaching the curve from the center, left and right. I then recorded the vehicle recovering from left and right side after navigating the curve.

I did not use track two for this submission.

Of the collected images, I discarded 70% of the images with driving angle close 0. This allowed me to get more balanced data-set at various angles.

I used both left and right camera images and also flipped them. Experimented with driving angle correction of 0.1, 0.15 and 0.2. Final submission included driving correction of 0.2

To augment the data set, I also flipped images and angles thinking various curves and left and right side driving can be better navigated.  Example of augmentations is as follows

![alt text][image2]

After the collection process, I had around 5000 data points. I put 20% of the data into a validation set.
Data collection and augmentation was done in a generator function to be able to accommodate large data sets.
After augmentation, I had 22,700 data points. Preprocessing (basic normalization and cropping) step was part of the model and no additional preprocessing was done.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by validation accuracy staying very steady afetr 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
