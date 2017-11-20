import csv
import numpy as np
import cv2
import random
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Convolution2D, Lambda, MaxPooling2D, Dropout
from keras.layers import Cropping2D
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('datadir', 'data', "Directory with training data. Must have driving_log.csv and IMG directory")
steering_correction = 0.20
data_dir = "./" + FLAGS.datadir + "/"
image_dir = data_dir + "IMG/"
model_name = FLAGS.datadir + ".h5"
lines = []

print("**** Read Training Data from {} and write model to {} ****".format(data_dir, model_name))
#
# Given the directory name read the driving log into lines for processing
# later through a generator function.
#
def read_data(wd) :

    fname = wd + 'driving_log.csv'
    pruned = []
    with open(fname, 'r') as csvfile :
        alllines = csv.reader(csvfile)
        for line in alllines:
            steer = float(line[3])   # steering measurement
            if (steer <= 0.001) :  # Discard 70% of the images with steering angle close to 0.
                if (random.randint(1,100) < 70) :
                    continue
            pruned.append(line)
    return pruned

#
# Generator function to give batches of images.
#
# Data Augmentation: Loops through each line in the driving log file and does the following to generate a "batch"
#       Add Center image
#       Add a flipped Center image
#       Add Left image with Steering correction
#       Add a flipped Left image
#       Add Right Image with Steering correction
#       Add a flipped Right image
# As a result, number of images in each batch returned by this generator is 6*batch_size
#  'train' controls the training vs validation generator.
#     Validation generator uses only center images (no multiple as a result)
#
def data_generator(lines, batch_size=32, train=True) :

    n_images = len(lines)
    print("Training {} Num of Images {}".format(train, n_images * 6))

    while 1:

        shuffle(lines)

        for start_idx in range(0, n_images, batch_size):

            lines_batch = lines[start_idx: start_idx+batch_size]
            images = []
            steering = []

            for line in lines_batch :

                steer = float(line[3])

                ### Read center image.
                ci_path = line[0] # center image path
                ci = image_dir + ci_path.split('/')[-1]  # center image filename

                image = cv2.imread(ci)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                steering.append(steer)

                if (train) :  # Add flipped image only if training
                    images.append(cv2.flip(image,1))
                    steering.append(steer * (-1.0))

                if (train) :  # Add Left and Right images only for training
                    ### Read Left image
                    li_path = line[1] # left image path
                    li = image_dir + li_path.split('/')[-1]  # left image filename

                    image = cv2.imread(li)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    steering.append(steer+steering_correction)

                    # Augment by flipping it
                    images.append(cv2.flip(image,1))
                    steering.append((steer+steering_correction) * (-1.0))


                    ### Read Right  image
                    ri_path = line[2] # right image path
                    ri = image_dir + ri_path.split('/')[-1]  # right image filename

                    image = cv2.imread(ri)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    steering.append(steer-steering_correction)

                    # Augment by flipping it
                    images.append(cv2.flip(image,1))
                    steering.append((steer-steering_correction) * (-1.0))

            X_train = np.array(images)
            y_train = np.array(steering)

            yield sklearn.utils.shuffle(X_train, y_train)


# Implements something similar to NVIDIA model
def define_model() :

    model = Sequential()

    # Crop the images - top 50 and botton 25 pixels.

    model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=(160,320,3)))

    model.add(Lambda(lambda x: x/255.0 -0.5)) # Normalize data

    model.add(Convolution2D(24, 5, 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(36, 5, 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convoluton with Dropout
    model.add(Convolution2D(48, 5, 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Dense with Dropout
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(10, activation='relu'))

    model.add(Dense(1)) # Output steering angle.

    # Compile the model with Adam optimizer with Mean Square Error as the loss function.

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model

#### Main ####

random.seed(47)
lines = read_data(data_dir)
train_data, valid_data = train_test_split(lines, test_size=0.2)

print("Total {} Training {}, Validation {}".format(len(lines), len(train_data), len(valid_data)))

train_generator = data_generator(train_data, batch_size=8)
valid_generator = data_generator(valid_data, batch_size=8, train=False)

model = define_model()

# Print the layer details
model.summary()

# Samples per epoch is number of lines in the training_data * 6, due to data augmentation.
model.fit_generator(train_generator, samples_per_epoch = len(train_data)*6,
                    validation_data=valid_generator,
                    nb_val_samples=len(valid_data), nb_epoch=5)

model.save(model_name)

exit()
