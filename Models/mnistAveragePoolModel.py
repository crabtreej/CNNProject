# Jacob - This baseline model was taken from keras.io CIFAR-10 tutorial and modified to be
# a simpler baseline model, so we can compare the results of various additions to the CNN
# for our research project.

# I'm prepending my comments with 'Jacob - ' just to make clear what I've said versus what
# the keras tutorial's comments were.
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
import os
from math import ceil
from sklearn.model_selection import train_test_split
import pickle

# Jacob - This loads the data, reports its shape, turns the classes into a matrix
# of one-hot encoded vectors, and divides the image data by 255 to rescale colors
# Splits training data into 90% training, 10% validation
# returns x_train, y_train, x_val, y_val, x_test, y_test
def prepare_mnist_data(num_classes):
    # The data, split between train, validation, and test sets:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    img_rows = 28
    img_cols = 28
    # Jacob - use train_test_split to get validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    # Convert class vectors to binary class matrices.
    # Jacob - Takes a scalar class label to a 1-hot enocded vector for each training example,
    # giving a 50000 x 10 matrix
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1) 
    
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    
    x_train /= 255
    x_val /= 255
    x_test /= 255

    return x_train, y_train, x_val, y_val, x_test, y_test, input_shape

# Jacob - Compiles this simple baseline CNN model and returns it, ready to train and test.
# Uses SGD as the optimizer, will probably add more inputs to adjust hyperparameters
# as time goes on.
def compile_CNN(input_shape, num_classes, pool_shape, pool_stride):
    # Jacob - I simplified the model to something that probably doesn't really work, but
    # it allows us to see what happens on the simplest possible CNN I can think of
    # 1 convolution -> 1 relu -> 1 pooling -> 1 dense -> softmax
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=pool_shape, strides=pool_stride, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))#, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=pool_shape, strides=pool_stride, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))#, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=pool_shape, strides=pool_stride, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


    # initiate RMSprop optimizer
    # Jacob - I replaced RMSprop with SGD simply because that's what we're used to
    opt = keras.optimizers.sgd(lr=0.01, decay=2e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

batch_size = 32
num_classes = 10
epochs = 100
save_dir = os.path.join(os.getcwd(), 'saved_models')
#model_name = 'max_pool'
#model_name_extension = '.h5'

x_train, y_train, x_val, y_val, x_test, y_test, input_shape = prepare_mnist_data(num_classes)

pool_params = (((2,2), 1), ((2,2), 2), ((2,2), 3), ((2,2), 4), ((3,3), 1), 
               ((3,3), 2), ((3,3), 3), ((3,3), 4), ((4,4), 1), ((4,4), 2), 
               ((4,4), 3), ((4,4), 4), ((8,8), 1), ((8,8), 2), ((8,8), 3), 
               ((8,8), 4), ((8,8), 8), ((14,14), 1), ((14,14), 2), ((14,14), 3), 
               ((14,14), 4), ((14,14), 14))

for param in pool_params:
  K.clear_session()
  tf.reset_default_graph()
  model_name = 'mnist_average_pool'
  model_name_extension = '.h5'
  pooling_shape = param[0]
  model_name += '_' + str(pooling_shape[0]) + '_' + str(pooling_shape[0])
  pooling_stride = param[1]
  model_name += '_' + str(pooling_stride)
  model = compile_CNN(input_shape, num_classes, pooling_shape, pooling_stride)

  print('Using real-time data augmentation. Currently set to do nothing.')
  print(model.summary())

  # Jacob - Reset all values in the ImageDataGenerator to their default/'do nothing'
  # states, that way we aren't applying any extra optimizations at this time.

  # This will do preprocessing and realtime data augmentation:
  datagen = ImageDataGenerator(
      featurewise_center=False,  # set input mean to 0 over the dataset
      samplewise_center=False,  # set each sample mean to 0
      featurewise_std_normalization=False,  # divide inputs by std of the dataset
      samplewise_std_normalization=False,  # divide each input by its std
      zca_whitening=False,  # apply ZCA whitening
      zca_epsilon=1e-06,  # epsilon for ZCA whitening
      rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
      # randomly shift images horizontally (fraction of total width)
      width_shift_range=0.0,
      # randomly shift images vertically (fraction of total height)
      height_shift_range=0.0,
      shear_range=0.,  # set range for random shear
      zoom_range=0.,  # set range for random zoom
      channel_shift_range=0.,  # set range for random channel shifts
      # set mode for filling points outside the input boundaries
      fill_mode='nearest',
      cval=0.,  # value used for fill_mode = "constant"
      horizontal_flip=False,  # randomly flip images
      vertical_flip=False,  # randomly flip images
      # set rescaling factor (applied before any other transformation)
      rescale=None,
      # set function that will be applied on each input
      preprocessing_function=None,
      # image data format, either "channels_first" or "channels_last"
      data_format=None,
      # fraction of images reserved for validation (strictly between 0 and 1)
      validation_split=0.0) 

  # Compute quantities required for feature-wise normalization
  # (std, mean, and principal components if ZCA whitening is applied).
  # Jacob - currently doesn't actually do anything.
  datagen.fit(x_train)

  # Jacob - I reduced the number of steps per epoch to something feasible for my
  # machine because I cannot use the GPU. Please use something like 
  # ceil(len(x_train) / batch_size) if you can use the GPU.
  arbitrary_steps_per_epoch = 1
  correct_steps_per_epoch = ceil(len(x_train) / batch_size)

  #steps_per_epoch = arbitrary_steps_per_epoch
  steps_per_epoch = correct_steps_per_epoch


  # Fit the model on the batches generated by datagen.flow().
  training_history = model.fit_generator(datagen.flow(x_train, y_train,
                                   batch_size=batch_size),
                      epochs=epochs,
                      validation_data=(x_val, y_val),
                      workers=4, steps_per_epoch=steps_per_epoch)

  # Save model and weights
  if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
  model_path = os.path.join(save_dir, model_name + model_name_extension)
  model.save(model_path)
  print('Saved trained model at %s ' % model_path)

  # Jacob - Save the training history for the model as a pickle file, as 
  # we can't recreate that for free like we can the final results.
  train_hist_file = open(os.path.join(save_dir, model_name + '.pckl'), 'wb')
  pickle.dump(training_history.history, train_hist_file)
  train_hist_file.close()

  # Score trained model.
  scores = model.evaluate(x_test, y_test, verbose=1)
  print(f'For Model: {model_name}')
  print('Test loss:', scores[0])
  print('Test accuracy:', scores[1])			
