import os
import argparse

# Jacob - takes a path to a model saved in a .h5 file 
# and returns a keras model that is ready for evaluation on a test set
def reload_keras_model(model_path):
    model = None

    if os.path.isfile(model_path):
        model = load_model(model_path)

        # Jacob - We don't actually need an optimizer, but we are required
        # to have one to compile the model. Maybe add an option for metrics.
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
    else:
        print(f'The model did not exist at {model_path}.')
        print(f'The code assumes os.getcwd()/saved_models contains the model.')
        print(f'If that\'s not the case, make sure to input the path as an argument.')
        quit()

    return model

# Jacob - we can turn this into something fancy later, but for now it just prints the
# accuracy the same way the keras tutorial does.
def report_test_metrics(model, x_test, y_test):
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


# Jacob - use argparse to get the path to the model, load in test data, and then
# evaluate the model on the test data. If you have an already existing model and you
# just want to see the test results, just import this file and call report_test_metrics
if __name__ == '__main__':
    default_model_dir = os.path.join(os.getcwd(), 'saved_models')

    # Jacob - using argparse just to make this nice to use
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', required=True, help='File name of saved model, include .h5')
    parser.add_argument('--model_dir', default=default_model_dir, help='Specify if model isn\'t in ./saved_models/')
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model_name)

    # Jacob - these take forever to import, so there's no reason to
    # do it until we know we need them (e.g. someone could specify -h for arg).
    import keras
    from keras.datasets import cifar10
    from keras.models import load_model

    # Jacob - assume cifar10 for now, we can adjust this if needed.
    num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Jacob - Prepare the testing data (assuming we're using cifar10, we divide by 255)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_test = x_test.astype('float32')
    x_test /= 255

    model = reload_keras_model(model_path)
    report_test_metrics(model, x_test, y_test)
