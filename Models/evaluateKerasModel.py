import os
import argparse
import pickle
import matplotlib.pyplot as plt

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
        print(f'The model did not exist at \'{model_path}\'.')
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

# Jacob - Returns x data and y-labels for cifar-10 test, ready to be passed into the model
def get_cifar_data():
    # Jacob - only load the data we need 
    from keras.datasets import cifar10

    # Jacob - cifar-10 has ten classes
    num_classes = 10

    # Jacob - The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Jacob - Prepare the testing data (assuming we're using cifar10, we divide by 255)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_test = x_test.astype('float32')
    x_test /= 255

    return x_test, y_test

# Jacob - returns x data and y-labels from the test set for MNIST, ready to be tested 
def get_mnist_data():
    # Jacob - get MNIST data
    from keras.datasets import mnist
    from keras import backend as K

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
   
    # Jacob - MNIST has ten classes
    num_classes = 10
    img_rows = 28
    img_cols = 28
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Jacob - make sure it's the right shape (from the Keras MNIST tutorial)
    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_test = x_test.astype('float32')
    x_test /= 255

    return x_test, y_test

# Jacob - This iterates through each epoch in the training history
# saved in the pickle file, and prints the validation and training
# accuracy. 
def report_training_history(train_hist_path, make_graph, graph_title):
    # Jacob - Read in the file
    train_hist_file = open(train_hist_path + '.pckl', 'rb')
    hist_dict = pickle.load(train_hist_file)
    train_hist_file.close()

    # Jacob - Get each list in the dictionary
    val_loss_hist = hist_dict['val_loss']
    train_loss_hist = hist_dict['loss']
    val_acc_hist = hist_dict['val_acc']
    train_acc_hist = hist_dict['acc']

    # Jacob - Zip everything together and then print out the stats.
    # Now that we pickle all the training results too, we can just
    # come back at some later time to actually make graphs and such
    num_epochs = len(val_loss_hist)
    print(f'Training Stats per Epoch. {num_epochs} total epochs.')
    epoch = 1
    for val_loss, train_loss, val_acc, train_acc in zip(val_loss_hist, train_loss_hist, val_acc_hist, train_acc_hist):
        print(f'Epoch {epoch} / {num_epochs}: \nValidation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\nTraining Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}\n')
        epoch += 1

    if make_graph:
        graph_error(val_loss_hist, val_acc_hist, train_loss_hist, train_acc_hist, graph_title)

# Jacob - This takes the validation and training losses and accuracies and
# makes a 2x2 Pyplot graph out of them. Will later accept an overall
# title for the graph and save that to disk.
def graph_error(val_losses, val_acc, train_losses, train_acc, graph_title):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    train_loss_ax, val_loss_ax, train_acc_ax, val_acc_ax = axes.flatten()

    train_loss_ax.plot(train_losses)
    train_loss_ax.set_xlabel('Epoch')
    train_loss_ax.set_ylabel('Training Loss')
    train_loss_ax.set_title('Training Loss vs. Epochs')

    train_acc_ax.plot(train_acc)
    train_acc_ax.set_xlabel('Epoch')
    train_acc_ax.set_ylabel('Training Accuracy')
    train_acc_ax.set_title('Training Accuracy vs. Epochs')

    val_loss_ax.plot(val_losses)
    val_loss_ax.set_xlabel('Epoch')
    val_loss_ax.set_ylabel('Validation Loss')
    val_loss_ax.set_title('Validation Loss vs. Epochs')

    val_acc_ax.plot(val_acc)
    val_acc_ax.set_xlabel('Epoch')
    val_acc_ax.set_ylabel('Validation Accuracy')
    val_acc_ax.set_title('Validation Accuracy vs. Epochs')

    fig.tight_layout()
    plt.show()

    # Jacob - TODO: take a name for the graph and save it to disk
    if graph_title:
        # Jacob - save the graph

# Jacob - use argparse to get the path to the model, load in test data, and then
# evaluate the model on the test data. If you have an already existing model and you
# just want to see the test results, just import this file and call report_test_metrics
if __name__ == '__main__':
    default_model_dir = os.path.join(os.getcwd(), 'saved_models')

    # Jacob - using argparse just to make this nice to use
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', required=True, help='File name of saved model, not including extension')
    parser.add_argument('--model_dir', default=default_model_dir, help='Specify if model isn\'t in ./saved_models/')
    parser.add_argument('-t', '--show_training', action='store_true', help='Specify if you want to see the training history (validation error, testing error, and loss)')
    parser.add_argument('-g', '--make_graph', action='store_true', help='Graph training and validation accuracy and losses on a 2x2 pyplot, must specify show_training')
    parser.add_argument('--mnist', action='store_true', help='set flag if analyzing results from MNIST dataset')
    parser.add_argument('--title', help='specify title if you want the resulting graph saved')
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model_name + '.h5')

    # Jacob - these take forever to import, so there's no reason to
    # do it until we know we need them (e.g. someone could specify -h for arg).
    import keras
    from keras.models import load_model

    # Jacob - different routines for loading the different datasets
    model = reload_keras_model(model_path)
    if args.mnist:
        x_test, y_test = get_mnist_data()
    else:
        x_test, y_test = get_cifar_data()

    report_test_metrics(model, x_test, y_test)

    if args.show_training:
        report_training_history(os.path.join(args.model_dir, args.model_name), args.make_graph, args.title)
