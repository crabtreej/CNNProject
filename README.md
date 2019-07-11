# CNNProject
Project Github for Neural Nets (5526)

The overarching goal of this project was to explore the effects of various pooling strategies in CNNs, specifically max and average pooling using various parameters. This project provided some useful empirical data on how to choose pooling strategies based on the type of data, and what to expect in terms of accuracy gain or loss as pooling parameters are tweaked. This was measured using a simple CNN setup run on both the cifar-10 and mnist image datasets.

This project was originally private, so the rest of the contents of the readme were originally used for the members of the team to share various discoveries, ideas, and useful links as we worked through the project.


Pipfiles included have tensorflow, sk-learn, and a few other libraries already installed.
If you don't know what pipenv is, I highly recommend you go learn about it, then install pipenv
for Python3. It automatically manages Python dependencies, and means that if we all work on 
the project in the pipenv defined, then everyone will be using the same version of all the
libraries, plus every time someone adds a new one through pipenv, it will get added to the file.
Then everyone gets it the next time they pull and activate pipenv, so there's no versioning
headaches.

I found a nice tutorial for setting up a CNN to work on the MNIST-Fashion dataset, it might
be a good starting point for us. Of course, we can also use whatever dataset we want, Dr.
Hamm recommended Cifar.
https://www.datacamp.com/community/tutorials/cnn-tensorflow-python/

Baseline is currently taken from keras.io cifar10 example, and the dataset is cifar10. Details
are below.

Jacob - 3/28/2019
I ran the model provided by keras.io (found here: https://keras.io/examples/cifar10_cnn/).
I had to adjust the steps_per_epoch parameter, it seems that's now a required parameter even
though the documentation says otherwise. I set it to be 20 steps_per_epoch because I can't
use GPU acceleration on my machine, and it was taking too long. Generally, steps_per_epoch
should actually be ceil(num_examples / batch_size), because each epoch is supposed to go
over every example in the dataset. The model and the results of the model are saved in 
Models/saved_models/keras_cifar10_trained_model.h5
Next, I made a "simplified" model where I pulled out practically every layer, and replaced
the rmsprop optimizer with sgd, and ran that model. It can be our baseline if we want, or we
could make a slightly better model the baseline. We'll have to use one for max and one for
average pooling, or something. The results for this one is also in 
Models/saved_models/basic_baseline_model.h5.

In case it's helpful, the saved model can be reloaded using load_model, which can be
seen at https://stackoverflow.com/questions/42763094/how-to-save-final-model-using-keras. This
way, we can collect results for actual tabulation later by loading our already trained models
and running them on the test sets, and then visualizing the data however we want. Make sure
that we try to save them all to the same place consistently (Models/saved_models) and name them
all something different so we don't overwrite someone's data.

I wrote a helper script for running a saved model on testing data. If you want to see how 
one of the models I specified above ran, just run evaluatedKerasMode.py -h to see the help
text on how to use it.

I found a resource called Google Collaboratory that offers 400GB of space in the cloud and 
access to CPU, GPU, and TPU Tensorflow for FREE in Google Cloud. You can just upload Python
snippets and run them, then download the results, it's literally just a Google cloud file
system. So if you can't use GPU on your machine, or it's still kind of slow, running
python3 simplifedBaseline.py took ~1.5 hours on my local machine, but 15 minutes in 
Collaboratory, just as a benchmark. You can run simplifiedBaseline yourself and check
if Collaboratory would be better for you too.
https://colab.research.google.com
Just sign in with your google account, add a code cell, upload the Python file you want
to run and copy-paste it verbatim into the cell, then go to runtime->runtime type->GPU, and
finally run the cell. It's a lot faster than anything my computer can achieve.
