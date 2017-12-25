# KKBOX's music recommandation challenge
Here is my implementation of the KKBOX's music recommandation challenge from Kaggle. 
My aim was much more playing with DNN and find new ways to bypass my laptop limitations 
rather than having a good score. And it indeed yielded a poor score! I reached 62.6% accuracy
with training set, and 58.1% as the final score with hidden data. The good point is I have 
been able to run it on my laptop without CUDA support in about one hour. This is anyway very far
from the best result for this challenge (about 74.5%).

The dataset consist of 3 files (not provided here, too large). The first file describe every users
through few categorical variables, the second the songs with a little amount of features aswell,
and a third consisting of pairs from the 2 previous files, a couple more features, and a label.
The label is 1 for "the user did listen to this song again within the next 30 days", and 0 else.
The problem is most of those categorical features could take thousands of values, making the one-hot
encoding approach not manageable given my current ressources. I also could not find any good feature
engineering to get more reasonable numbers without destroying the information. I decided to make
a really naive approach.

My first idea was to use a kind of home-made "word-embedding" for my data, to write every songs
and users as vector of extracted features. But I could not make it work in a reasonable amount of time,
so I diecided to go for something as basic as possible (as my score actually!). My main idea was to 
compress the information to its smallest size, that's to say as a binary vector. Given the fact 
that some features from the songs dataset had over 300k classes, it sounded as a good idea at first. 
You could label-encode those class features as numbers, then writte them as binary vectors (each power 
of 2 being a new feature), drastically compressing the data while formally keeping all the information.

But how to read it now? I first started by writting an Auto-Encoder with an overcomplete representation,
that's to say with more units than features. I then tried to feed the results of both Auto-Encoders (one for
users, one for songs) to a Deep Neural Network (because I simply wanted to do one, remember?). I finally kicked
the Auto-Encders out, because I could easily encode the songs/users (above 99,9% accuracy) but I realized it 
brought nothing explicit enough to the DNN. I then designed a simple DNN, and fed it with features built from 
the corresponding features in songs/users datasets for each pairs in the train set.

## Architecture
I performed lot of testings but there are many left. I finally and ended up with a 4 hidden layers 
architecture:
- hidden layer 1: 500 linear units
- hidden layer 2: 200 tanh units
- hidden layer 3: 200 linear units
- hidden layer 4: 200 tanh units

### Activation functions
I used a tanh activation on even layers to feed the following layer with normalized inputs. The idea was
to treat each pair of layers as a feature extractor, the first layer giving me a particular pattern, the 
second squashing it in a (-1;1) range. It gave me slightly better results than using sigmoid 
activation functions on the same layers, but nothing earth-shattering. Overall, using different activation 
functions (and different combinations also) did not change drastically the results, which is not really 
surprising with a step back. Whatever the output of an activation function is, if a logit A > logit B then 
activation(A) > activation(B) unless using some non-monotonic activation functions (typically a ReLU which 
I did not test yet ironically).

### Dimensionality of the NN
I tried many different architecture (wider and/or less deep mostly) but had even porrer results. What I overall noticed is
that the number of total parameters is not as improtant as the architecture itself. Having a wider neural network with 
one less hidden layer will result in a much higher number of parameters, and a much higher computation time. 
Still, it scores even more badly than the previous one. Alex Karpathy mentionned in his CNN lessons that usually,
deeper is better than wider. In this case, it seems also to be true for a DNN.

### Batchsize
Yes I tried differnet batchsize also! The higher the batchsize, the faster the code runs, obvious. But I also noticed
that the results improves at some point with batchsize, up to around 10. I think it is if you treat each pair from the 
train set one by one, weights are updated on a too small scope of the data, thus pushing them in a very particular direction.
Using bigger batchsize will update weights to an overall direction which corresponds better to the most proeminent trends 
in the data (sacrificing particular cases, which we don't really care given our problem and the final result we had anyway).
It is also worth to mention that if using too large batches (namely around 1000 thousand training examples), the loss fails to converge.

### Learning parameters
I used a learning rate of 0.01, and a decay rate of 0.9/(1+epoch). The decay rate is applied only if the total loss from the
epoch i is not below 99% of the loss from epoch i-1. I ran it initially on 10 epochs but the loss usually converge at second 
or third. I initialized weights and biases with a normal distribution of mean 0 and standard deviation of 1. I wanted each
neuron to be quite different from step one to encourage 'free thought'. With a narrow standard deviation, the loss often
failed to converge to an acceptable value.

### Loss
I used the classic softmax cross-entropy function and also the squared difference between sigmoid(logits) and labels. Both gave
me very consistent results. I should say 'consistently bad results'. An interesting thing to mention is that the loss converge to
about 0.66 if using the cross-entropy loss function, and about 0.23 if using the other. In both case, since we are doing a 2-class
classification, it means that the sigmoid function outputs something in the 0.52 to 0.55 range for positive class, and in 0.48 to 0.45
range for negative class in average. It could mean either the DNN hardly can separate the two classes, or reproduce very nicely some
data and massively fails for others. I should extract more information from the loss later, by outputting it in a file and analyzing
it.

### About the result
In this precise case, the DNN fails to read combinations of features unfortunately. It needs some guidance to go
to the right direction, either by introducing more explicit features, either by training it differently (pre-training
some parts alone typically). The problem is we try to make it read too precise combinations instead of asking him to 
extract hidden features. I still think it is possible, but much more work is required to have just a decent result here.
There are much more efficient models to deal with this kind of problems anyway (Factorization Machines for example), but the 
experience was very instructive!


## Requirements
I created this script using:
- Python 3.6.2
- Scikit-Learn 0.19.0
- Pandas 0.20.3
- TensorFLow 1.2.1


## Usage
Execute the main script, `kkbox.py` to run what you want after setting some options. There are 3 main places
to look at in order to tweak the model/training:
- `kklib\models.py`: change the architecture of the neural networks here.
- `kklib\settings.py`: change the options for the execution. You can run:
    - the preprocessing or not (if the files are already saved)
    - the binary encoding or not (if files aren't already saved) 
    - the matching files builder to find position of elements of each pairs in
the songs and users datasets (beware, it is really long so do it once)
    - with or without an Auto-Encoder step
    - wether you should train the Auto-Encoder and/or the DNN or read weights from disk
- `kkbox.py`: change the training parameters when calling the `NN_model` function (initial learning rate, batchsize, etc...)


## The files
- `kkbox.py`: the main script.
- `preproc.py`: the script containing the functions to preprocess the data from first stage (cleaning) to last step (binary encoding).
- `settings.py`: the script containing the main options (such as perform the trianing or load weights, write intermediate files to disk or read them etc...)
This script also caintains the global variables declaration.
- `utils.py`: a script containing various utilities functions, such as the binary encoding algorithm, the batch builder etc...
- `models.py`: script containing the models used. All the tensorflow parts of the code can be found therein.



