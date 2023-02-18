import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# load the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape and normalize the input data
x_train = x_train.reshape(60000, 784)

print(mnist.load_data())