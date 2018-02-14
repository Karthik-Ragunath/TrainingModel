from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

'''
first layer has 8 inputs and 12 neurons with activation type as relu
second layer has 8 neurons
third layer is sigmod layer with 1 neuron so that value is maintained between 0 and 1
'''
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

'''
binary_crossentropy deals with logarithmic loss in Keras
optimizer adam refers to gradient descent algorithm
'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

'''
150 rows are used for training data
after every 10 training data, weights are updated
'''
model.fit(X, Y, epochs=150, batch_size=10)

'''
We are evaluating with same input and output as used in training data.
'''
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
