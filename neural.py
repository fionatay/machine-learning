import math, random


#
#  Arrays are represented by lists of (equal-sized) lists. There are
#  packages, like numpy, that will speed up the indexing.
#
def makeZeroArray(m, n):
    """Create an m x n array filled with zero values."""
    result = []
    for i in range(m):
	result.append([0.0] * n)
    return result

def makeRandomArray(m, n, lower=-2.0, upper=2.0):
    """Create an m x n array filled with random values."""
    result = []
    for i in range(m):
	result.append(map(lambda j: random.uniform(lower, upper), [0] * n))
    return result


#
#  The activation function and its derivative. We provide the sigmoid function
#  and the tanh function, a commonly-used alternative.
#
def sigmoid(x):
    """Return 1/(1+e^-x)."""
    return 1.0 / (1.0 + math.exp(-x))

def dsigmoid(y):
    """Return the derivative of sigmoid, based on the value of the function."""
    return y * (1.0 - y)

def tanh(x):
    """Return an alternative to sigmoid."""
    return math.tanh(x)

def dtanh(y):
    """Return the derivative of the alternative to sigmoid,
       based on the value of the function."""
    return 1.0 - y * y


#
#  The neural network itself. It contains three lists (the activation values
#  for each of the layers) and four arrays (two for the weights and two more
#  for the most recent changes--for momentum).
#
class neuralNet:
    class sizeMismatch(Exception):
	"""Exception raised when the wrong number of input values
	   is offered."""
	def __init__(self, desired, actual):
	    self.desired = desired
	    self.actual = actual
	def __str__(self):
	    return "Incorrect number of inputs: " + str(self.desired) + \
                   " required, " + str(self.actual) + " received."


    def __init__(self, nInput, nHidden, nOutput):
	self.numInput = nInput + 1   # one extra for the bias node
	self.numHidden = nHidden + 1 # one extra for the bias node
	self.numOutput = nOutput
	self.inputLayer = [1.0] * self.numInput
	self.hiddenLayer = [1.0] * self.numHidden
	self.outputLayer = [1.0] * self.numOutput
	self.ihWeights = makeRandomArray(self.numInput, self.numHidden - 1)
	self.hoWeights = makeRandomArray(self.numHidden, self.numOutput)
	self.ihChanges = makeZeroArray(self.numInput, self.numHidden - 1)
	self.hoChanges = makeZeroArray(self.numHidden, self.numOutput)
	self.actFunction = sigmoid
	self.dactFunction = dsigmoid


    def evaluate(self, inputs):
	"""Carries out forward propagation on the neural net."""
	if len(inputs) != self.numInput - 1:
	    raise self.sizeMismatch(self.numInput - 1, len(inputs))

	# Create the input layer.
	self.inputLayer = inputs + [1.0]

	# Evaluate the hidden layer.
	for h in xrange(self.numHidden - 1):
	    accum = 0.0
	    for i in xrange(self.numInput):
		accum += self.ihWeights[i][h] * self.inputLayer[i]
	    self.hiddenLayer[h] = self.actFunction(accum)

	# Evaluate the output layer.
	for o in xrange(self.numOutput):
	    accum = 0.0
	    for h in xrange(self.numHidden):
		accum += self.hoWeights[h][o] * self.hiddenLayer[h]
	    self.outputLayer[o] = self.actFunction(accum)

	# Return a *copy* of the output layer.
	return self.outputLayer[:]


    def test(self, data):
	"""Tests the neural net on a list of values.
           Requires a list of input-output pairs.
           Returns a list of triples:
           (input, desired-output, actual-output)."""
	return map(lambda (x,y): (x,y,self.evaluate(x)), data)


    def train(self, data,
              learningRate=0.5, momentumFactor=0.1,
              iterations=1000, printInterval=100):
	"""Carries out a training cycle on the neural net.
           The training data must be a list of input-output pairs."""
        if printInterval <= 0:
            printCount = 0
            leftOver = iterations
        else:
            printCount = iterations / printInterval
            leftOver   = iterations % printInterval

	def onePass():
	    for (x,y) in data:
		self.backPropagate(x, y, learningRate, momentumFactor)

	def onePassWithError():
	    error = 0.0
	    for (x,y) in data:
		error += self.backPropagate(x, y, learningRate, momentumFactor)
	    return error

        for i in xrange(printCount):
            for j in range(printInterval-1):
                onePass()
            print "error %-14f" % onePassWithError()
        for i in xrange(leftOver):
            onePass()


    def backPropagate(self,
		      inputs, desiredResult,
		      learningRate, momentumFactor):
        """The basic back propagation algorithm for adjusting weights."""

	# Carry out the forward pass.
	outputs = self.evaluate(inputs)

	# Compute the deltas at the output layer.
	outputDeltas = [0.0] * self.numOutput
	for o in xrange(self.numOutput):
	    outputDeltas[o] = self.dactFunction(outputs[o]) * \
		              (desiredResult[o] - outputs[o])

	# Compute the deltas at the hidden layer.
	hiddenDeltas = [0.0] * self.numHidden
	for h in xrange(self.numHidden - 1):
	    error = 0.0
	    for o in xrange(self.numOutput):
		error += outputDeltas[o] * self.hoWeights[h][o]
	    hiddenDeltas[h] = self.dactFunction(self.hiddenLayer[h]) * error

	# Update the weights and changes for the hidden-output layers.
	for h in xrange(self.numHidden):
	    for o in xrange(self.numOutput):
		change = outputDeltas[o] * self.hiddenLayer[h]
		self.hoWeights[h][o] += learningRate * change + \
 		                        momentumFactor * self.hoChanges[h][o]
		self.hoChanges[h][o] = change

	# Update the weights and changes for the input-hidden layers.
	for i in xrange(self.numInput):
	    for h in xrange(self.numHidden - 1):
		change = hiddenDeltas[h] * self.inputLayer[i]
		self.ihWeights[i][h] += learningRate * change + \
		                        momentumFactor * self.ihChanges[i][h]
		self.ihChanges[i][h] = change

	# Compute the (half the sum of squares of the) errors.
	squareErrors = 0.0
	for o in xrange(self.numOutput):
	    squareErrors += (desiredResult[o] - self.outputLayer[o])**2
	return 0.5 * squareErrors


    def getIHWeights(self):
	"""Returns the input-hidden weights as a list of lists."""
	return self.ihWeights


    def getHOWeights(self):
	"""Returns the hidden-output weights as a list of lists."""
	return self.hoWeights


    def useTanh():
	"""Changes the activation function from the sigmoid function
           to the hyperbolic tangent."""
        self.actFunction = tanh
	self.dactFunction = dtanh





###################################################################
### an example using the above Neural Net class to classify xor ###

xorTrainingData = [([0.0, 0.0], [0.0]),
                   ([0.0, 1.0], [1.0]),
                   ([1.0, 0.0], [1.0]),
                   ([1.0, 1.0], [0.0])]

#set up a neural net with two input nodes, 5 hidden and 1 output
xorNN = neuralNet(2,5,1)

#train the neural net on a set of data
xorNN.train(xorTrainingData)
#along the way, this will print the error during each epoch

#test the system
print "testing our xor neural network: " 
print xorNN.test(xorTrainingData)
#notice that this prints each example and the actual output from the NN

### an example using the Neural Net class on a simple voter preference
###    task (ouput is 0.0 for democrat, 1.0 for republican; inputs are
###    values on various issues like social security, budget, etc.)

voterPreferenceTrainingData = [([0.9,0.6,0.8,0.3,0.1], [1.0]),
                               ([0.8,0.8,0.4,0.6,0.4], [1.0]),
                               ([0.7,0.2,0.4,0.6,0.3], [1.0]),
                               ([0.5,0.5,0.8,0.4,0.8], [0.0]),
                               ([0.3,0.1,0.6,0.8,0.8], [0.0]),
                               ([0.6,0.3,0.4,0.3,0.6], [0.0])]

voterPrefNN = neuralNet(5,10,1)
voterPrefNN.train(voterPreferenceTrainingData)
print "testing our voter preference neural net: "
print voterPrefNN.test(voterPreferenceTrainingData)
                               
