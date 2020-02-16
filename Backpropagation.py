import numpy as np

# Safe Incinerator System
# We can, however, extend the functionality of the system by adding to it logic circuitry designed to detect
# if any one of the sensors does not agree with the other two. If all three sensors are operating properly,
# they should detect flame with equal accuracy. Thus, they should either all register “low” (000: no flame)
# or all register “high” (111: good flame). Any other output combination (001, 010, 011, 100, 101, or 110)
# constitutes a disagreement between sensors, and may therefore serve as an indicator of a potential sensor
# failure. If we added circuitry to detect any one of the six “sensor disagreement” conditions, we could use
# the output of that circuitry to activate an alarm. Whoever is monitoring the incinerator would then exercise
# judgment in either continuing to operate with a possible failed sensor (inputs: 011, 101, or 110), or shut
# the incinerator down to be absolutely safe. Also, if the incinerator is shut down (no flame), and one or
# more of the sensors still indicates flame (001, 010, 011, 100, 101, or 110) while the other(s) indicate(s)
# no flame, it will be known that a definite sensor problem exists. The first step in designing this “sensor
# disagreement” detection circuit is to write a truth table describing its behavior. Since we already have a
# truth table describing the output of the “good flame” logic circuit, we can simply add another output column
# to the table to represent the second circuit, and make a table representing the entire logic system:


# Reference: https://www.allaboutcircuits.com/textbook/digital/chpt-7/converting-truth-tables-boolean-expressions/
#            Simon Haykin
#            geeksnome/machine-learning-made-easy


# create a neural network consisting of input layer (3 nodes), hidden layer (2 nodes) and an output layer (2 nodes)
# and use backpropagation to train the model.

# create the input array 'x' which has 3 inputs per row x1, x2, x3
# and output array 'y' which has 2 outputs per row y1, y2
# as per the problem described above.

x = np.array(([0,0,0],[1,1,1],[0,0,1],[0,1,0],[1,1,0],[0,1,1],[1,0,0],[1,0,1]), dtype = int)
y = np.array(([0,0],[1,0],[0,1],[0,1],[1,1],[1,1],[0,1],[1,1]), dtype = int)

# creating the weight array (wi) of the transition from input to hidden layer {[w11,21],[w12,w22],[w13,w23]}

wi = np.array(([0.1,0.2],[0.2,0.3],[0.3,0.4]), dtype = float)

# similarly creating the weight array (wj) for the transition from hiden to output layer

wj = np.array(([0.2,0.1],[0.3,0.2]), dtype = float)

# bias for the hidden layer and output layers are b0 = [b01,b02] and b1 = [b11,b12] respectively

b0 = np.array(([0,1]), dtype = float)
b1 = np.array(([1,0]), dtype = float) #these values taken after hit and trial

# we assume the learning rate to be 0.5

n = 0.5

# input of the hidden layer is given as hin = x1*w11 + x2*w12 + x3*w13 + b01
# output of the hidden layer is given as hout = f(hin) = 1/(1 + exp^(-hin)), where f(x) = 1/(1 + exp^(-x)) {sigmoid func.}

# in the similar fashion we calculate yin and yout, input and output of the output layer

# we will be calculating the hout for each of the values in the training set x and adjust the weights for
# each set of inputs

# the error e is calculated as e = (target - actual)  and local gradient, lg = e * f'(yin) for the output layer
# dw11 = n*lg*hout(1); w11 = w11 + dw11

# for hidden layer since we have two output nodes, we will have 2 errors e1 and e2 corresponding to each yout
# lg = f'(hin[1])*(e1*f'(yin[1])*w'11 + e2*f'(yin[2])*w'21)
# dw11 = n*lg*x1; w11 = w11 + dw11

print("Input set: \n", x)
print("Target output: \n", y)
print("Initial weights set for input->hidden: \n", wi)
print("Initial weights set for hidden->output: \n", wj)
print("Bias of hidden layer:\n", b0)
print("Bias of output layer:\n", b1)
for j in range(0,1000):    # no. of epoch

    # forward pass

    hin = np.dot(x, wi) + b0
    hout = 1/(1 + np.exp(-hin))
    yin = np.dot(hout, wj) + b1
    yout = 1/(1 + np.exp(-yin))

    # backward pass

    eout = y - yout #error
    deri_sigmod_yout = (1 - 1/(1 + np.exp(-yout))) * 1/(1 + np.exp(-yout)) #derivative of activation func.
    lgo = eout * deri_sigmod_yout #local gradient or delta for output layer
    # update the weights for transition from hidden layer to output layer
    dwj = n * hout.T.dot(lgo)
    db1 = n * b1 * np.mean(lgo)
    wj = np.add(wj,dwj)
    b1 = np.add(b1,db1)
    # calculate the local gradient for each of the hidden layer nodes
    ehid = lgo.dot(wj.T)
    deri_sigmoid_hout = (1 - 1/(1 + np.exp(-hout))) * 1/(1 + np.exp(-hout))
    lgh = ehid * deri_sigmoid_hout #local gradient or delta for hidden layer
    # update the weights for transition from input to hidden layer
    dwi = n * x.T.dot(lgh)
    db0 = n * b0 * np.mean(lgh)
    wi = np.add(wi,dwi)
    b0 = np.add(b0,db0)
    if(j%10 == 0):
        print("Weights set for input->hidden: \n", wi)
        print("Bais for input->hidden:\n", b0)
        print("Weights set for hidden->output: \n", wj)
        print("Bais for hidden->output:\n", b1)
        print("Predicted output: \n", yout)
        print("Predicted output after rounding off: \n", np.round(yout,0))
