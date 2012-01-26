#
# Name: Sara Owsley Sood
# 
# Description: This file contains a basic class for storing data sets, called DataSet.  
#            It also contains many example uses of this class.
#
#
#
#
#

import sys
sys.path.append('source/')

from utils import *
import agents, random, operator

#______________________________________________________________________________

class DataSet:
    """A data set for a machine learning problem.  It has the following fields:

    d.examples    A list of examples (including both inputs and outputs).  
                  Each one is a list of attribute values.
    d.attrs       A list of integers to index into an example, so example[attr]
                  gives a value. Normally the same as range(len(d.examples)). 
    d.attrnames   Optional list of mnemonic names for corresponding attrs 
                  (including both inputs and outputs).
    d.target      The index of the attribute that a learning algorithm will try 
                  to predict. By default the final attribute.
    d.targetname  The name of the attribute that the learning algorithm will try
                  to predict.
    d.inputs      The list of attrs (indices, not names) without the target (in 
                  other words, a list of the indices of input attributes).
    d.values      A list of lists, each sublist is the set of possible
                  values for the corresponding attribute (including both inputs
                  and outputs). If None, it is computed from the known examples 
                  by self.setproblem. If not None, an erroneous value raises 
                  ValueError.
    d.name        Name of the data set (for output display only).
    d.source      URL or other source where the data came from.

    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs."""

    def __init__(self, examples=None, attrs=None, target=-1, values=None,
                 attrnames=None, name='', source='',
                 inputs=None, exclude=(), doc=''):
        """Accepts any of DataSet's fields.  Examples can also be a string 
        or file from which to parse examples using parse_csv.
        >>> DataSet(examples='1, 2, 3')
        <DataSet(): 1 examples, 3 attributes>
        """
        #grab the name, source and values args and store them in the 
        #   attributes of our dataset class.  If you've not seen the 
        #   'update' function before, in python, type 'help(update)'
        update(self, name=name, source=source, values=values)

        # Initialize .examples from string or list or data directory
        if isinstance(examples, str):
            #if the examples passed in are an actual string, then just
            #   parse them.
            self.examples = parse_csv(examples)
        elif examples is None:
            #if no examples are passed in, assume that the examples are 
            #  stored in a file named 'name'.csv.
            self.examples = parse_csv(DataFile(name+'.csv').read())
        else:
            #otherwise, assume that the examples sent in were in the 
            #  appropriate list format
            self.examples = examples
        
        #check to make sure that the examples have been read in properly,
        #  mostly just that their values are in the right range for each
        #  attribute
        map(self.check_example, self.examples)
        
        # Attrs are the indicies of examples, unless otherwise stated.
        if not attrs and self.examples:
            attrs = range(len(self.examples[0]))
        self.attrs = attrs
        
        # Initialize .attrnames from string, list, or by default
        if isinstance(attrnames, str): 
            #if the input was a string, just split it and assume the result
            #  are the appropriate attribute names.
            self.attrnames = attrnames.split()
        else:
            #otherwise, if attrnames has a value, maybe it is a list already,
            #  so use it, but if not, default to simply the indices.
            self.attrnames = attrnames or attrs
        
        
        self.setproblem(target, inputs=inputs, exclude=exclude)
        
        #grab the output attribute name, just for printing
        self.targetname = self.attrnames[self.target]

    def __str__(self):
        """Returns a string representation of the DataSet object."""
        s = "The " + str(self.name) + " dataset contains the following " + str(len(self.inputs)) + " input attributes (followed by their possible values):\n" 
        for input in self.inputs:
            s += str(self.attrnames[input]) + "\t" + str(self.values[input]) + "\n"
        s += "The output to be predicted is \"" + str(self.attrnames[self.target]) + "\" with possible values:\n" 
        s += str(self.values[self.target]) + ".\n"    
        s += "The dataset contains " + str(len(self.examples)) + " training examples.  Here is one example:\n" 
        s += str(self.examples[0])
        return s

    def setproblem(self, target, inputs=None, exclude=()):
        """Set (or change) the target and/or inputs.
        This way, one DataSet can be used multiple ways. inputs, if specified,
        is a list of attributes, or specify exclude as a list of attributes
        to not put use in inputs. Attributes can be -n .. n, or an attrname.
        Also computes the list of possible values, if that wasn't done yet."""
        self.target = self.attrnum(target)
        exclude = map(self.attrnum, exclude)
        if inputs:
            self.inputs = removall(self.target, inputs)
        else:
            self.inputs = [a for a in self.attrs
                           if a is not self.target and a not in exclude]
        if not self.values:
            self.values = map(unique, zip(*self.examples))

    def add_example(self, example):
        """Add an example to the list of examples, checking it first."""
        self.check_example(example)
        self.examples.append(example)

    def check_example(self, example):
        """Raise ValueError if example has any invalid values."""
        if self.values:
            for a in self.attrs:
                if example[a] not in self.values[a]:
                    raise ValueError('Bad value %s for attribute %s in %s' %
                                     (example[a], self.attrnames[a], example))

    def attrnum(self, attr):
        "Returns the number used for attr, which can be a name, or -n .. n."
        if attr < 0:
            return len(self.attrs) + attr
        elif isinstance(attr, str): 
            return self.attrnames.index(attr)
        else:
            return attr

    def sanitize(self, example):
       "Return a copy of example, with non-input attributes replaced by 0."
       return [i in self.inputs and example[i] for i in range(len(example))] 

    def __repr__(self):
        return '<DataSet(%s): %d examples, %d attributes>' % (
            self.name, len(self.examples), len(self.attrs))

#______________________________________________________________________________

def parse_csv(input, delim=','):
    """Input is a string consisting of lines, each line has comma-delimited 
    fields.  Convert this into a list of lists.  Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but '\t' and None are also reasonable values.
    >>> parse_csv('1, 2, 3 \n 0, 2, na')
    [[1, 2, 3], [0, 2, 'na']]
    """
    #Here is the code from AIMA - below it is a more verbose version....
    #lines = [line for line in input.splitlines() if line.strip() is not '']
    #return [map(num_or_str, line.split(delim)) for line in lines]

    #separate the input into a list of rawlines (only non-empty ones)
    rawlines = []
    for line in input.splitlines():
        if not line.strip() == '':
            rawlines.append(line.strip())
 
    #split each line into a list of cells and turn cells into nums where appropriate
    lines = []
    for line in rawlines:
        cells = line.split(delim)
        cells = map(num_or_str, cells)
        lines.append(cells)
        
    return lines

#______________________________________________________________________________
# The rest of this file gives Data sets for machine learning problems.

orings = DataSet(name='orings', target='Distressed',
                 attrnames="Rings Distressed Temp Pressure Flightnum")


zoo = DataSet(name='zoo', target='type', exclude=['name'],
              attrnames="name hair feathers eggs milk airborne aquatic " +
              "predator toothed backbone breathes venomous fins legs tail " +
              "domestic catsize type") 

#three more data sets from the UCI Machine Learning Repository 
iris = DataSet(name="iris", target="class",
               attrnames="sepal-len sepal-width petal-len petal-width class")

wine = DataSet(name="wine", target="class",
               attrnames="class Alcohol MalicAcid Ash AlcalinityOfAsh Magnesium " +
               "TotalPhenols Flavanoids NonflavanoidPhenols Proanthocyanins " +
               "ColorIntensity Hue OD280|OD315OfDilutedWines Proline") 

mushroom = DataSet(name="mushroom", target="class", #class means edible or poisonous
                   attrnames="class cap-shape cap-surface cap-color bruises odor " +
                   "gill-attachment gill-spacing gill-size gill-color stalk-shape " +
                   "stalk-root stalk-surface-above-ring stalk-surface-below-ring " +
                   "stalk-color-above-ring stalk-color-below-ring veil-type veil-color " +
                   "ring-number ring-type spore-print-color population habitat")

#______________________________________________________________________________
# The Restaurant example from Fig. 18.2 (restaurant.csv)

def RestaurantDS(examples=None):
    "Build a DataSet of Restaurant waiting examples."
    return DataSet(name='restaurant', target='Wait', examples=examples,
                  attrnames='Alternate Bar Fri/Sat Hungry Patrons Price '
                   + 'Raining Reservation Type WaitEstimate Wait')

restaurant = RestaurantDS()

#______________________________________________________________________________
# Artificial, generated  examples.

def Majority(k, n):
    """Return a DataSet with n k-bit examples of the majority problem:
    k random bits followed by a 1 if more than half the bits are 1, else 0."""
    examples = []
    for i in range(n):
        bits = [random.choice([0, 1]) for i in range(k)]
        bits.append(sum(bits) > k/2)
        examples.append(bits)
    return DataSet(name="majority", examples=examples)

MajorityDataSet = Majority(3, 16)

def Parity(k, n, name="parity"):
    """Return a DataSet with n k-bit examples of the parity problem:
    k random bits followed by a 1 if an odd number of bits are 1, else 0."""
    examples = []
    for i in range(n):
        bits = [random.choice([0, 1]) for i in range(k)]
        bits.append(sum(bits) % 2)
        examples.append(bits)
    return DataSet(name=name, examples=examples)

ParityDataSet = Parity(3, 16)

def Xor(n):
    """Return a DataSet with n examples of 2-input xor."""
    return Parity(2, n, name="xor")

XorDataSet = Xor(16)

def ContinuousXor(n):
    "2 inputs are chosen uniformly form (0.0 .. 2.0]; output is xor of ints."
    examples = []
    for i in range(n):
        x, y = [random.uniform(0.0, 2.0) for i in '12']
        examples.append([x, y, int(x) != int(y)])
    return DataSet(name="continuous xor", examples=examples)

ContinuousXorDataSet = ContinuousXor(16)


#______________________________________________________________________________

    
