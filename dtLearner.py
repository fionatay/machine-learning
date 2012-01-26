"""Learn to estimate functions  from examples. (Chapters 18-20)"""
import sys, copy
sys.path.append('source/')
execfile("dataset.py")

from utils import *
import agents, random, operator

#______________________________________________________________________________

class Learner:
    """A Learner, or Learning Algorithm, can be trained with a dataset,
    and then asked to predict the target attribute of an example."""

    def train(self, dataset): 
        self.dataset = dataset

    def predict(self, example): 
        abstract

#______________________________________________________________________________

class DecisionTree:
    """A DecisionTree holds an attribute that is being tested, and a
    dict of {attrval: Tree} entries.  If Tree here is not a DecisionTree
    then it is the final classification of the example."""

    def __init__(self, attr, attrname=None, branches=None):
        "Initialize by saying what attribute this node tests."
        update(self, attr=attr, attrname=attrname or attr,
               branches=branches or {})

    def predict(self, example):
        "Given an example, use the tree to classify the example."
        child = self.branches[example[self.attr]]
        if isinstance(child, DecisionTree):
            return child.predict(example)
        else:
            return child

    def add(self, val, subtree):
        "Add a branch.  If self.attr = val, go to the given subtree."
        self.branches[val] = subtree
        return self

    def display(self, indent=0):
        name = self.attrname
        print 'Test', name
        for (val, subtree) in self.branches.items():
            print ' '*4*indent, name, '=', val, '==>',
            if isinstance(subtree, DecisionTree):
                subtree.display(indent+1)
            else:
                print 'RESULT = ', subtree                

    def getMostCommonLeafVal(self):
        """Return the most commonly occuring leaf value in the decision
        tree (for use in pruning)."""
        return "something"

    def copy(self, exclude=""):
        """Return a copy of the decision tree, with the 'exclude' node and its subtrees
        replaced by their most common leaf value."""
        return self

    def __repr__(self):
        return 'DecisionTree(%r, %r, %r)' % (
            self.attr, self.attrname, self.branches)

Yes, No = True, False
        
#______________________________________________________________________________

class DecisionTreeLearner(Learner):

    def predict(self, example):
        if isinstance(self.dt, DecisionTree):
            return self.dt.predict(example)
        else:
            return self.dt

    def train(self, dataset):
        self.dataset = dataset
        self.attrnames = dataset.attrnames
        self.dt = self.decision_tree_learning(dataset.examples, dataset.inputs)

    def decision_tree_learning(self, examples, attrs, default=None):
        if len(examples) == 0:
            return default
        elif self.all_same_class(examples):
            return examples[0][self.dataset.target]
        elif  len(attrs) == 0:
            return self.majority_value(examples)
        else:
            best = self.choose_attribute(attrs, examples)
            tree = DecisionTree(best, self.attrnames[best])
            for (v, examples_i) in self.split_by(best, examples):
                subtree = self.decision_tree_learning(examples_i,
                  removeall(best, attrs), self.majority_value(examples))
                tree.add(v, subtree)
            return tree

    def choose_attribute(self, attrs, examples):
        "Choose the attribute with the highest information gain."
        return argmax(attrs, lambda a: self.information_gain(a, examples))

    def all_same_class(self, examples):
        "Are all these examples in the same target class?"
        target = self.dataset.target
        class0 = examples[0][target]
        for e in examples:
           if e[target] != class0: return False
        return True

    def majority_value(self, examples):
        """Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality.)"""
        g = self.dataset.target
        return argmax(self.dataset.values[g],
                      lambda v: self.count(g, v, examples))

    def count(self, attr, val, examples):
        """Given an attribute index attr, a particular value val, and a set of
        examples, count how many of those examples have the value val in attribute 
        number attr."""
        ct = 0
        for example in examples:
            if example[attr] == val:
                ct = ct + 1
        return ct
    
    def information_gain(self, attr, examples):
        """Given an attribute attr and set of examples (examples), return 
        the information gain for that attribute."""
        return 0.5
    
    def split_by(self, attr, examples=None):
        """Return a list of (val, examples) pairs for each val of attr, assuming
        we took that split."""
        if examples == None:
            examples = self.dataset.examples
        return [(v, [e for e in examples if e[attr] == v])
                for v in self.dataset.values[attr]]
    
def entropy(values):
    "Number of bits to represent the probability distribution in values."
    return 1

#______________________________________________________________________________

def test(learner, dataset, examples=None, verbose=0):
    """Return the proportion of the examples that are correctly predicted.
    Assumes the learner has already been trained."""
    #if we aren't explicitly passed in any examples, set 'examples' to 
    #  be the ones that are in the dataset
    if examples == None: examples = dataset.examples
    #if we aren't given any examples, then your accuracy is 0.0
    if len(examples) == 0: return 0.0
    #initialize our 'right' or 'correct' count
    right = 0.0
    for example in examples:
        #grab the target index from the dataset and get that val from the ex.
        desired = example[dataset.target]
        #use the learner to predict the output value
        output = learner.predict(dataset.sanitize(example))
        #if it was right
        if output == desired:
            #increment right
            right += 1
            #if we're being verbose, then print out the example info
            if verbose >= 2:
               print '   OK: got %s for %s' % (desired, example)
        #otherwise if it was wrong and we're being verbose, 
        #  then print out the example info
        elif verbose:
            print 'WRONG: got %s, expected %s for %s' % (
               output, desired, example)
    #return the portion of test examples for which our learner was 'right'
    return right / len(examples)

def train_and_test(learner, dataset, start, end):
    """Reserve dataset.examples[start:end] for test; train on the remainder.
    Return the proportion of examples correct on the test examples."""
    #in the examples variable, save the original examples in the dataset.
    #  we'll be altering the dataset to pull out testing examples and 
    #  will need to revert it in the end.
    examples = dataset.examples
    try:
        #dataset.examples are the ones that it will TRAIN with - 
        #  those falling before 'start' and after 'end'
        dataset.examples = examples[:start] + examples[end:]
        learner.dataset = dataset 
        learner.train(dataset)
        return test(learner, dataset, examples[start:end])
    finally:
        #at the end, be sure to revert the dataset to contain all of
        #   its original examples
        dataset.examples = examples

def train_prune_and_test(learner, dataset, start, end):
    """See assignment for how to complete this function."""
    return 1.0

def cross_validation(learner, dataset, k=10, trials=1):
    """Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; If trials>1, average over several shuffles."""
    return 1.0
    
def learningcurve(learner, dataset, trials=10, sizes=None):
    if sizes == None:
        sizes = range(2, len(dataset.examples)-10, 2)
    def score(learner, size):
        random.shuffle(dataset.examples)
        return train_and_test(learner, dataset, 0, size)
    return [(size, mean([score(learner, size) for t in range(trials)]))
            for size in sizes]
#______________________________________________________________________________
