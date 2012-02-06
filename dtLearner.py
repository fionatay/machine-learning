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
        original_entropy = entropy(self.split_by(self.dataset.target, examples))
        size = len(examples)
        
        split = self.split_by(attr, examples)
        weighted_entropies = 0
        for sub in split:
            sub_examples = sub[1]
            this_entropy = entropy(self.split_by(self.dataset.target, sub_examples))
            weighted_entropy = this_entropy * len(sub_examples)/size
            weighted_entropies += weighted_entropy
        return original_entropy - weighted_entropies
    
    def split_by(self, attr, examples=None):
        """Return a list of (val, examples) pairs for each val of attr, assuming
        we took that split."""
        if examples == None:
            examples = self.dataset.examples
        return [(v, [e for e in examples if e[attr] == v])
                for v in self.dataset.values[attr]]

    def prune(self, validation_examples):
        return
    
def entropy(values):
    "Number of bits to represent the probability distribution in values."
    sizes = [len(sub[1]) for sub in values]
    totalSize = sum(sizes) * 1.0
    proportions = [size/totalSize if totalSize > 0 else 0 for size in sizes]
    entropies = [prop * math.log(prop, 2) if prop > 0 else 0 for prop in proportions]
    return -sum(entropies)
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
    #In the examples variable, save the original examples in the dataset.
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

def train_prune_and_test(learner, dataset, start, end, ratio = 0.9):
    """See assignment for how to complete this function."""
    examples = dataset.examples
    try:
        all_training = examples[:start] + examples[end:]
        num_training = int(ratio * len(all_training))
        
        training = all_training[:num_training]
        dataset.examples = training
        learner.train(dataset)
        
        validation = all_training[num_training:]
        learner.prune(validation)

        print "Given " + str(len(all_training)) + " examples, training on " \
              + str(len(learner.dataset.examples)) + ", pruning on " + str(len(validation))
        return test(learner, dataset, examples[start:end])
    finally:
        dataset.examples = examples

def cross_validation(learner, dataset, k=10, trials=1):
    """Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; If trials>1, average over several shuffles."""
    if trials > 1:
        list_trials = [cross_validation(learner, dataset, k, 1)
                       for trial in range(trials)]
        return sum(list_trials)/len(list_trials)
    else:
        test_size = len(dataset.examples)/k
        train_and_test(learner, dataset, 0, test_size)
    
    
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

simpleData = DataSet(examples=[[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]], attrs=[[0, 1], [0, 1], [0,1]], target=0)

def testAll():
    print "===Tests starting==="
    print "Entropy - checking computed entropies" 
    whole_set = [(1, [[1, 1], [1,2]])]
    check(entropy(whole_set), 0)
    
    split_set = [(1, [[1, 0], [1, 1]]), (2, [[2, 0], [2, 1]])]
    check(entropy(split_set), 1)
    print

    print "Info gain - checking computed info gains"
    learner = DecisionTreeLearner()
    learner.dataset = simpleData
    check(learner.information_gain(1, learner.dataset.examples), 1)
    check(learner.information_gain(2, learner.dataset.examples), 0)
    print
    
    print "Learner - checking accuracy"
    iris1 = train_and_test(DecisionTreeLearner(), iris, 135, 150)
    check(iris1, 0.66666666666666663)
    orings1 = 0.7692307692307692
    check(orings1, train_and_test(DecisionTreeLearner(), orings, 10, 23))
    zoo1 = 0.71999999999999997
    check(zoo1, train_and_test(DecisionTreeLearner(), zoo, 75, 100))
    print

    print "Pruning - checking that no pruning gives same results"
    iris1 = train_and_test(DecisionTreeLearner(), iris, 135, 150)
    iris2 = train_prune_and_test(DecisionTreeLearner(), iris, 135, 150, 1)
    check(iris1, iris2)
    
    zoo1 = train_and_test(DecisionTreeLearner(), zoo, 75, 100)
    zoo2 = train_prune_and_test(DecisionTreeLearner(), zoo, 75, 100, 1)
    check(zoo1, zoo2)
    print

    print "Pruning - checking that pruning gives better results"
    iris1 = train_and_test(DecisionTreeLearner(), iris, 135, 150)
    iris2 = train_prune_and_test(DecisionTreeLearner(), iris, 135, 150, 0.9)
    better(iris1, iris2)

    zoo1 = train_and_test(DecisionTreeLearner(), zoo, 75, 100)
    zoo2 = train_prune_and_test(DecisionTreeLearner(), zoo, 75, 100, 0.9)
    better(zoo1, zoo2)
    print "===Tests finished==="

def better(original, better):
    if better > original: print "Test passed - improved from " + \
       str(original) + " to " + str(better)
    else: print "Test failed - deproved from " + \
          str(original) + " to " + str(better)

def check(result, expected):
    if result == expected: print "Test passed"
    else: print "Test failed. Expected " + str(expected) + ", got " + str(result)
