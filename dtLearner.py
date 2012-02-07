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

    def getMostCommonLeafVal(self, values, target):
        """Return the most commonly occuring leaf value in the decision
        tree (for use in pruning).
        Input is of the form List<Tuple<val, List<examples>>"""
        targets = {}
        for val in values:
            if val[target] in targets: targets[val[target]] += 1
            else: targets[val[target]] = 1
        return argmax(targets.items(), lambda pair: pair[1])[0]

    def uniquify(self):
        return self.rename([])
    
    def rename(self, used_names):
        """Assign a unique name to each node in the tree and return a list of node
        names"""
        for val, subtree in self.branches.items():
            if isinstance(subtree, DecisionTree):
                subtree.rename(used_names)

        self.attrname = "Node" + str(len(used_names))
        used_names.append(self.attrname)
        return used_names
        
    def copy(self, exclude, values, target):
        """Return a copy of the decision tree, with the 'exclude' node and its subtrees
        replaced by their most common leaf value."""
        if self.attrname == exclude:
            return self.getMostCommonLeafVal(values, target)
        else:
            new_branches = {}
            for val, subtree in self.branches.items():
                if isinstance(subtree, DecisionTree):
                    sub_values = filter_by(self.attr, val, values)
                    new_branches[val] = subtree.copy(exclude, sub_values, target)
                else:
                    new_branches[val] = subtree
            return DecisionTree(self.attr, self.attrname, new_branches)

    def __repr__(self):
        return 'DecisionTree(%r, %r, %r)' % (
            self.attr, self.attrname, self.branches)

def filter_by(attr_number, attr_value, values):
    def has_attr_value(example):
        return example[attr_number] == attr_value
    return filter(has_attr_value, values)

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
        new_learner = DecisionTreeLearner() # Temporary learner to store/test new dt
        new_learner.dataset = self.dataset
        new_learner.dt = None
        
        while(isinstance(self.dt, DecisionTree)):
            original_accuracy = test(self, self.dataset, validation_examples)

            best_node = None
            best_tree = None
            best_acc = 0
            node_names = self.dt.uniquify()[:-1] # Try to delete every node, except the root
            for removal in node_names:
                new_tree = self.dt.copy(removal, self.dataset.examples, self.dataset.target)
                new_learner.dt = new_tree
                acc = test(new_learner, self.dataset, validation_examples)
                if acc > best_acc:
                    best_node = removal
                    best_tree = new_tree
                    best_acc = acc

            print 'Accuracy new <- old ', best_acc, original_accuracy, 'from', best_node
            if best_acc > original_accuracy:
                print 'Deleting ', best_node, 'beneficial'
                self.dt = best_tree
            else:
                print 'Deleting ', best_node, 'not beneficial'
            print 'Old tree: ', self.dt
            print 'New tree: ', new_learner.dt
            print
            break            
        return

def entropy(values):
    """Takes input of List<Tuple(Val, Examples)>. Computes the entropy associated with the split
    among each Val."""
    sizes = [len(sub[1]) for sub in values]
    totalSize = sum(sizes)
    proportions = [1.0*size/totalSize if totalSize > 0 else 0 for size in sizes]
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

def train_prune_and_test(learner, dataset, start, end, ratio = 0.66):
    """Train and prune the tree using the given examples. Divides between training/validation
    according to given ratio. Tests on remaining examples."""
    examples = dataset.examples
    try:
        all_training = examples[:start] + examples[end:]
        num_training = int(ratio * len(all_training))

        training = all_training[:num_training]
        dataset.examples = training
        learner.train(dataset)

        validation = all_training[num_training:]
        print 'Total training', len(all_training), 'Initial training', len(training), 'Pruning', len(validation), 'Testing', end-start
        learner.prune(validation)

        return test(learner, dataset, examples[start:end])
    finally:
        dataset.examples = examples

def cross_validation(learner, dataset, prune=False, k=10, trials=1):
    """Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; If trials>1, average over several shuffles."""
    examples = dataset.examples[:]
    trials = [single_cross_validation(learner, shuffle(dataset), k) \
                       for trial in range(trials)]
    dataset.examples = examples
    return sum(trials)/len(trials)

def single_cross_validation(learner, dataset, k):
    """A single run of k-fold cross-validation, which returns the mean"""
    num_testing = int(len(dataset.examples)/k)
    trials = [train_prune_and_test(learner, dataset, i*num_testing, (i+1)*num_testing) if prune else \
              train_and_test(learner, dataset, i*num_testing, (i+1)*num_testing) \
              for i in range(k)]
    return sum(trials)/len(trials)

def shuffle(dataset):
    random.shuffle(dataset.examples)
    return dataset
    
def learningcurve(learner, dataset, trials=10, sizes=None):
    if sizes == None:
        sizes = range(2, len(dataset.examples)-10, 2)
    def score(learner, size):
        return train_and_test(learner, shuffle(dataset), 0, size)
    return [(size, mean([score(learner, size) for t in range(trials)]))
            for size in sizes]
#______________________________________________________________________________

simpleData = DataSet(examples=[[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]], attrs=[[0, 1], [0, 1], [0,1]], target=0)

def testEntropy():
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

def testDT():
    print "DTs - testing copy should exclude/combine nodes correctly"
    leftTree = DecisionTree(0, "Temp", { 50 : 0, 80 : 1 })
    rightTree = 1
    tree = DecisionTree(1, "Humid", { True : leftTree, False : rightTree })
    
    exclude_temp = tree.copy("Temp", [[50, True, 0],\
                                      [50, True, 0],\
                                      [50, True, 1]], 2)
    new_tree = DecisionTree(1, "Humid", { True : 0, False : 1 })
    check(exclude_temp, new_tree)

    exclude_temp_2 = tree.copy("Temp", [[50, False, 1],\
                                        [50, False, 1],\
                                        [80, False, 1],\
                                        [50, True, 0],\
                                        [50, True, 0]], 2)
    new_tree_2 = DecisionTree(1, "Humid", { True: 0, False : 1})
    check(exclude_temp_2, new_tree_2)
    print
    
def testAccuracy():
    print "Learner - checking accuracy"
    iris1 = train_and_test(DecisionTreeLearner(), iris, 135, 150)
    check(iris1, 0.66666666666666663)
    orings1 = 0.7692307692307692
    check(orings1, train_and_test(DecisionTreeLearner(), orings, 10, 23))
    zoo1 = 0.71999999999999997
    check(zoo1, train_and_test(DecisionTreeLearner(), zoo, 75, 100))
    print

    print "Cross validation - checking for reasonable results"
    acc = cross_validation(DecisionTreeLearner(), iris, 5, 1)
    better(0, acc)
    prinr

def testPruning():
    print "Pruning - checking that pruning gives better results"
    iris1 = train_and_test(DecisionTreeLearner(), iris, 135, 150)
    iris2 = train_prune_and_test(DecisionTreeLearner(), iris, 130, 150, 0.66) #Standard 2/3-1/3 split
    better(iris1, iris2)

    print "Pruning - checking that pruning gives better results"
    zoo1 = train_and_test(DecisionTreeLearner(), zoo, 75, 100)
    zoo2 = train_prune_and_test(DecisionTreeLearner(), zoo, 70, 100, 0.66) #Standard 2/3-1/3 split
    better(zoo1, zoo2)

def testCrossV():
    
    

def better(original, better):
    if better > original: print "Test passed - improved from " + \
       str(original) + " to " + str(better)
    else: print "Test failed - deproved from " + \
          str(original) + " to " + str(better)

def check(result, expected):
    if result == expected: print "Test passed"
    else: print "Test failed. Expected " + str(expected) + ", got " + str(result)
