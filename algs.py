
#______________________________________________________________________________

def rms_error(predictions, targets):
    return math.sqrt(ms_error(predictions, targets))

def ms_error(predictions, targets):
    return mean([(p - t)**2 for p, t in zip(predictions, targets)])

def mean_error(predictions, targets):
    return mean([abs(p - t) for p, t in zip(predictions, targets)])

def mean_boolean_error(predictions, targets):
    return mean([(p != t)   for p, t in zip(predictions, targets)])


#______________________________________________________________________________

class NaiveBayesLearner(Learner):
    
    def train(self, dataset):
        """Just count the target/attr/val occurences.
        Count how many times each value of each attribute occurs.
        Store count in N[targetvalue][attr][val]. Let N[attr][None] be the
        sum over all vals."""
        N = {}
        self.dataset = dataset
        ## Initialize to 0
        for gv in self.dataset.values[self.dataset.target]:
            N[gv] = {}
            for attr in self.dataset.attrs:
                N[gv][attr] = {}
                for val in self.dataset.values[attr]:
                    N[gv][attr][val] = 0
                    N[gv][attr][None] = 0
        ## Go thru examples
        for example in self.dataset.examples:
            Ngv = N[example[self.dataset.target]]
            for attr in self.dataset.attrs:
                Ngv[attr][example[attr]] += 1
                Ngv[attr][None] += 1
        self._N = N

    def N(self, targetval, attr, attrval):
       "Return the count in the training data of this combination."
       try:
          return self._N[targetval][attr][attrval]
       except KeyError:
          return 0

    def P(self, targetval, attr, attrval):
        """Smooth the raw counts to give a probability estimate.
        Estimate adds 1 to numerator and len(possible vals) to denominator."""
        return ((self.N(targetval, attr, attrval) + 1.0) /
                (self.N(targetval, attr, None) + len(self.dataset.values[attr])))

    def predict(self, example):
        """Predict the target value for example. Consider each possible value,
        choose the most likely, by looking at each attribute independently."""
        possible_values = self.dataset.values[self.dataset.target]
        def class_probability(targetval):
            return product([self.P(targetval, a, example[a])
                            for a in self.dataset.inputs])  #removed ',1' from the arglist as product takes a list of vals to mult
        return argmax(possible_values, class_probability)

#______________________________________________________________________________

class NearestNeighborLearner(Learner):

    def __init__(self, k=1):
        "k-NearestNeighbor: the k nearest neighbors vote."
        self.k = k

    def predict(self, example):
        """With k=1, find the point closest to example.
        With k>1, find k closest, and have them vote for the best."""
        if self.k == 1:
            neighbor = argmin(self.dataset.examples,
                              lambda e: self.distance(e, example))
            return neighbor[self.dataset.target]
        else:
            ## Maintain a sorted list of (distance, example) pairs.
            ## For very large k, a PriorityQueue would be better
            best = [] 
            for e in self.dataset.examples:
                d = self.distance(e, example)
                if len(best) < self.k: 
                    best.append((d, e))
                elif d < best[-1][0]:
                    best[-1] = (d, e)
                    best.sort()
            return mode([e[self.dataset.target] for (d, e) in best])

    def distance(self, e1, e2):
        return mean_boolean_error(e1, e2)

#______________________________________________________________________________

class EnsembleLearner(Learner):
    """Given a list of learning algorithms, have them vote."""

    def __init__(self, learners=[]):
        self.learners=learners

    def train(self, dataset):
        for learner in self.learners:
           learner.train(dataset)

    def predict(self, example):
        return mode([learner.predict(example) for learner in self.learners])

#______________________________________________________________________________

