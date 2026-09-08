import numpy as np
import math

class DecisionTree: 
    def __init__(self, criterion, max_depth, root):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):

        # Step 1: Check if all labels are the same
        if identical_labels(y):
            return y[0]

        
        # Step 2: Check if all datapoints have the same features
        if identical_features(X):
            most_common_label = np.bincount(y).argmax()

            return most_common_label


        # Step 3: Calculate information gain

        feature_gains = []
        label_entropy = decision_entropy(y)

        for feature in X.T:
            information_gain = label_entropy - conditional_entropy(feature, y)

            feature_gains.append(information_gain)


    def predict(self, X):
        # IMPLEMENT
        pass



# Helper methods 

def decision_entropy(column):
    
    threshold = np.mean(column)
    count_low = 0
    count_high = 0

    for label in column:
        if label <= threshold:
            count_low += 1
        else: count_high += 1


    sum = count_low + count_high

    decision_zero = (count_low/(sum)) * math.log2(count_low/(sum))
    decision_one = (count_high/sum) * math.log2(count_high/sum)

    entropy = -(decision_zero + decision_one)

    return entropy


def conditional_entropy(feature, y):

    threshold = np.mean(feature)

    count_low = 0
    count_high = 0
    label_low = []
    label_high = []

    for index, value in enumerate(feature):
        if value <= threshold:
            count_low += 1
            label_low.append(int(y[index]))

        else: 
            count_high += 1
            label_high.append(int(y[index]))

    sum = count_low + count_high

    
    entropy = count_low/sum * decision_entropy(label_low) + count_high/sum * decision_entropy(label_high)
    
    return entropy


def identical_labels(y):
    first_label = y[0]

    for label in y:
        if label != first_label:
            return False
        
    return True

def identical_features(X):

    first_row = X[0]

    for row in X:
        for index in range(len(row)):
            if row[index] != first_row[index]:
                return False

    return True