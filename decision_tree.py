import numpy as np

class DecisionTree: 
    def __init__(self, criterion, max_depth, root):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None


    def identical_features(X):

        first_row = X[0]

        for row in X:
            for index in range(len(row)):
                if row[index] != first_row[index]:
                    return False

        return True


    def fit(self, X, y):

        first_leaf = y[0]

        # Step 1: Check if all labels are the same
        for i in range(y): 
            if i != first_leaf:
                break

        else: return first_leaf # Doesn't matter which leaf

        # Step 2: Check if all datapoints have the same features

        if self.identical_features(X):
            most_common_label = np.bincount(y).argmax()

            return most_common_label

        

        


    def predict(self, X):
        # IMPLEMENT
        pass

