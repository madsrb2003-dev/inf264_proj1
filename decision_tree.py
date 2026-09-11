import numpy as np
import math

class DecisionTree: 
    def __init__(self, criterion, max_depth, root):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None


    # Made by Chat
    def print_tree(self, node, feature_names, indent=""):
        if not isinstance(node, dict):
            print(f"{indent}Predict: {node}")
            return

        name = feature_names[node["feature"]]
        print(f"{indent}{name} <= {node['threshold']:.3f}")

        print(f"{indent}  Yes:")
        self.print_tree(node["left"], feature_names, indent + "    ")

        print(f"{indent}  No:")
        self.print_tree(node["right"], feature_names, indent + "    ")



    def fit(self, X, y, depth=0):

        # Step 1: Check if all labels are the same
        if identical_labels(y):
            return y[0]

        
        # Step 2: Check if all datapoints have the same features
        elif identical_features(X) or (self.max_depth is not None and depth >= self.max_depth):
            most_common_label = np.bincount(y).argmax()

            return most_common_label


        # Step 3: Calculate information gain
        else: 
            feature_gains = []
            label_entropy = decision_entropy(y)

            for feature in X.T:
                if np.all(feature == feature[0]):
                    feature_gains.append(-np.inf)
                    continue
                information_gain = label_entropy - conditional_entropy(feature, y)

                feature_gains.append(information_gain)


        # Split the branch
            split_feature = np.argmax(feature_gains)
            column = X[:, split_feature]
            threshold = np.mean(column)

            low_mask = column <= threshold
            high_mask = ~low_mask

            X_low = X[low_mask]
            X_high = X[high_mask]
            y_low = y[low_mask]
            y_high = y[high_mask]

            node =  {
                "feature": split_feature,
                "threshold": threshold,
                "left": self.fit(X_low, y_low, depth + 1),
                "right": self.fit(X_high, y_high, depth + 1)
            }

        if depth == 0:
            self.root = node

        return node

        

    def predict(self, X):

        # Beginning of the tree
        node = self.root
        predicted_labels = []

        for row in X: 
        
            while isinstance(node, dict):
                feature = node["feature"]
                threshold = node["threshold"]

                if row[feature] <= threshold:
                    node = node["left"]

                else:
                    node = node["right"]

            predicted_labels.append[node]

        return predicted_labels



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

    if count_low == len(column):
        return 0

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