import pandas as pd
import numpy as np
import sklearn.linear_model as sk_linear
import sklearn.naive_bayes as sk_bayes
import sklearn.tree as sk_tree
import sklearn.svm as sk_svm
import sklearn.neural_network as sk_nn
import sklearn.neighbors as sk_neighbors
import sklearn.model_selection as sk_model_selection
from joblib import dump

train_x = pd.read_csv('preprocessed_train_A.csv')
train_y = pd.read_csv('preprocessed_train_A_label.csv')
train_y = train_y['0']

def run(model):
    accs = sk_model_selection.cross_val_score(model, train_x, y=train_y, cv=10, n_jobs=1)
    print('score: ', accs.mean())

# LogisticRegression
# score: 0.8011811023622046, verify score: 0.8729480199190002
model = sk_linear.LogisticRegression(penalty='l2', dual=False, C=1.0, n_jobs=1, random_state=20, fit_intercept=True)

# GaussianNB
# score: 0.776771653543307, verify score: 0.8285853018579511
# model = sk_bayes.GaussianNB()

# Decision Tree
# score: 0.7314960629921259, verify score: 0.7814072915517596
# model = sk_tree.DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0)

# NN (Neural Network)
# score: 0.7795275590551182, verify score: 0.8239748963865294
# model = sk_nn.MLPClassifier(activation='logistic', solver='lbfgs', max_iter=300, random_state=20)

run(model)

model.fit(train_x, train_y)
dump(model, 'model.joblib')




