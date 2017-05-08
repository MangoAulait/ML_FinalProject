import os.path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

path_to_nonbot = ''
path_to_bot = ''
path_to_userTweets = '/Users/kanye/Documents/cs/6923/Final_Project/userTweets/'

# Retrive data
data_nonbot = pd.read_csv(path_to_nonbot + 'nonbots_data.csv', header = 0, encoding='latin-1')
data_bot = pd.read_csv(path_to_bot + 'bots_data.csv', header = 0, encoding='latin-1')
data = pd.concat([data_nonbot, data_bot])
data = pd.read_csv('training_data_2_csv_UTF.csv', header = 0, encoding='latin-1')
print data.shape
# Fill na
data = data.fillna(value = '0')


# Cross validation
X_train, X_test, y_train, y_test = train_test_split(np.array(data[data.columns[:19]]), np.array(data['bot']), test_size=0.25, random_state=0)

# LR for numerical feature
#feature_selected = [6,7,8,10,11,12,15,16,17]
feature_selected = [6,7,8,10,11,12]
#feature_selected = [6,7,10,12]

X_train_for_numerical_feature = X_train[:,feature_selected]
X_test_for_numerical_feature = X_test[:,feature_selected]

#clf1 = linear_model.LogisticRegression(penalty = 'l2', C = 0.1)
#clf1 = SVC(probability = True)
#clf1 = RandomForestClassifier()
#clf1 = KNeighborsClassifier(n_neighbors = 2)
#clf1 = MultinomialNB()
clf1 = MLPClassifier()

clf1 = clf1.fit(X_train_for_numerical_feature, y_train)
#print clf1.feature_importances_
predicted_proba1 = clf1.predict_proba(X_test_for_numerical_feature)

predicted = clf1.predict(X_test_for_numerical_feature)
print 'Accuracy score: ' + str(accuracy_score(y_test, predicted))
print '---------------------------------------------'

# LR for string feature
def listOfList_to_list(listOfList):
    new_list = ['']
    for l in listOfList.tolist():
        new_list.append(l[0])
    return np.array(new_list[1:])
predicted_probas = []

for i in [2,3,4,5,9,13,14,18]:
    clf = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('clf', linear_model.LogisticRegression())])
    clf = clf.fit(listOfList_to_list(X_train[:,[i]]), y_train)
    predicted_proba = clf.predict_proba(listOfList_to_list(X_test[:,[i]]))
    predicted_probas.append(predicted_proba.tolist())

    predicted = clf.predict(listOfList_to_list(X_test[:,[i]]))
    print 'Accuracy score: ' + str(accuracy_score(y_test, predicted))

predicted_probas.append(predicted_proba1.tolist())

probas = [0.0 for i in range(0,len(predicted_proba1))]
for i in range(0, len(predicted_probas)):
    for j in range(0,len(predicted_proba1)):
        probas[j] += predicted_probas[i][j][1]

for i in range(0,len(probas)):
    if probas[i] / len(predicted_probas) > 0.5:
        probas[i] = 1
    else:
        probas[i] = 0
print 'Accuracy score (no hard-coded rule): ' + str(accuracy_score(y_test, probas))

# add some hard-coded rule
for index, name in enumerate(X_test.T[2].tolist()):
    if 'bot' in name.lower():
        probas[index] = 1
for index, name in enumerate(X_test.T[4].tolist()):
    if 'bot' in name.lower():
        probas[index] = 1



# metrics & auc
#print metrics.classification_report(y_test, probas,
#     target_names=['0','1'])
#print 'Auc score: ' + str(roc_auc_score(y_test, probas))
print 'Accuracy score (hard-coded rule): ' + str(accuracy_score(y_test, probas))








