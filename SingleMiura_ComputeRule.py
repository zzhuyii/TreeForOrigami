import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from DecisionTree import TreeMethod


# load the gripper performance data set
data1 = np.loadtxt('data0_SingleMiura.txt',delimiter=',')

fileTemp= open("data0_FeatureName.txt", "r")
tempfeatureName=fileTemp.readline()
tempfeatureName=tempfeatureName.split()
featureName=tempfeatureName[0:4]

# Separate the data and the label
dataFeature = (data1[:1000,0:4])
dataPerformance = (data1[:1000,4:7])

# Set up the feature database
totalX=dataFeature

# Set up the label for the data base
totalY=(dataPerformance[:,1]<6000)
print('data number that meets target', sum(totalY))

# Randomly split the data into testing and training sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    totalX,totalY, test_size = 0.4, random_state=0)

# Apply the proposed method for computing the design rules
tree=TreeMethod()
tree.setParameter(alpha=0.0001, depth=20, num_tree=100)
tree.train(X_train, Y_train, featureName)
tree.computeRule(X_train, Y_train, ruleNumber=1)
tree.testRule(X_test,Y_test)
tree.printRule()

# Predict the results of the testing set using the embeded Random Forest in
# the proposed method
Y_pred=tree.predict(X_test)

# calculate the accruacy of the prediction
accuratePredict = sum(Y_pred==Y_test)
accurateRateRF = accuratePredict / len(Y_pred)

# Precision of the prediction
accurateClass1 = sum((Y_pred==Y_test) * (Y_pred==1))
Precision = accurateClass1 / sum(Y_pred==1)

print('testing data Num for two classes: ')
print(sum(Y_pred==0), sum(Y_pred==1), '\n')

print('Embeded Random Forest')
print('accuracy is: ', accurateRateRF)
print('precision is: ', Precision, '\n') 

# with the sklearn kit
# We will compare the result with other methods from sklearn pacakge
# Use the random forest to predict the results of the testing set
skRF=RandomForestClassifier(n_estimators=100,min_samples_leaf=4,
                            min_samples_split=8,criterion="entropy",
                            class_weight='balanced')
skBoost=GradientBoostingClassifier(n_estimators=200,criterion='friedman_mse'
                                   ,min_samples_leaf=4,min_samples_split=8)
skMLP=MLPClassifier(hidden_layer_sizes=(32, 64, 64, 32), 
                    max_iter=1000)
skKNN=KNeighborsClassifier()


# Train the other machine leanring algorithm
skRF.fit(X_train, Y_train)
skBoost.fit(X_train, Y_train)
skMLP.fit(X_train, Y_train)
skKNN.fit(X_train, Y_train)

# Use the other mahchine learning to predict the data
sk_Y_pred=skRF.predict(X_test)
skB_Y_pred=skBoost.predict(X_test)
skMLP_Y_pred=skMLP.predict(X_test)
skKNN_Y_pred=skKNN.predict(X_test)

# calculate the accruacy of other method
# accuray of RF from SK learn
accuratePredict = sum(sk_Y_pred==Y_test)
accurateRate = accuratePredict / len(sk_Y_pred)
print('sklearn Random Forest')
print('the accuracy is:', accurateRate)
# Precision 
accurateClass1 = sum((sk_Y_pred==Y_test) * (sk_Y_pred==1))
Precision = accurateClass1 / sum(sk_Y_pred==1)
Recall = accurateClass1 / sum(Y_test==1)
print('precision is: ',Precision, '\n')

# accuray of boosting from SK learn
accuratePredict = sum(skB_Y_pred==Y_test)
accurateRate = accuratePredict / len(skB_Y_pred)
print('sklearn Boosting')
print('the accuracy is:', accurateRate)
# Precision 
accurateClass1 = sum((skB_Y_pred==Y_test) * (skB_Y_pred==1))
Precision = accurateClass1 / sum(skB_Y_pred==1)
Recall = accurateClass1 / sum(Y_test==1)
print('precision is: ',Precision, '\n')

# accuray of neural network from SK learn
accuratePredict = sum(skMLP_Y_pred==Y_test)
accurateRate = accuratePredict / len(skMLP_Y_pred)
print('sklearn MLP')
print('the accuracy is:', accurateRate)
# Precision 
accurateClass1 = sum((skMLP_Y_pred==Y_test) * (skMLP_Y_pred==1))
Precision = accurateClass1 / sum(skMLP_Y_pred==1)
Recall = accurateClass1 / sum(Y_test==1)
print('precision is: ',Precision, '\n')

# accuray of knn from SK learn
accuratePredict = sum(skKNN_Y_pred==Y_test)
accurateRate = accuratePredict / len(skKNN_Y_pred)
print('sklearn KNN')
print('the accuracy is:', accurateRate)
# Precision 
accurateClass1 = sum((skKNN_Y_pred==Y_test) * (skKNN_Y_pred==1))
Precision = accurateClass1 / sum(skKNN_Y_pred==1)
Recall = accurateClass1 / sum(Y_test==1)
print('precision is: ',Precision, '\n')
