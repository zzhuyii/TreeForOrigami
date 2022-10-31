import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from DecisionTree import TreeMethod



# load the gripper performance data set
data1 = np.loadtxt('data4_MiuraSheetMat.txt',delimiter=',')
data2 = np.loadtxt('data4_TMPSheetMat.txt',delimiter=',')

fileTemp= open("data4_FeatureName.txt", "r")
tempfeatureName=fileTemp.readline()
tempfeatureName=tempfeatureName.split()
tempfeatureName=tempfeatureName[1:8]

# Separate the data and the label
dataFeature1 = (data1[:2000,1:8])
dataPerformance1 = (data1[:2000,8:14])
dataFeature2 = (data2[:2000,1:8])
dataPerformance2 = (data2[:2000,8:14])
dataFeature = np.concatenate((dataFeature1, dataFeature2), axis=0)
dataPerformance = np.concatenate((dataPerformance1, dataPerformance2), axis=0)

# Set up the feature database
# First we create the data matrix before using the OneHotEncoder
tempX = dataFeature
tempX[:,2]=tempX[:,2]*1000
tempX[:,3]=tempX[:,3]*1000
tempX[:,4]=tempX[:,4]*1000
tempX[:,5]=tempX[:,5]/1000000000
tempX[:,6]=tempX[:,6]/1000000000

categoryX=tempX[:,:2]
otherX=tempX[:,2:7]
categoryFeatureName=tempfeatureName[:2]
otherFeatureName=tempfeatureName[2:7]

# use the OneHotEncoder to transform the system
encoder = OneHotEncoder(sparse=False)
encodeX = encoder.fit_transform(categoryX)
encodeFeatureName=encoder.get_feature_names()

# Convert the names back to the normal names
for i in range(6):
    encodeFeatureName[i]=encodeFeatureName[i].replace('x0_','m=')
    encodeFeatureName[i]=encodeFeatureName[i].replace('x1_','n=')

# Reconstruct the feature database
totalX=np.concatenate((encodeX,otherX),axis=1)
featureName=np.concatenate((encodeFeatureName,otherFeatureName),axis=0)

# we need another feature to determine the type of pattern
patternType=np.concatenate((np.zeros((2000,1)),np.ones((2000,1))),axis=0)
totalX=np.concatenate((patternType,totalX),axis=1)
featureName=np.concatenate((['pattern'],featureName),axis=0)


totalY=(dataPerformance[:,2]<200)
# totalY=(dataPerformance[:,2]>200)*(dataPerformance[:,2]<400)
# totalY=(dataPerformance[:,2]>400)*(dataPerformance[:,2]<800)
# totalY=(dataPerformance[:,2]>800)*(dataPerformance[:,2]<1600)

print('data number that meets target', sum(totalY))

# Randomly split the data into testing and training sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    totalX,totalY, test_size = 0.4, random_state=0)

# Apply the proposed method for computing the design rules
tree=TreeMethod()
tree.setParameter(alpha=0.0001, depth=20, num_tree=100)
tree.train(X_train, Y_train, featureName)
tree.computeRule(X_train, Y_train, ruleNumber=1, minData=15)
tree.testRule(X_test,Y_test)
tree.printRule()

# Plot the rules for easy interpretation
featureMin=[0,0,0,0,0,0,0,0.5,1.0,1.0,1.0,1.0] # minimum value for feature
featureMax=[1,1,1,1,1,1,1,1.0,6.0,4.0,5.0,5.0] # maximum value for feature
tree.plotRule(featureMin,featureMax)


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

