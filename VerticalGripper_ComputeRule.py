import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from DecisionTree import TreeMethod

# load the gripper performance data set
data1 = np.loadtxt('data2_MiuraBased.txt',delimiter=',')
data2 = np.loadtxt('data2_SplittedMiura.txt',delimiter=',')
data3 = np.loadtxt('data2_TwoPanel.txt',delimiter=',')

fileTemp= open("data2_FeatureName.txt" , "r")
featureName=fileTemp.readline()
featureName=featureName.split()
featureName=featureName[0:8]

fileTemp= open("data2_FeatureName.txt" , "r")
performanceName=fileTemp.readline()
performanceName=performanceName.split()
performanceName=performanceName[8:12]

# Separate the data and the label
dataFeature1 = (data1[0:2000,0:8])
dataPerformance1 = (data1[0:2000,8:12])
dataFeature2 = (data2[0:2000,0:8])
dataPerformance2 = (data2[0:2000,8:12])
dataFeature3 = (data3[0:2000,0:8])
dataPerformance3 = (data3[0:2000,8:12])

dataFeature = np.concatenate((dataFeature1, dataFeature2), axis=0)
dataPerformance = np.concatenate((dataPerformance1, dataPerformance2), axis=0)
dataFeature = np.concatenate((dataFeature, dataFeature3), axis=0)
dataPerformance = np.concatenate((dataPerformance, dataPerformance3), axis=0)

# Prepare the data to set up the one-hot-encoder
categoryX=dataFeature[:,0]
categoryX=categoryX.reshape(-1,1)
otherX=dataFeature[:,1:8]
categoryFeatureName=featureName[0]
otherFeatureName=featureName[1:8]

# use the OneHotEncoder to transform the system
encoder = OneHotEncoder(sparse=False)
encodeX = encoder.fit_transform(categoryX)
encodeFeatureName=encoder.get_feature_names()

# Reconstruct the feature database
totalX=np.concatenate((encodeX,otherX),axis=1)
featureName=np.concatenate((encodeFeatureName,otherFeatureName),axis=0)

# Set up the feature database unit for easier reading
totalX[:,3]=totalX[:,3]*1000 # to mm
totalX[:,4]=totalX[:,4]*1000 # to mm
totalX[:,5]=totalX[:,5]*1000000 # to um
totalX[:,6]=totalX[:,6]*1000000 # to um
totalX[:,7]=totalX[:,7]*1000000 # to um
totalX[:,9]=totalX[:,9]*1000000 # to um

# Set up the label for the data base
# Target 1
# totalY_target1=(dataPerformance[:,0]>10)*(dataPerformance[:,0]<40) # frequency (Hz)
# totalY_target2=dataPerformance[:,1]<0.2 # input power (W)
# totalY_target3=dataPerformance[:,2]<200 # maximum temperature (degree C)
# totalY=totalY_target1*totalY_target2*totalY_target3
# print('data number that meets target', sum(totalY))

# Target 2
totalY_target1=(dataPerformance[:,0]>35)*(dataPerformance[:,0]<50) # frequency (Hz)
totalY_target2=(dataPerformance[:,1]>0.3) # input power (W)
totalY=totalY_target1*totalY_target2
print('data number that meets target', sum(totalY))

# Target 3
# totalY_target1=(dataPerformance[:,0]>10) # Frequency (Hz)
# totalY_target2=(dataPerformance[:,1]<0.7) # input power (W)
# totalY_target3=(dataPerformance[:,2]<500) # Maximum Temperature (degree C)
# totalY_target4=(dataPerformance[:,3]>0.002)# stiffness (N/m)
# totalY=totalY_target1*totalY_target2*totalY_target3*totalY_target4
# print('data number that meets target', sum(totalY))


# Randomly split the data into testing and training sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    totalX,totalY, test_size = 0.4, random_state=0)

# Create the tree method for extracting the design method 
tree=TreeMethod()

# Use the proposed method to compute rules
tree.setParameter(alpha=0.0001, depth=20, num_tree=100)
tree.train(X_train, Y_train, featureName)
tree.computeRule(X_train, Y_train, 
                ruleNumber=2)
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
print('accuracy is:', accurateRate)
# Precision 
accurateClass1 = sum((sk_Y_pred==Y_test) * (sk_Y_pred==1))
Precision = accurateClass1 / sum(sk_Y_pred==1)
print('precision is: ',Precision, '\n')

# accuray of boosting from SK learn
accuratePredict = sum(skB_Y_pred==Y_test)
accurateRate = accuratePredict / len(skB_Y_pred)
print('sklearn Boosting')
print('accuracy is:', accurateRate)
# Precision 
accurateClass1 = sum((skB_Y_pred==Y_test) * (skB_Y_pred==1))
Precision = accurateClass1 / sum(skB_Y_pred==1)
print('precision is: ',Precision, '\n')

# accuray of neural network from SK learn
accuratePredict = sum(skMLP_Y_pred==Y_test)
accurateRate = accuratePredict / len(skMLP_Y_pred)
print('sklearn MLP')
print('accuracy is:', accurateRate)
# Precision 
accurateClass1 = sum((skMLP_Y_pred==Y_test) * (skMLP_Y_pred==1))
Precision = accurateClass1 / sum(skMLP_Y_pred==1)
print('precision is: ',Precision, '\n')


# accuray of knn from SK learn
accuratePredict = sum(skKNN_Y_pred==Y_test)
accurateRate = accuratePredict / len(skKNN_Y_pred)
print('sklearn KNN')
print('accuracy is:', accurateRate)
# Precision 
accurateClass1 = sum((skKNN_Y_pred==Y_test) * (skKNN_Y_pred==1))
Precision = accurateClass1 / sum(skKNN_Y_pred==1)
print('precision is: ',Precision, '\n')

