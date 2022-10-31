import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from DecisionTree import TreeMethod

# load the shape fitting data set
data = np.loadtxt('data3_ShapeFit.txt',delimiter=',')

fileTemp= open("data3_FeatureName.txt" , "r")
featureName=fileTemp.readline()
featureName=featureName.split()
featureName=featureName[0:7]

fileTemp= open("data3_FeatureName.txt" , "r")
performanceName=fileTemp.readline()
performanceName=performanceName.split()
performanceName=performanceName[7:11]

# Separate the data and the label
dataFeature = (data[0:3000,0:7])
dataPerformance = (data[0:3000,7:11])


# Prepare the data with the one-hot-encoder for categorical lables
categoryX=dataFeature[:,0]
categoryX=categoryX.reshape(-1,1)
otherX=dataFeature[:,1:7]
categoryFeatureName=featureName[0]
otherFeatureName=featureName[1:7]

# use the OneHotEncoder to transform the categorical data
encoder = OneHotEncoder(sparse=False)
encodeX = encoder.fit_transform(categoryX)
encodeFeatureName=encoder.get_feature_names()

# Convert the names back to the normal names
for i in range(4):
    encodeFeatureName[i]=encodeFeatureName[i].replace('x0_','m=')


# Reconstruct the feature database
totalX=np.concatenate((encodeX,otherX),axis=1)
featureName=np.concatenate((encodeFeatureName,otherFeatureName),axis=0)

# Set up the feature database unit for easier reading
totalX[:,4]=totalX[:,4]*1000 # to mm
totalX[:,5]=totalX[:,5]*1000 # to mm
totalX[:,6]=totalX[:,6]*1000 # to mm
totalX[:,7]=totalX[:,7]*1000 # to mm
totalX[:,8]=totalX[:,8]*1000 # to mm
totalX[:,9]=totalX[:,9]*1000 # to mm

# Set up the label based on different target performance
# Target 1
# totalY_target1=(dataPerformance[:,1]==1) # Has snaping happend
# totalY=totalY_target1
# print('data number that meets target', sum(totalY))

# Target 2
totalY_target1=(dataPerformance[:,0]>800) # Stiffness in Z (N/3m)
totalY_target2=dataPerformance[:,2]>600 # Stiffness in X (N/m)
totalY_target3=dataPerformance[:,3]<0.1 # error 
totalY=totalY_target1*totalY_target2*totalY_target3
print('data number that meets target', sum(totalY))

# Target 3
# totalY_target1=(dataPerformance[:,0]>300) # Stiffness in Z (N/m)
# totalY_target2=dataPerformance[:,2]>150 # Stiffness in X (N/m)
# totalY_target3=dataPerformance[:,3]<0.06 # error 
# totalY=totalY_target1*totalY_target2*totalY_target3
# print('data number that meets target', sum(totalY))


# Randomly split the data into testing and training sets
# we use a common hold hout setup with 0.4 test size
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    totalX,totalY, test_size = 0.4, random_state= 10)

#
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    totalX,totalY, test_size = 0.4, random_state= np.random)

# Apply the proposed method for computing the design rules
tree=TreeMethod()
tree.setParameter(alpha=0.0001, depth=20, num_tree=100)
tree.train(X_train, Y_train, featureName)
tree.computeRule(X_train, Y_train, ruleNumber=1)
tree.testRule(X_test,Y_test)
tree.printRule()

# Plot the rules for easy interpretation
featureMin=[0,0,0,0,0.5,1.0,1.0,100.0,10.0,50.0] # minimum value for feature
featureMax=[1,1,1,1,1.0,6.0,4.0,300.0,40.0,250.0] # maximum value for feature
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
print('precision is: ',Precision, '\n')

# accuray of boosting from SK learn
accuratePredict = sum(skB_Y_pred==Y_test)
accurateRate = accuratePredict / len(skB_Y_pred)
print('sklearn Boosting')
print('the accuracy is:', accurateRate)
# Precision and Recall
accurateClass1 = sum((skB_Y_pred==Y_test) * (skB_Y_pred==1))
Precision = accurateClass1 / sum(skB_Y_pred==1)
print('precision is: ',Precision, '\n')

# accuray of neural network from SK learn
accuratePredict = sum(skMLP_Y_pred==Y_test)
accurateRate = accuratePredict / len(skMLP_Y_pred)
print('sklearn MLP')
print('the accuracy is:', accurateRate)
# Precision 
accurateClass1 = sum((skMLP_Y_pred==Y_test) * (skMLP_Y_pred==1))
Precision = accurateClass1 / sum(skMLP_Y_pred==1)
print('precision is: ',Precision, '\n')

# accuray of knn from SK learn
accuratePredict = sum(skKNN_Y_pred==Y_test)
accurateRate = accuratePredict / len(skKNN_Y_pred)
print('sklearn KNN')
print('the accuracy is:', accurateRate)
# Precision 
accurateClass1 = sum((skKNN_Y_pred==Y_test) * (skKNN_Y_pred==1))
Precision = accurateClass1 / sum(skKNN_Y_pred==1)
print('precision is: ',Precision, '\n')