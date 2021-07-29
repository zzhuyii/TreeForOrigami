# This code use the sklearn package to apply a decission tree method onto the 
# gripper data base.

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


from DecisionTree import TreeMethod
import numpy as np


# load the gripper performance data set
data1 = np.loadtxt('dataset_OurDesign.txt',skiprows=1)
data2 = np.loadtxt('dataset_Howell.txt',skiprows=1)
data3 = np.loadtxt('dataset_SimpleGripper.txt',skiprows=1)

fileTemp= open("dataset_OurDesign.txt" , "r")
featureName=fileTemp.readline()
featureName=featureName.split()
featureName=featureName[1:9]

# Separate the data and the label
dataFeature1 = (data1[:2000,1:9])
dataPerformance1 = (data1[0:2000,9:13])

dataFeature2 = (data2[:2000,1:9])
dataPerformance2 = (data2[:2000,9:13])

dataFeature3 = (data3[:2000,1:9])
dataPerformance3 = (data3[:2000,9:13])

dataFeature = np.concatenate((dataFeature1, dataFeature2), axis=0)
dataPerformance = np.concatenate((dataPerformance1, dataPerformance2), axis=0)

dataFeature = np.concatenate((dataFeature, dataFeature3), axis=0)
dataPerformance = np.concatenate((dataPerformance, dataPerformance3), axis=0)


# Set up the feature database
totalX = dataFeature
totalX[:,1]=totalX[:,1]*1000
totalX[:,2]=totalX[:,2]*1000
totalX[:,3]=totalX[:,3]*1000000
totalX[:,4]=totalX[:,4]*1000000
totalX[:,5]=totalX[:,5]*1000000
totalX[:,7]=totalX[:,7]*1000000


# Set up the label for the data base

# totalY_target1=dataPerformance[:,0]>30
# totalY_target2=dataPerformance[:,1]<0.6
# totalY_target3=dataPerformance[:,1]>0.3
# totalY=totalY_target1*totalY_target2*totalY_target3
# thresholdData=30;
# thresholdAccuracy=0.95;

# totalY_target1=dataPerformance[:,1]<0.4
# totalY_target2=dataPerformance[:,3]>0.0004
# totalY=totalY_target1*totalY_target2
# thresholdData=30;
# thresholdAccuracy=0.95;

totalY_target1=dataPerformance[:,0]>30
totalY_target2=dataPerformance[:,1]<0.5
totalY_target3=dataPerformance[:,2]<350
totalY_target4=dataPerformance[:,3]>0.0003
totalY=totalY_target1*totalY_target2*totalY_target3*totalY_target4
thresholdData=45
thresholdAccuracy=0.95

# totalY_target1=dataPerformance[:,0]<80
# totalY_target2=dataPerformance[:,0]>60
# totalY_target3=dataPerformance[:,1]<0.6
# totalY=totalY_target1*totalY_target2*totalY_target3



# Randomly split the data into testing and training sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    totalX,totalY, test_size = 0.2, random_state=0)

# Create the tree method for extracting the design method
tree=TreeMethod()
skRF=RandomForestClassifier(n_estimators=100,min_samples_leaf=4,
                            min_samples_split=8,criterion="gini",
                            class_weight='balanced')
skBoost=GradientBoostingClassifier(n_estimators=200,criterion='friedman_mse'
                                   ,min_samples_leaf=4,min_samples_split=8)
skKNN=KNeighborsClassifier()


# set the data
tree.setData(X_train, Y_train, featureName)
skRF.fit(X_train, Y_train)
skBoost.fit(X_train, Y_train)
skKNN.fit(X_train, Y_train)


# set the design parameter
tree.setParameter(alpha=0.0001, depth=14, num_tree=100)

# train the random forest
tree.train()
        
# Compute the rules and print them
tree.collectRule()
tree.selectRule(thresholdReliability=thresholdAccuracy,
                   thresholdDataNum=thresholdData,)
tree.testRule(X_test,Y_test)
tree.printRuleForClass(1)


# Use the trandom forest to predict the results of the testing set
Y_pred=tree.predict(X_test)


# calculate the accruacy
accuratePredict = sum(Y_pred==Y_test)
accurateRateRF = accuratePredict / len(Y_pred)

accurateClass1 = sum((Y_pred==Y_test) * (Y_pred==0))
accurateRateRFClass1 = accurateClass1 / sum(Y_pred==0)

accurateClass2 = sum((Y_pred==Y_test) * (Y_pred==1))
accurateRateRFClass2 = accurateClass2 / sum(Y_pred==1)


print('testing data Num for two classes: ')
print(sum(Y_pred==0), sum(Y_pred==1), '\n')

print('Random Forest')
print('the accuracy is: ')
print(accurateRateRF)
print('the perclass accuracy is: ')
print(accurateRateRFClass1) 
print(accurateRateRFClass2, '\n')


# with the sklearn
# Use the trandom forest to predict the results of the testing set
sk_Y_pred=skRF.predict(X_test)
skB_Y_pred=skBoost.predict(X_test)
skKNN_Y_pred=skKNN.predict(X_test)

# calculate the accruacy of other method
# accuray of RF from SK learn
accuratePredict = sum(sk_Y_pred==Y_test)
accurateRate = accuratePredict / len(sk_Y_pred)
print('sklearn Random Forest')
print('the accuracy is:', accurateRate)

accurateClass1 = sum((sk_Y_pred==Y_test) * (sk_Y_pred==0))
accurateRateRFClass1 = accurateClass1 / sum(sk_Y_pred==0)

accurateClass2 = sum((sk_Y_pred==Y_test) * (sk_Y_pred==1))
accurateRateRFClass2 = accurateClass2 / sum(sk_Y_pred==1)

print('the perclass accuracy is: ')
print(accurateRateRFClass1) 
print(accurateRateRFClass2, '\n')


# accuray of boosting from SK learn

accuratePredict = sum(skB_Y_pred==Y_test)
accurateRate = accuratePredict / len(skB_Y_pred)
print('sklearn Boosting')
print('the accuracy is:', accurateRate)


accurateClass1 = sum((skB_Y_pred==Y_test) * (skB_Y_pred==0))
accurateRateRFClass1 = accurateClass1 / sum(skB_Y_pred==0)

accurateClass2 = sum((skB_Y_pred==Y_test) * (skB_Y_pred==1))
accurateRateRFClass2 = accurateClass2 / sum(skB_Y_pred==1)

print('the perclass accuracy is: ')
print(accurateRateRFClass1) 
print(accurateRateRFClass2, '\n')


# accuray of knn from SK learn

accuratePredict = sum(skKNN_Y_pred==Y_test)
accurateRate = accuratePredict / len(skKNN_Y_pred)
print('sklearn KNN')
print('the accuracy is:', accurateRate)

accurateClass1 = sum((skKNN_Y_pred==Y_test) * (skKNN_Y_pred==0))
accurateRateRFClass1 = accurateClass1 / sum(skKNN_Y_pred==0)

accurateClass2 = sum((skKNN_Y_pred==Y_test) * (skKNN_Y_pred==1))
accurateRateRFClass2 = accurateClass2 / sum(skKNN_Y_pred==1)

print('the perclass accuracy is: ')
print(accurateRateRFClass1) 
print(accurateRateRFClass2, '\n')
