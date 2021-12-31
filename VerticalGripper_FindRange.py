from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from DecisionTree import TreeMethod
import numpy as np

# load the gripper performance data set
data1 = np.loadtxt('data2_MiuraBased.txt',delimiter=',')
data2 = np.loadtxt('data2_SplittedMiura.txt',delimiter=',')
data3 = np.loadtxt('data2_TwoPanel.txt',delimiter=',')

fileTemp= open("data2_FeatureName.txt" , "r")
varName=fileTemp.readline()
varName=varName.split()

data = np.concatenate((data1, data2), axis=0)
data = np.concatenate((data, data3), axis=0)

categoryX=data[:,0]
categoryX=categoryX.reshape(-1,1)
otherX=data[:,1:12]
categoryFeatureName=varName[0]
otherFeatureName=varName[1:12]

# use the OneHotEncoder to transform the system
encoder = OneHotEncoder(sparse=False)
encodeX = encoder.fit_transform(categoryX)
encodeFeatureName=encoder.get_feature_names()

# Reconstruct the feature database
totalX=np.concatenate((encodeX,otherX),axis=1)
featureName=np.concatenate((encodeFeatureName,otherFeatureName),axis=0)

dataPerformance=totalX[:,10:14]
featureName=featureName[:10]

# Set up the feature database for easier reading
totalX[:,3]=totalX[:,3]*1000 # to mm
totalX[:,4]=totalX[:,4]*1000 # to mm
totalX[:,5]=totalX[:,5]*1000000 # to um
totalX[:,6]=totalX[:,6]*1000000 # to um
totalX[:,7]=totalX[:,7]*1000000 # to um
totalX[:,9]=totalX[:,9]*1000000 # to um

# Set up the label for the data base
# Target 2
totalY_target1=(dataPerformance[:,0]>35)*(dataPerformance[:,0]<50) # frequency (Hz)
totalY_target2=(dataPerformance[:,1]>0.3) # input power (W)
totalY=totalY_target1*totalY_target2
print('data number that meets target', sum(totalY))

# Randomly split the data into testing and training sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    totalX,totalY, test_size = 0.4, random_state=0)


dataPerformanceTrain = X_train[:,10:14]
X_train=X_train[:,:10]

dataPerformanceTest = X_test[:,10:14]
X_test=X_test[:,:10]


# Create the tree method for extracting the design method 
tree=TreeMethod()

# set the design parameter
tree.setParameter(alpha=0.0001, depth=20, num_tree=100)

# train the random forest
tree.train(X_train, Y_train, featureName)
        
# Compute the rules and print them
tree.computeRule(X_train,Y_train,
                 ruleNumber=2)

# Test the rules
tree.testRule(X_test,Y_test)

# Print the rules
tree.printRule()

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

# Find the datapoints that satisfies the rules
tree.findDataFitFinalRule(X_train, Y_train, X_test, Y_test)
testDataFitRules=tree.dataFitFinalRule_TestData
trainDataFitRules=tree.dataFitFinalRule_TrainData

testDataFitRules0=testDataFitRules[str(0)]
testDataFitRules1=testDataFitRules[str(1)]

testDataInexFitRules0=np.where(testDataFitRules0==1)
testDataInexFitRules1=np.where(testDataFitRules1==1)

performanceMat_TestsetRule0=dataPerformanceTest[testDataInexFitRules0,:]
performanceMat_TestsetRule1=dataPerformanceTest[testDataInexFitRules1,:]


trainDataFitRules0=trainDataFitRules[str(0)]
trainDataFitRules1=trainDataFitRules[str(1)]

trainDataInexFitRules0=np.where(trainDataFitRules0==1)
trainDataInexFitRules1=np.where(trainDataFitRules1==1)

performanceMat_TrainsetRule0=dataPerformanceTrain[trainDataInexFitRules0,:]
performanceMat_TrainsetRule1=dataPerformanceTrain[trainDataInexFitRules1,:]


