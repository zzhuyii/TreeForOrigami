# This code use the sklearn package to apply a decission tree method onto the 
# gripper data base.

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from DecisionTree import TreeMethod
import numpy as np

# load the gripper performance data set
data1 = np.loadtxt('data1_MiuraSheet.txt',delimiter=',')
data2 = np.loadtxt('data1_TMPSheet.txt',delimiter=',')

fileTemp= open("data1_FeatureName.txt", "r")
tempfeatureName=fileTemp.readline()
tempfeatureName=tempfeatureName.split()
tempfeatureName=tempfeatureName[1:6]

# Separate the data and the label
dataFeature1 = (data1[:2000,1:6])
dataPerformance1 = (data1[:2000,6:12])
dataFeature2 = (data2[:2000,1:6])
dataPerformance2 = (data2[:2000,6:12])
dataFeature = np.concatenate((dataFeature1, dataFeature2), axis=0)
dataPerformance = np.concatenate((dataPerformance1, dataPerformance2), axis=0)

# Set up the feature database
# First we create the data matrix before using the OneHotEncoder
tempX = dataFeature
tempX[:,2]=tempX[:,2]*1000
tempX[:,3]=tempX[:,3]*1000
tempX[:,4]=tempX[:,4]*1000

categoryX=tempX[:,:2]
otherX=tempX[:,2:5]
categoryFeatureName=tempfeatureName[:2]
otherFeatureName=tempfeatureName[2:5]

# use the OneHotEncoder to transform the system
encoder = OneHotEncoder(sparse=False)
encodeX = encoder.fit_transform(categoryX)
encodeFeatureName=encoder.get_feature_names()

# Reconstruct the feature database
totalX=np.concatenate((encodeX,otherX),axis=1)
featureName=np.concatenate((encodeFeatureName,otherFeatureName),axis=0)

# we need another feature to determine the type of pattern
patternType=np.concatenate((np.zeros((2000,1)),np.ones((2000,1))),axis=0)
totalX=np.concatenate((patternType,totalX),axis=1)
featureName=np.concatenate((['pattern'],featureName),axis=0)

# Set up the label for the data base
totalY_target1=dataPerformance[:,1]<20000
totalY_target2=dataPerformance[:,5]<250000

# 90% bending stiffness as shifting target
# totalY_target_shift=(dataPerformance[:,2]<200)
# totalY_target_shift=(dataPerformance[:,2]>200)*(dataPerformance[:,2]<400)
# totalY_target_shift=(dataPerformance[:,2]>400)*(dataPerformance[:,2]<800)
totalY_target_shift=(dataPerformance[:,2]>800)*(dataPerformance[:,2]<1600)

totalY=totalY_target1*totalY_target2*totalY_target_shift
print(sum(totalY)) # The total number of data satisfying rule 2


relVSacc=np.zeros((10,2))
for q in range(10):
    
    # Randomly split the data into testing and training sets
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        totalX,totalY, test_size = 0.4, random_state=q)
    print('Positive Sample in Training')
    print(sum(Y_train))
    print('Positive Sample in Testing')
    print(sum(Y_test))
    
    # Create the tree method for extracting the design method
    tree=TreeMethod()
    
    # set the data
    tree.setData(X_train, Y_train, featureName)
    
    # set the design parameter
    tree.setParameter(alpha=0.00001, depth=24, num_tree=400)
    
    # train the random forest
    tree.train()
            
    # Compute the rules and print them
    tree.collectRule()
    tree.selectRule(thresholdReliability=0.9,
                       thresholdDataNum=10,
                       ruleNumber=5)
    tree.testRule(X_test,Y_test)
    tree.printRuleForClass()    
        
    reliability=tree.ruleTestReliability
    reliability=reliability[1,:]
    relVSacc[q,1]=np.mean(reliability)
    
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
    print(sum(Y_test==0), sum(Y_test==1), '\n')
    print('predicted data Num for two classes: ')
    print(sum(Y_pred==0), sum(Y_pred==1), '\n')
    print('Random Forest')
    print('the accuracy is: ')
    print(accurateRateRF)
    print('the perclass accuracy is: ')
    print(accurateRateRFClass1) 
    print(accurateRateRFClass2, '\n')

    relVSacc[q,0]=accurateRateRFClass2
