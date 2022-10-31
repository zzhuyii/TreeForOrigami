import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from DecisionTree import TreeMethod
from timeit import default_timer as timer

#Studying the influence of size of data

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

# Set up the label for the data base
# We use this target to study the effect of data size
totalY=(dataPerformance[:,4]>80000)
print('data number that meets target', sum(totalY))

# Study the size of data set
datasetSizeRate=[0.1, 0.2, 0.4, 0.6, 0.8]

# testing precision storage
testPrecision = np.zeros([10,6])
timeTraining = np.zeros([10,6])
dataSize = np.zeros(6)

# Set up the study
for i in range(10):    
    
    # Randomly split the data into testing and training sets
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    totalX, totalY, test_size = 0.4, random_state=i)
    
    for j in range(6):

        if j != 5:            
            # Randomly select data for training
            X_select, X_rest, Y_select, Y_rest = model_selection.train_test_split(
            X_train, Y_train, test_size = 1-datasetSizeRate[j], random_state=j)
        else:
            X_select=X_train
            Y_select=Y_train        

        dataSize[j] = len(X_select)


        # Apply the proposed method for computing the design rules
        tree=TreeMethod()
        tree.setParameter(alpha=0.0001, depth=20, num_tree=100)
        
        # Also record the time for training trees and finding the rules    
        start = timer()
        tree.train(X_select, Y_select, featureName)          
        tree.computeRule(X_select, Y_select, ruleNumber=1)
        end = timer()
        timeTraining[i,j] = (end - start)
        
        tree.testRule(X_test,Y_test)
        tree.printRule()
        
        # Plot the rules for easy interpretation
        featureMin=[0,0,0,0,0,0,0,0.5,1.0,1.0] # minimum value for feature
        featureMax=[1,1,1,1,1,1,1,1.0,6.0,4.0] # maximum value for feature
        tree.plotRule(featureMin,featureMax)
                
        # Predict the results of the testing set using the embeded Random Forest in
        # the proposed method
        Y_pred=tree.predict(X_test)
        
        # calculate the accruacy of the prediction
        accuratePredict = sum(Y_pred==Y_test)
        accurateRateRF = accuratePredict / len(Y_pred)
        
        # Precision of the prediction
        accurateClass1 = sum((Y_pred==Y_test) * (Y_pred==1))
        testPrecision[i,j] = accurateClass1 / sum(Y_pred==1)
 
        
 


