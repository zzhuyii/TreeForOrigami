import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from DecisionTree import TreeMethod
from timeit import default_timer as timer


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
        tree.computeRule(X_select, Y_select, ruleNumber=1 ,minData=3)
        end = timer()
        timeTraining[i,j] = (end - start)
        
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
        testPrecision[i,j] = accurateClass1 / sum(Y_pred==1)
