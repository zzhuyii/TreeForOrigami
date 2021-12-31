# This code use the sklearn package to apply a decission tree method onto the 
# gripper data base.

from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder

from DecisionTree import TreeMethod
# from DecisionTree import TreeMethod
import numpy as np

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

# Set up the feature database for easier reading
totalX[:,3]=totalX[:,3]*1000 # to mm
totalX[:,4]=totalX[:,4]*1000 # to mm
totalX[:,5]=totalX[:,5]*1000000 # to um
totalX[:,6]=totalX[:,6]*1000000 # to um
totalX[:,7]=totalX[:,7]*1000000 # to um
totalX[:,9]=totalX[:,9]*1000000 # to um

# Set up the label for the data base
# Target 1
totalY_target1=(dataPerformance[:,0]>20)*(dataPerformance[:,0]<40) # frequency (Hz)
totalY_target2=dataPerformance[:,1]<0.2 # input power (W)
totalY_target3=dataPerformance[:,2]<250 # maximum temperature (degree C)
totalY=totalY_target1*totalY_target2*totalY_target3
robustThreshold=20;
print('data number that meets target', sum(totalY))

# Target 2
# totalY_target1=(dataPerformance[:,0]>35)*(dataPerformance[:,0]<50) # frequency (Hz)
# totalY_target2=(dataPerformance[:,1]>0.3) # input power (W)
# totalY=totalY_target1*totalY_target2
# robustThreshold=10;
# print('data number that meets target', sum(totalY))


# Target 3
# totalY_target1=(dataPerformance[:,0]>10) # Frequency (Hz)
# totalY_target2=(dataPerformance[:,1]<0.7) # input power (W)
# totalY_target3=(dataPerformance[:,2]<500) # Maximum Temperature (degree C)
# totalY_target4=(dataPerformance[:,3]>0.004)# stiffness (N/m)
# totalY=totalY_target1*totalY_target2*totalY_target3*totalY_target4
# robustThreshold=10;
# print('data number that meets target', sum(totalY))


# Randomly split the data into testing and training sets
numdata=10
relVSacc=np.zeros((numdata,2))
for q in range(numdata):

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        totalX,totalY, test_size = 0.4, random_state=q+np.random.randint(1000))
    
    # Create the tree method for extracting the design method 
    tree=TreeMethod()
    
    # set the data
    tree.setData(X_train, Y_train, featureName)
    
    # set the design parameter
    tree.setParameter(alpha=0.00001, depth=28, num_tree=100)
    
    # train the random forest
    tree.train()
            
    # Compute the rules and print them
    tree.collectRule()
    tree.selectRule(thresholdReliability=0.9,
                       thresholdDataNum=robustThreshold,
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
    print(sum(Y_pred==0), sum(Y_pred==1), '\n')
    print('Random Forest')
    print('the accuracy is: ')
    print(accurateRateRF)
    print('the perclass accuracy is: ')
    print(accurateRateRFClass1) 
    print(accurateRateRFClass2, '\n')
    
    relVSacc[q,0]=accurateRateRFClass2
    


