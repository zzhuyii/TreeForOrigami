import numpy as np
from sklearn import model_selection
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


# Target 3
totalY_target1=(dataPerformance[:,0]>300) # Stiffness in Z (N/m)
totalY_target2=dataPerformance[:,2]>150 # Stiffness in X (N/m)
totalY_target3=dataPerformance[:,3]<0.06 # error 
totalY=totalY_target1*totalY_target2*totalY_target3
print('data number that meets target', sum(totalY))


# Randomly split the data into testing and training sets
# we use a common hold hout setup with 0.4 test size
# Use different random seed to check the stability
for i in range(6):
    
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        totalX,totalY, test_size = 0.4, random_state= i)

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

