# This code use the sklearn package to apply a decission tree method onto the 
# gripper data base.

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed
from DecisionTree import TreeMethod
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


# Set up the feature database for easier reading
totalX[:,3]=totalX[:,3]*1000 # to mm
totalX[:,4]=totalX[:,4]*1000 # to mm
totalX[:,5]=totalX[:,5]*1000000 # to um
totalX[:,6]=totalX[:,6]*1000000 # to um
totalX[:,7]=totalX[:,7]*1000000 # to um
totalX[:,9]=totalX[:,9]*1000000 # to um


# Set up the label for the data base
totalY_target1=(dataPerformance[:,0]>10) # Frequency (Hz)
totalY_target2=(dataPerformance[:,1]<0.7) # input power (W)
totalY_target3=(dataPerformance[:,2]<500) # Maximum Temperature (degree C)
totalY_target4=(dataPerformance[:,3]>0.0016)# stiffness (N/m)
totalY=totalY_target1*totalY_target2*totalY_target3*totalY_target4
print('data number that meets target', sum(totalY))

# # Number of different random partition of hold out testing
# NumberOfRealization = 5

# Acc=np.zeros((NumberOfRealization,7,5))
# Precision=np.zeros((NumberOfRealization,7,5))
# Score=np.zeros((NumberOfRealization,7,5))

# for q in range(NumberOfRealization):

#     # Randomly split the data into testing and training sets
#     X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
#         totalX,totalY, test_size = 0.4, random_state=q+1000)
    
#     runs = [TreeMethod(i, j) for i in range(7) for j in range(5)]
    
#     with Parallel(n_jobs=8, verbose=8) as parallel:
#         delayed_funcs = [delayed(lambda x:x.runParellel( X_train, Y_train, featureName, X_test, Y_test))(run) for run in runs]
#         results = parallel(delayed_funcs)
        
#     for i in range (5*7):
#         tempV=results[i]
#         Acc[q,tempV[0],tempV[1]]=tempV[2]
#         Precision[q,tempV[0],tempV[1]]=tempV[3]
#         Score[q,tempV[0],tempV[1]]=tempV[4]


# meanAcc=np.mean(Acc,axis=0)
# meanPrecision=np.mean(Precision,axis=0)
# meanScore=np.mean(Score,axis=0)

# print('Random Forest')
# print('the accuracy is: ')
# print(meanAcc)
# print('the precision is: ')
# print(meanPrecision) 
# print('the F-score is: ')
# print(meanScore) 


# with the sklearn
skRF=RandomForestClassifier(n_estimators=100,min_samples_leaf=2,
                            min_samples_split=4,criterion="entropy",
                            class_weight='balanced')
skBoost=GradientBoostingClassifier(n_estimators=200,criterion='friedman_mse'
                                    ,min_samples_leaf=4,min_samples_split=8)
skKNN=KNeighborsClassifier()

preRF=np.zeros(5)
preBoost=np.zeros(5)
preKnn=np.zeros(5)

for q in range(5):

    # Randomly split the data into testing and training sets
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        totalX,totalY, test_size = 0.4, random_state=q)    

    # Use the trandom forest to predict the results of the testing set
    skRF.fit(X_train, Y_train)
    skBoost.fit(X_train, Y_train)
    skKNN.fit(X_train, Y_train)
    
    sk_Y_pred=skRF.predict(X_test)
    skB_Y_pred=skBoost.predict(X_test)
    skKNN_Y_pred=skKNN.predict(X_test)

    # calculate the accruacy of other method
    # accuray of RF from SK learn
    accuratePredict = sum(sk_Y_pred==Y_test)     
    accurateClass2 = sum((sk_Y_pred==Y_test) * (sk_Y_pred==1))
    preRF[q] = accurateClass2 / sum(sk_Y_pred==1)
    
    # accuray of boosting from SK learn
    accuratePredict = sum(skB_Y_pred==Y_test)    
    accurateClass2 = sum((skB_Y_pred==Y_test) * (skB_Y_pred==1))
    preBoost[q] = accurateClass2 / sum(skB_Y_pred==1)


    # accuray of knn from SK learn
    accuratePredict = sum(skKNN_Y_pred==Y_test)   
    accurateClass2 = sum((skKNN_Y_pred==Y_test) * (skKNN_Y_pred==1))
    preKnn[q] = accurateClass2 / sum(skKNN_Y_pred==1)


print('sklearn random forest')
print('the precision is: ')
print(np.mean(preRF), '\n')

print('sklearn Boosting')
print('the precision is: ') 
print(np.mean(preBoost), '\n')

print('sklearn KNN')
print('the precision is: ')
print(np.mean(preKnn), '\n')

