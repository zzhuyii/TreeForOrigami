import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed
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
# Target 2
totalY_target1=(dataPerformance[:,0]>800) # Stiffness in Z (N/3m)
totalY_target2=dataPerformance[:,2]>600 # Stiffness in X (N/m)
totalY_target3=dataPerformance[:,3]<0.1 # error 
totalY=totalY_target1*totalY_target2*totalY_target3
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
