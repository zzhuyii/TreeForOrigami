import numpy as np
import streamlit as st
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from DecisionTree import TreeMethod

def RunTreeDesign(BS30LB,BS30UB,BS60LB,BS60UB,BS90LB,BS90UB):    

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
    
    #categoryFeatureName=tempfeatureName[:2]
    #otherFeatureName=tempfeatureName[2:5]
    
    # use the OneHotEncoder to transform the system
    encoder = OneHotEncoder(sparse_output=False)
    encodeX = encoder.fit_transform(categoryX)
    
    #encodeFeatureName=encoder.get_feature_names()
    
    # Convert the names back to the normal names
    #for i in range(6):
        #encodeFeatureName[i]=encodeFeatureName[i].replace('x0_','m=')
        #encodeFeatureName[i]=encodeFeatureName[i].replace('x1_','n=')
    
    # Reconstruct the feature database
    totalX=np.concatenate([encodeX,otherX],axis=1)
    
    #featureName=np.concatenate((encodeFeatureName,otherFeatureName),axis=0)
    
    # we need another feature to determine the type of pattern
    patternType=np.concatenate((np.zeros((2000,1)),np.ones((2000,1))),axis=0)
    totalX=np.concatenate((patternType,totalX),axis=1)
    
    
    featureName=['pattern','m=24','m=30','m=36','n=6','n=9','n=12','tc','tp','W']

    
    totalY=(dataPerformance[:,0]>BS30LB)*(dataPerformance[:,0]<BS30UB)*(dataPerformance[:,1]>BS60LB)*(dataPerformance[:,1]<BS60UB)*(dataPerformance[:,2]>BS90LB)*(dataPerformance[:,2]<BS90UB)
    
    st.text('data number that meets target')
    st.text(sum(totalY))
    
    # Randomly split the data into testing and training sets
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        totalX,totalY, test_size = 0.4, random_state=0)
    
    # Apply the proposed method for computing the design rules
    tree=TreeMethod()
    tree.setParameter(alpha=0.0001, depth=20, num_tree=100)
    tree.train(X_train, Y_train, featureName)
    
    success = tree.computeRule(X_train, Y_train, ruleNumber=1)
    
    tree.testRule(X_test,Y_test)
    tree.printRule()
    
    # Plot the rules for easy interpretation
    featureMin=[0,0,0,0,0,0,0,0.5,1.0,1.0] # minimum value for feature
    featureMax=[1,1,1,1,1,1,1,1.0,6.0,4.0] # maximum value for feature
    fig=tree.plotRule(featureMin,featureMax)
    
    return success,fig
    
    # # Predict the results of the testing set using the embeded Random Forest in
    # # the proposed method
    # Y_pred=tree.predict(X_test)
    
    # # calculate the accruacy of the prediction
    # accuratePredict = sum(Y_pred==Y_test)
    # accurateRateRF = accuratePredict / len(Y_pred)
    
    # # Precision of the prediction
    # accurateClass1 = sum((Y_pred==Y_test) * (Y_pred==1))
    # Precision = accurateClass1 / sum(Y_pred==1)
    
    # print('testing data Num for two classes: ')
    # print(sum(Y_pred==0), sum(Y_pred==1), '\n')
    
    # print('Embeded Random Forest')
    # print('accuracy is: ', accurateRateRF)
    # print('precision is: ', Precision, '\n') 
    
    


st.subheader("Inverse Design of Origami Metamaterial Sheet")

st.text('Developer: Dr. Yi Zhu')

st.text('This is a demo for using the Sim-FAST package to simulate the deployment ' + 
        'and load carrying capacity of kirigami truss bridges. We assume that ' +
        'connections are rigid, all members share the same cross-section, and ' +
        'ignore buckling related failure mode when calculating the loading.')

st.subheader("Inverse Design Targets")

BS30LB = st.selectbox(
     "Select lower bond of stiffness at 30% extension (N/m):",
     [10000,100000,400000])

BS30UB = st.selectbox(
     "Select upper bond of stiffness at 30% extension (N/m):",
     [400000,800000,1600000,3200000])

BS60LB = st.selectbox(
     "Select lower bond of stiffness at 60% extension (N/m):",
     [1000,5000,10000,50000,100000])

BS60UB = st.selectbox(
     "Select upper bond of stiffness at 60% extension (N/m):",
     [100000,500000,1000000])

BS90LB = st.selectbox(
     "Select lower bond of stiffness at 90% extension (N/m):",
     [5000,10000,40000])

BS90UB = st.selectbox(
     "Select upper bond of stiffness at 90% extension (N/m):",
     [40000,80000,160000,320000])


success,fig=RunTreeDesign(BS30LB,BS30UB,BS60LB,BS60UB,BS90LB,BS90UB)


st.subheader("Inverse Design Results")


if success==True:
    st.pyplot(fig)
else:
    st.text('Seems that we cannot find a feasible solution. ' + 
        'Please consider use a less strick search target, or ' +
        'add additional data to the data set.' )    