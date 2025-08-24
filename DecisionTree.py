# This code is based on the sklearn package
# Apart from implementing the standard Random Forest Classifier based on the 
# Decision Tree from sklearn, the code further unfold the trees to compute the
# Design rules of the target. 

from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# The class is called TreeMethod
class TreeMethod():
    
    # The following methods are used for parallely solving the precision for
    # Grid search of the hyperparameters
    def __init__(self, i=0,j=0):
        self.i=i
        self.j=j
        self.dep=[8,12,16,20,24,28,32]
        self.treeNum=[20,40,60,100,200]
        
    def runParellel(self,  X_train, Y_train, featureName, X_test, Y_test):
        
        i=self.i
        j=self.j
        
        self.setParameter(alpha=0.0001, depth=self.dep[i], num_tree=self.treeNum[j])
        self.train(X_train, Y_train, featureName)
        self.computeRule(X_train, Y_train, ruleNumber=1)
        self.testRule(X_test,Y_test)
        
        # tree.printRule()  
        Pre = (self.finalRuleTestPrecision[0])
        Score = (self.finalRuleScore[0])
        
        # Use the trandom forest to predict the results of the testing set
        Y_pred=self.predict(X_test)
        
        # calculate the accruacy
        accuratePredict = sum(Y_pred==Y_test)            
        Acc = accuratePredict / len(Y_pred)

        return [i,j, Acc, Pre, Score]
 
    
    
    ###########################################################################
    # The following method set the parameters for the decision trees
    ###########################################################################     
    
    def setParameter(self,alpha=0.0001,depth=20,num_tree=100):
        self.alpha=alpha
        self.depth=depth
        self.num_tree=num_tree      
        self.treeList=[]
        for i in range(self.num_tree):
            if alpha==0:
                 self.treeList.append(DecisionTreeClassifier(
                                 max_depth=depth,
                                 min_samples_leaf=2,
                                 min_samples_split=4,
                                 random_state=i,
                                 criterion="entropy",
                                 splitter='best',
                                 class_weight='balanced'))
            else: 
                self.treeList.append(DecisionTreeClassifier(
                                 ccp_alpha=alpha,
                                 max_depth=depth,
                                 min_samples_leaf=2,
                                 min_samples_split=4,
                                 random_state=i,
                                 criterion="entropy",
                                 splitter='best',
                                 class_weight='balanced'))
                
            # Here the random state is fixed for demonstration purpose. 
            # In real practice the random state need not be fixed.            
            
    ###########################################################################
    # The following method train the decision trees
    ########################################################################### 
        
    def train(self,X_train,Y_train,featureName):        
      
        self.featureNum=len(X_train[0,:])
        self.featureName=featureName
        
        # Here we need to use the random sub data set for different trees
        for i in range(self.num_tree):
            X_subtrain, X_remain, Y_subtrain, Y_remain = model_selection.train_test_split(
                        X_train, Y_train, test_size = 0.5, random_state=i)
            
            self.treeList[i].fit(X_subtrain, Y_subtrain)
            
            # Here we use random subsets to encourage generating different 
            # trees, which helps to promote the performance of the algorithm
    
    ###########################################################################
    # The following method will gives the predict label class
    ###########################################################################              
        
    def predict(self,X_test):        
        # Build a matrix to store the prediciton of each tree
        Y_pred_mat = np.zeros((self.num_tree,len(X_test)))   
        
        # obtain the prediction of different trees
        for i in range(self.num_tree):            
            Y_pred_mat[i,:]=(self.treeList[i]).predict(X_test)
            
        # The final prediction is the mode of different trees
        Y_pred=stats.mode(Y_pred_mat)
        return np.squeeze(Y_pred.mode)


    ###########################################################################
    # The following method compute the rules for the target design. We first
    # compute the precision, recall, and data size associated with each rules.
    # Then we eliminate bad rules that does not meet the min data requirement 
    # and min precision requirement. Finally, we select the rules based on the 
    # F-scroe.
    ###########################################################################      
            
    def computeRule(self,
                   X_train,
                   Y_train,
                   ruleNumber=1,
                   selectedClass=1,
                   minData=10,
                   minPrecision=0.9,
                   beta=0.2):
        
        # First run the analysis to collect all rules
        self.collectRule()
        
        # How many rules we want to select
        self.ruleNumber=ruleNumber        
       
        tempRule=self.collectedRule
        ruleNum=np.size(tempRule,axis=0) 
        
        # We are interested in the precision and recall of the rules
        # in addition to the recall, knowing how many data fits the rule 
        # discription is also helpful so we also track it.
        self.ruleTrainPrecision=np.zeros((ruleNum))
        self.ruleTrainRecall=np.zeros((ruleNum))
        self.ruleTrainSize=np.zeros((ruleNum))
        
        
        for k in range(ruleNum):
            trainDataFitRules=np.ones(np.size(Y_train))                          
            for p in range(self.featureNum):
                if tempRule[k,p,0]!=0:
                    trainDataFitRules=np.multiply(trainDataFitRules,
                        (X_train[:,p]<tempRule[k,p,0]))
                if tempRule[k,p,1]!=0:
                    trainDataFitRules=np.multiply(trainDataFitRules,
                        (X_train[:,p]>tempRule[k,p,1]))
        
            # print(testDataFitRules)
            if sum(trainDataFitRules)!=0:
                TP=np.multiply (trainDataFitRules , (selectedClass==Y_train))                        
                self.ruleTrainPrecision[k]=sum(TP)/sum(trainDataFitRules)
                self.ruleTrainRecall[k]=sum(TP)/sum((selectedClass==Y_train))
                self.ruleTrainSize[k]=sum(trainDataFitRules)
                
        # AFter we have computed the precision, recall, and data size   
        # next step is to eliminate bad rules by deleting the rules
        # that does not meet the minimum threshold
        
        self.selectedRule={}        
        
        checkVec1=(self.ruleTrainSize>=minData)
        checkVec2=(self.ruleTrainPrecision>=minPrecision)
        
        checkVec=checkVec1*checkVec2
        
        deleteIndex=[]
        for q in range(len(checkVec)):
            if checkVec[q]==0:
                deleteIndex.append(q)
                
        tempRule=np.delete(tempRule, deleteIndex, axis=0)
        
        self.ruleTrainSize=np.delete(self.ruleTrainSize,deleteIndex,axis=0)
        self.ruleTrainPrecision=np.delete(self.ruleTrainPrecision,deleteIndex,axis=0)
        self.ruleTrainRecall=np.delete(self.ruleTrainRecall,deleteIndex,axis=0)        
       
        self.selectedRule=tempRule
        
        if np.linalg.norm(tempRule)==0:
            print('Cannot find available rules. Please relax cretarion.')
                            
        # Next we select the rules based on their score             
        self.finalRule={}
               
        # Here we compute the F-score of rules
        self.ruleScore=(1+beta*beta)*self.ruleTrainPrecision*self.ruleTrainRecall/(self.ruleTrainPrecision*beta*beta+self.ruleTrainRecall)   
        
        # Here we rank the rules based on F-scores
        sequenceCount=np.argsort(-self.ruleScore)     
        
        sizeMat=np.shape(tempRule)            
        self.finalRule=np.zeros((ruleNumber,sizeMat[1],2))
        self.finalRuleTrainSize=np.zeros(ruleNumber)
        self.finalRuleTrainPrecision=np.zeros(ruleNumber)
        self.finalRuleTrainRecall=np.zeros(ruleNumber)
        self.finalRuleScore=np.zeros(ruleNumber)
        
        for j in range(ruleNumber):
            self.finalRule[j,:,:]=tempRule[sequenceCount[j],:,:]
            self.finalRuleTrainSize[j]=self.ruleTrainSize[sequenceCount[j]]
            self.finalRuleTrainPrecision[j]=self.ruleTrainPrecision[sequenceCount[j]]
            self.finalRuleTrainRecall[j]=self.ruleTrainRecall[sequenceCount[j]]
            self.finalRuleScore[j]=self.ruleScore[sequenceCount[j]]
        
    
    
    ###########################################################################
    # The following method will print the rules ((in text)
    ###########################################################################                
    
    def printRule(self):  
        
        for i in range(self.ruleNumber):
            print('training precision', 
                  self.finalRuleTrainPrecision[i])
            print('training data size', 
                  self.finalRuleTrainSize[i])
            print('testing precision' ,
                  self.finalRuleTestPrecision[i])
            print('testing data size', 
                  self.finalRuleTestSize[i])

            tempString = ''   
                    
            for j in range(self.featureNum):
                
                if self.finalRule[i,j,0]!=0:
                    tempString+= str(self.featureName[j])
                    tempString+='<'
                    tempString+=str(self.finalRule[i,j,0])
                    tempString+='\n'
                if self.finalRule[i,j,1]!=0:
                    tempString+= str(self.featureName[j])
                    tempString+='>='
                    tempString+=str(self.finalRule[i,j,1])
                    tempString+='\n'   
                    
            print(tempString)  
            
            
    ###########################################################################
    # The following set of codes will plot the rules using figures
    ###########################################################################  
         
    def plotRule(self,featureMin,featureMax):  
                      
        
        for i in range(self.ruleNumber):
            
            fig, ax=plt.subplots(self.featureNum,1,figsize=[5,8])   
            fig.dpi=300
                    
            for j in range(self.featureNum):
                
                selectLowThreshold=featureMin[j]
                if self.finalRule[i,j,1]!=0:
                    selectLowThreshold=self.finalRule[i,j,1]
                
                selectHighThreshold=featureMax[j]
                if self.finalRule[i,j,0]!=0:
                    selectHighThreshold=self.finalRule[i,j,0]
                
                stats=[{'med': None, 
                        'q1': selectLowThreshold, 
                        'q3': selectHighThreshold, 
                        'whislo': featureMin[j],
                        'whishi': featureMax[j], 
                        'label': str(self.featureName[j]) }]
            
                ax[j].bxp(stats,showfliers=False,
                       showmeans=False,
                       vert=False,
                       widths=0.6)  
                
            fig.tight_layout()
            fig.show()
            return fig
            

            
    ###########################################################################
    # The following method will find the list of data that fit rules
    ###########################################################################      
            
    def findDataFitFinalRule(self,X_train,Y_train,X_test,Y_test):
        
        self.dataFitFinalRule_TrainData={}
        self.dataFitFinalRule_TestData={}
        
        for j in range(self.ruleNumber):
                    
            trainDataFitRules=np.ones(np.size(Y_train))
            for p in range(self.featureNum):
                if self.finalRule[j,p,0]!=0:
                    trainDataFitRules=np.multiply(trainDataFitRules,
                        (X_train[:,p]<self.finalRule[j,p,0]))
                if self.finalRule[j,p,1]!=0:
                    trainDataFitRules=np.multiply(trainDataFitRules,
                        (X_train[:,p]>self.finalRule[j,p,1]))
                        
            self.dataFitFinalRule_TrainData[str(j)]=trainDataFitRules             
                    
            testDataFitRules=np.ones(np.size(Y_test))
            for p in range(self.featureNum):
                if self.finalRule[j,p,0]!=0:
                    testDataFitRules=np.multiply(testDataFitRules,
                        (X_test[:,p]<self.finalRule[j,p,0]))
                if self.finalRule[j,p,1]!=0:
                    testDataFitRules=np.multiply(testDataFitRules,
                        (X_test[:,p]>self.finalRule[j,p,1]))
            
            self.dataFitFinalRule_TestData[str(j)]=testDataFitRules

        
    
    ###########################################################################
    # The following method will perform testing of the rules. It computes 
    # the precision of the rule for both training and testing data set and 
    # check how many data are associated with a rule set. 
    ###########################################################################      
            
    def testRule(self,X_test,Y_test,selectedClass=1):
        # Compute the rule precision and robostness
        self.finalRuleTestPrecision=np.zeros(self.ruleNumber)
        self.finalRuleTestRecall=np.zeros(self.ruleNumber)        
        self.finalRuleTestSize=np.zeros(self.ruleNumber)
        
        # We also recompute the training precision using the entire training 
        # data rather than using the 10 subsets                        
        tempRule=self.finalRule
        
        for j in range(self.ruleNumber):
                    
            testDataFitRules=np.ones(np.size(Y_test))
            for p in range(self.featureNum):
                if tempRule[j,p,0]!=0:
                    #print(testDataFitRules)
                    testDataFitRules=np.multiply(testDataFitRules,
                        (X_test[:,p]<tempRule[j,p,0]))
                    #print(testDataFitRules)
                if tempRule[j,p,1]!=0:
                    testDataFitRules=np.multiply(testDataFitRules,
                        (X_test[:,p]>tempRule[j,p,1]))
            
            # print(testDataFitRules)
            if sum(testDataFitRules)!=0:
                TP=np.multiply (testDataFitRules , (selectedClass==Y_test))                        
                self.finalRuleTestPrecision[j]=sum(TP)/sum(testDataFitRules)                
                self.finalRuleTestRecall[j]=sum(TP)/sum(selectedClass==Y_test)
                self.finalRuleTestSize[j]=sum(testDataFitRules)


    
    ###########################################################################
    # The following method will use the unfoldTrees() to collect rules for
    # selected classes. Those zero arrays will be rmoved
    ###########################################################################
   
    def collectRule(self,selectedClass=1):
           
        # In the following code we generate a more structured rule data
        # by removing those arrays that are fully zeros
        tempRule=self.unfoldTrees(selectedClass)    
        self.collectedRule={}
        ruleShape=np.shape(tempRule)
        tempRule=np.reshape(tempRule, (ruleShape[0]*ruleShape[1], self.featureNum ,2 ))
        
        checkNull=np.reshape(tempRule, (ruleShape[0]*ruleShape[1], self.featureNum*2 ))
        checkNull=np.sum(checkNull**2,axis=1)
        checkNull=np.sign(checkNull)
        
        deleteIndex=[]
        for q in range(len(checkNull)):
            if checkNull[q]==0:
                deleteIndex.append(q)
        tempRule=np.delete(tempRule, deleteIndex, axis=0)
        
        self.collectedRule=tempRule
            
            
    
    ###########################################################################
    # The following set of codes will unfold trees for selected class
    ###########################################################################   
    
    def unfoldTrees(self,selectedClass):
        # This code unfold the decision trees for a selected calss
               
        self.max_leafNum=0
        for k in range(self.num_tree):
            if self.treeList[k].get_n_leaves()>self.max_leafNum:
                self.max_leafNum=self.treeList[k].get_n_leaves()
                
        collectedRule=np.zeros((self.num_tree,self.max_leafNum,
                                   self.featureNum,2))
        
        
        for k in range(self.num_tree):
            # code for reconstructing the tree
            treeValue=self.treeList[k].tree_.value
            childrenLeft=self.treeList[k].tree_.children_left
            childrenRight=self.treeList[k].tree_.children_right
            
            leafNumber=self.treeList[k].get_n_leaves()
            # print(leafNumber)
            
            leafIndex=(np.where(childrenLeft==-1))[0]
            # print(leafIndex)
            
            children=np.stack((childrenLeft,childrenRight),axis=-1)
            
            
            for i in range(leafNumber):
    
                # the node index of current leaf node
                tempIndex=leafIndex[i]
                # the class precition of the leaf node
                classIndex=np.argmax(treeValue[tempIndex])
                # print(classIndex)
                
                tempRule=np.zeros((self.featureNum,2))

                if classIndex==selectedClass:             
                    for j in range(self.treeList[k].get_depth()):
                        if tempIndex==0:
                            break
                        else:
                            # first find the parent node
                            parentInfo=(np.where(children==tempIndex))
                            parentNode=int(parentInfo[0])
                            # figure out if the parent node is left or right
                            # sklearn use smaller than for going left
                            leftOrRight=int(parentInfo[1])
                                  
                            # figure out the selected feature and threshold               
                            featureSelected=int(self.treeList[k].tree_.feature[parentNode])
                            threshold=self.treeList[k].tree_.threshold[parentNode]    
                            
                            # print out the reslts if needed
                            # print(parentNode,leftOrRight,featureSelected,threshold)
                            
                            if leftOrRight==0:
                                if tempRule[featureSelected,leftOrRight]!=0:
                                    if tempRule[featureSelected,leftOrRight]>threshold:
                                        tempRule[featureSelected,leftOrRight]=threshold
                                else:
                                    tempRule[featureSelected,leftOrRight]=threshold
                            else:
                                if tempRule[featureSelected,leftOrRight]!=0:
                                    if tempRule[featureSelected,leftOrRight]<threshold:
                                        tempRule[featureSelected,leftOrRight]=threshold
                                else:
                                    tempRule[featureSelected,leftOrRight]=threshold
                            tempIndex=parentNode 
                            
                            collectedRule[k,i,:,:]=tempRule                    
        return collectedRule
    
    