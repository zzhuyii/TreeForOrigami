# This code use the sklearn package to apply a decission tree method onto the 
# gripper data base.

from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import numpy as np

# define the class for achieving the tree method
class TreeMethod():
    
    
    
    ###########################################################################
    # The following method input the data
    ########################################################################### 
    
    def setData(self,X_train, Y_train, featureName):
        self.X_train=X_train
        self.Y_train=Y_train
        self.featureNum=len(X_train[0,:])
        self.classNum=int(max(Y_train))+1
        self.featureName=featureName
        
        
        
        
    ###########################################################################
    # The following method set the parameters for the tree method
    ###########################################################################     
        
    def setParameter(self,alpha=0.0001,depth=8,num_tree=10):
        self.alpha=alpha
        self.depth=depth
        self.num_tree=num_tree      
        self.treeList=[]
        for i in range(self.num_tree):
            if alpha==0:
                 self.treeList.append(DecisionTreeClassifier(
                                 max_depth=depth,
                                 max_features=self.featureNum,
                                 random_state=i,
                                 criterion="gini",
                                 splitter='best',
                                 class_weight='balanced'))
            else: 
                self.treeList.append(DecisionTreeClassifier(
                                 ccp_alpha=alpha,
                                 max_depth=depth,
                                 max_features=self.featureNum,
                                 random_state=i,
                                 criterion="gini",
                                 splitter='best',
                                 class_weight='balanced'))
            # Here the random state is fixed for demonstration purpose. 
            # In real practice the random state need not be fixed.
            
            
            
            
    ###########################################################################
    # The following method train the tree method
    ########################################################################### 
        
    def train(self):
        # Here we need to use the random sub data set for different trees
        for i in range(self.num_tree):
            X_subtrain, X_remain, Y_subtrain, Y_remain = model_selection.train_test_split(
                        self.X_train, self.Y_train, test_size = 0.3,random_state=i)
            self.treeList[i].fit(X_subtrain, Y_subtrain)
            
        
    
    
    
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
    # The following method will select rules. The rules selected should be 
    # statistically significant meaning that the branch should contain more 
    # data points than the specified threshold.
    ###########################################################################                      
    
    
    def printRuleForClass(self,selectedClass):
        
        dicIndex=str(selectedClass)            
        tempRule=self.finalRule[dicIndex]
        
        for i in range(self.ruleNumber):

            print('training reliability', 
                  self.ruleTrainReliability[selectedClass,i])
            print('training data size', 
                  self.ruleTrainSize[selectedClass,i])
            print('testing reliability' ,
                  self.ruleTestReliability[selectedClass,i])
            print('testing data size', 
                  self.ruleTestSize[selectedClass,i])

            tempString = ''   
                    
            for j in range(self.featureNum):
                
                if tempRule[i,j,0]!=0:
                    tempString+= str(self.featureName[j])
                    tempString+='<'
                    tempString+=str(tempRule[i,j,0])
                    tempString+='\n'
                if tempRule[i,j,1]!=0:
                    tempString+= str(self.featureName[j])
                    tempString+='>='
                    tempString+=str(tempRule[i,j,1])
                    tempString+='\n'   
                    
            print(tempString)  

        
    
    ###########################################################################
    # The following method will perform testing of the rules. It computes 
    # the reliability of the rule for both training and testing data set and 
    # check how many data are associated with a rule set. 
    ###########################################################################      
            
    def testRule(self,X_test,Y_test):
        # select rules based on reliability
        self.ruleTrainReliability=np.zeros((self.classNum,self.ruleNumber))
        self.ruleTestReliability=np.zeros((self.classNum,self.ruleNumber))
        self.ruleTrainSize=np.zeros((self.classNum,self.ruleNumber))
        self.ruleTestSize=np.zeros((self.classNum,self.ruleNumber))
        
        for i in range(self.classNum):
            dicIndex=str(i)            
            tempRule=self.finalRule[dicIndex]
            
            for j in range(self.ruleNumber):
                        
                testDataFitRules=np.ones(np.size(self.Y_train))
                for p in range(self.featureNum):
                    if tempRule[j,p,0]!=0:
                        #print(testDataFitRules)
                        testDataFitRules=np.multiply(testDataFitRules,
                            (self.X_train[:,p]<tempRule[j,p,0]))
                        #print(testDataFitRules)
                    if tempRule[j,p,1]!=0:
                        testDataFitRules=np.multiply(testDataFitRules,
                            (self.X_train[:,p]>tempRule[j,p,1]))
                
                # print(testDataFitRules)
                if sum(testDataFitRules)!=0:
                    tempY=np.multiply (testDataFitRules , (i==self.Y_train))                        
                    self.ruleTrainReliability[i,j]=sum(tempY)/sum(testDataFitRules)
                    self.ruleTrainSize[i,j]=sum(testDataFitRules)
                    # print(sum(tempY)/sum(testDataFitRules))
                        
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
                    tempY=np.multiply (testDataFitRules , (i==Y_test))                        
                    self.ruleTestReliability[i,j]=sum(tempY)/sum(testDataFitRules)
                    self.ruleTestSize[i,j]=sum(testDataFitRules)
                    # print(sum(tempY)/sum(testDataFitRules))



    ###########################################################################
    # The following method will perform selection of the rules. It computes 
    # the reliability of the rule using the training data, and select the rules 
    # that are statistically significant
    ###########################################################################      
            
    def selectRule(self,thresholdReliability=0.9,
                   thresholdDataNum=20, 
                   ruleNumber=3):
        
        self.ruleNumber=ruleNumber
        
        # select rules based on reliability
        self.ruleTrainReliability={}
        self.ruleTrainSize={}
        
        for i in range(self.classNum):
            dicIndex=str(i)
            tempRule=self.collectedRule[dicIndex]
            ruleNum=np.size(tempRule,axis=0) 
            ruleTrainReliability=np.zeros(ruleNum)
            ruleTrainSize=np.zeros(ruleNum)
            
            for k in range(ruleNum):
                testDataFitRules=np.ones(np.size(self.Y_train))
                for p in range(self.featureNum):
                    if tempRule[k,p,0]!=0:
                        #print(testDataFitRules)
                        testDataFitRules=np.multiply(testDataFitRules,
                            (self.X_train[:,p]<tempRule[k,p,0]))
                        #print(testDataFitRules)
                    if tempRule[k,p,1]!=0:
                        testDataFitRules=np.multiply(testDataFitRules,
                            (self.X_train[:,p]>tempRule[k,p,1]))
                
                # print(testDataFitRules)
                if sum(testDataFitRules)!=0:
                    tempY=np.multiply (testDataFitRules , (i==self.Y_train))                        
                    ruleTrainReliability[k]=sum(tempY)/sum(testDataFitRules)
                    ruleTrainSize[k]=sum(testDataFitRules)
                    # print(sum(tempY)/sum(testDataFitRules))
                    
                self.ruleTrainReliability[dicIndex]=ruleTrainReliability
                self.ruleTrainSize[dicIndex]=ruleTrainSize      
                
        # Here we have computed the size of the rules and the reliability of 
        # them the next step is to further select them
        
        self.selectedRule={}
        for i in range(self.classNum):
            dicIndex=str(i)
            
            tempRule=self.collectedRule[dicIndex]
               
            ruleTrainReliability=self.ruleTrainReliability[dicIndex]  
            ruleTrainSize=self.ruleTrainSize[dicIndex]
            
            checkVec1=(ruleTrainSize>=thresholdDataNum)
            checkVec2=(ruleTrainReliability>=thresholdReliability)
            
            checkVec=checkVec1*checkVec2
            
            deleteIndex=[]
            for q in range(len(checkVec)):
                if checkVec[q]==0:
                    deleteIndex.append(q)
                    
            tempRule=np.delete(tempRule, deleteIndex, axis=0)
            self.selectedRule[dicIndex]=tempRule
            
        # The next step is to find the unique sparsity patterns and find out 
        # the most common sparsity patterns. After selecting the most common 
        # patterns we can compute the mean and finally select the rules
        self.uniqueSparsePattern={}
        self.uniqueSparsePatternCount={}
        
        self.finalRule={}
        for i in range(self.classNum):
            
            dicIndex=str(i)
            tempRule=self.selectedRule[dicIndex]
            
            sparsePattern=np.sign(tempRule)
            sizeMat=np.shape(sparsePattern)
            
            if sizeMat[0]!=0:
            
                sparsePattern=np.reshape(sparsePattern, (sizeMat[0],sizeMat[1]*2))   
              
                # use the unique function to find the unique sparse pattern
                uniqueSparsePattern,patternCount=np.unique(sparsePattern,return_counts=True,axis=0)
                
                self.uniqueSparsePattern[dicIndex]=uniqueSparsePattern
                self.uniqueSparsePatternCount[dicIndex]=patternCount
                
                sequenceCount=np.argsort(-patternCount)         
                
                
                finalRule=np.zeros((ruleNumber,sizeMat[1],2))
                for j in range(ruleNumber):
                    
                    for p in range(sizeMat[0]):
                        if (sparsePattern[p,:] == uniqueSparsePattern[sequenceCount[j],:]).all():
                            
                            finalRule[j,:,:]+=tempRule[p,:,:]
                            
                    finalRule[j,:,:]=finalRule[j,:,:]/patternCount[sequenceCount[j]]
                    
                self.finalRule[dicIndex]=finalRule   
                
            else:
                print('Cannot find available rules. Please relax cretarion.')
                    
    
    ###########################################################################
    # The following method will use the detailBackTrack() to collect rules for
    # all classes. In the learnedRule array, those zero arrays will be rmoved
    ###########################################################################
   
    def collectRule(self):
        # this command will compute the generated rules
        collectedAllRule={}
        for i in range(self.classNum):
            dicIndex=str(i)
            collectedAllRule[dicIndex]=self.detailBackTrack(i)       
        # The above code only collect all the rules that are generated
        # the next step is to select the better rule based on the reliability 
        # of each rule 
        
        # In the following code we generate a more structured rule data
        self.collectedRule={}
        for i in range(self.classNum):
            dicIndex=str(i)
            tempRule=collectedAllRule[dicIndex] 
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
            
            self.collectedRule[dicIndex]=tempRule
            
            
    
    ###########################################################################
    # The following set of codes will track all the rules learned from training
    # the decision tree method for a single class
    ###########################################################################
    
    def detailBackTrack(self,selectedClass):
        # we will back track the design rules for class with no inegration
               
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
                    
                    ##########################################################
                    # One can activate the following code to check the process 
                    # of collecting all the rules
                    ##########################################################
                    
                    # tempString = 'The rule is:\n'      
                    
                    # for j in range(self.featureNum):
                        
                    #     if tempRule[j,0]!=0:
                    #         tempString+='feature '
                    #         tempString+= str(j)
                    #         tempString+='<'
                    #         tempString+=str(tempRule[j,0])
                    #         tempString+='\n'
                    #     if tempRule[j,1]!=0:
                    #         tempString+='feature '
                    #         tempString+= str(j)
                    #         tempString+='>'
                    #         tempString+=str(tempRule[j,1])
                    #         tempString+='\n' 
                    # print(tempString)
                    
        return collectedRule