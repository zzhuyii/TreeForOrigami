from DecisionTree import TreeMethod

class runTree():
    def __init__(self, i, j):
        self.i=i
        self.j=j
        
    def Compute(self,  X_train, Y_train, featureName, X_test, Y_test):
        
        dep=[8,12,16,20,24,28,32]
        treeNum=[20,50,100,200,400]
        
        i=self.i
        j=self.j
        
        tree=TreeMethod()
        tree.setParameter(alpha=0.00001, depth=dep[i], num_tree=treeNum[j])
        tree.train(X_train, Y_train, featureName)
        tree.computeRule(X_train, Y_train, minData=2, ruleNumber=3)
        tree.testRule(X_test,Y_test)
        
        # tree.printRule()  
        Pre = (1/3*tree.finalRuleTestPrecision[0]+
                1/3*tree.finalRuleTestPrecision[1]+
                1/3*tree.finalRuleTestPrecision[2])
        
        # Use the trandom forest to predict the results of the testing set
        Y_pred=tree.predict(X_test)
        
        # calculate the accruacy
        accuratePredict = sum(Y_pred==Y_test)            
        Acc = accuratePredict / len(Y_pred)

        return [i,j, Acc, Pre]