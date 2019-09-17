
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
import numpy as np
import csv
import math


# In[2]:


C_Lambda = 0.03 #where used?# In closedformsolution
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10 
PHI = []
IsSynthetic = False #where used? To remove columns with zero variance


# In[3]:


def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f: #rU is a file mode. Means read as text file with newline delimiter
        reader = csv.reader(f) #returns an iterator over each row in file
        for row in reader:  #row is a string list of the values that were seperated by comma ie.. "a,b" as ["a","b"]
            t.append(int(row[0])) #1st column is the target column. Hence, only it is appended as an int.
    #print("Raw Training Generated..")
    return t

#what is role of IsSynthetic?#
#why columns 5,6,7,8,9 are deleted if IsSynthetic==False?#
#These columns have only value 0 which implies the variance in these columns are zero. Zero variance can't be tolerated 
#while computing inverse covariance matrix.Hence removed
def GenerateRawData(filePath, IsSynthetic): 
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column)) #each column value of a row is stored as float value in dataRow list
            dataMatrix.append(dataRow)   
    
    if IsSynthetic == False :
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1) #axis=0 means 2nd arg is row number.axis=1 column number
    dataMatrix = np.transpose(dataMatrix) #transpose of dataMatrix    
    #print ("Data Matrix Generated..")
    return dataMatrix

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80): #rawTraining is target vector
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01))) #computes num.rows * 0.8
    t           = rawTraining[:TrainingLen] #extracts row numbers 1 to num.rows*0.8 or row index 0 to (num.rows*0.8) -1
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80): #rowData is data matrix
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent)) #same as GenerateTrainingTarget but on data matrix
    d2 = rawData[:,0:T_len] #T_len used on column bounding as data matrix is transposed.
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01)) # num.rows out of total num.rows equivalent to valPercent
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End] #rows right after traing data for the size of valsize is extracted
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01)) #same as GenerateValData but to find target vector
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

#what is bigsigma?#sigma^2#
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic): #Raw data 41Xn, MuMatrix 10X41
    BigSigma    = np.zeros((len(Data),len(Data))) #creates 41X41 matrix of zeros
    DataT       = np.transpose(Data) #nX41
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01)) #first 80 percent lines        
    varVect     = []
    for i in range(0,len(DataT[0])): #41 times
        vct = []
        for j in range(0,int(TrainingLen)): #55k times
            vct.append(Data[i][j])    
        varVect.append(np.var(vct)) #variance of each feature
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j] #reassigning diagonal values in zero matrix
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma) 
    else:
        BigSigma = np.dot(200,BigSigma) #multiplying each covariance with 200 #why?#
    ##print ("BigSigma Generated..")
    #print("varVect: ",varVect)
    return BigSigma

#Computes Gaussian power value#
def GetScalar(DataRow,MuRow, BigSigInv):  #1X41,1X41,41X41
    R = np.subtract(DataRow,MuRow) #x-Mu for each column #1X41
    T = np.dot(BigSigInv,np.transpose(R)) #(x-Mu)T/sigma^2 #41X1 
    L = np.dot(R,T) #(x-Mu)T/sigma^2 X (x-Mu) #1X1 #Scalar
    return L

#Computes Phi# 
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

#Generates phi matrix#
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):#Raw data 41Xn, Mumatrix 10X41, bigsigma 41X41
    DataT = np.transpose(Data) #nX41
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01)) #80% of total records        
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))  #55kX10 zero matrix
    BigSigInv = np.linalg.inv(BigSigma) #bigsigma inverse
    for  C in range(0,len(MuMatrix)): #10
        for R in range(0,int(TrainingLen)): #55k
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)#e^(-((x-Mu)T/2*sigma^2) * (x-Mu))
    #print ("PHI Generated..")
    return PHI

#computes the weight vector
def GetWeightsClosedForm(PHI, T, Lambda): #10X55k, 55k X 1
    Lambda_I = np.identity(len(PHI[0])) #identity matrix 10X10
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda #diagonal values as lambda
    PHI_T       = np.transpose(PHI) #10X55k
    PHI_SQR     = np.dot(PHI_T,PHI) #10X10
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR) #adding regularization factor lambda
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T) #phi inverse #10X55k
    W           = np.dot(INTER, T) #weights 10X1
    ##print ("Training Weights Generated..")
    return W

#This generates target output for given PHI matrix and weights computed earlier
def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI)) #computes dot product of W vector and VAL_PHI matrix
    ##print ("Test Out Generated..")
    return Y

#Given computed and actual target value vectors, computes ERMS error
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2) #(computed - actual)^2
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]): #if integer round off of computed value = actual value
            counter+=1                                           #increment counter by 1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT))) #Values matched divided by total values in percentage
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT)))) #accuracy and RMS value of error


# ## Fetch and Prepare Dataset

# In[4]:


RawTarget = GetTargetVector('Querylevelnorm_t.csv')
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic) #transpose of data as matrix


# ## Prepare Training Data

# In[5]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)


# ## Prepare Validation Data

# In[6]:


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Prepare Test Data

# In[7]:


TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# In[8]:


kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData)) #finding M clusters in training data
Mu = kmeans.cluster_centers_ #Getting cluster centers #for each centroid 41 values ie.. it finds 10 centroids for each feature
BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic) #returns bigsigma for training set 41X41
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent) #55k X 10
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) #10X1 weight vector obtained
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) #using same bigsigma as in training set
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100) #using same bigsigma as in training set


# In[9]:


print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set 

# In[10]:


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W) #returns target output value for the train dataset
VAL_TEST_OUT = GetValTest(VAL_PHI,W) #returns target output value for the validation dataset
TEST_OUT     = GetValTest(TEST_PHI,W) #returns target output value for the test dataset

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


# In[11]:


print ('UBITname      = baalajiv')
print ('Person Number = 50287414')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = " + str(M)+ "\nLambda = " + str(C_Lambda))
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))


# ## Gradient Descent solution for Linear Regression

# In[12]:


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


# In[13]:


W_Now        = np.dot(220, W) #randomly 220 times the weights computed using closed form method
La           = 2 #Lambda value for SGD
learningRate = 0.01
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

for i in range(0,400):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])#-(target-computed)* phi #negative as change must be in opposite direction of gradient
    La_Delta_E_W  = np.dot(La,W_Now) #(lambda) * Ew
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)  #Total delta E = delta Ed + delta Ew  
    Delta_W       = -np.dot(learningRate,Delta_E) #changind weights based on their partial differential values times learning rate
    W_T_Next      = W_Now + Delta_W #The change in weights are subtracted from previous weights
    W_Now         = W_T_Next #updating weights
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) #computes target output values for given phi matrix and weights
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget) #gets training set accuracy given computed and actual target values
    L_Erms_TR.append(float(Erms_TR.split(',')[1])) #takes ERMS value and discards accuracy from output of previous line
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) #computes target output values for given phi matrix and weights
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct) #gets validation set accuracy given computed and actual target values
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))#takes ERMS value and discards accuracy from output of previous line
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) #computes target output values for given phi matrix and weights
    Erms_Test = GetErms(TEST_OUT,TestDataAct) #gets test set accuracy given computed and actual target values
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))#takes ERMS value and discards accuracy from output of previous line


# In[14]:


print ('----------Gradient Descent Solution--------------------')
print ("M = " + str(M) + "\nLambda  = " + str(La) + "\neta=0.01")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

