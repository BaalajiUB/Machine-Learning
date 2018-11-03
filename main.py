import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils

from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import random

from sklearn.utils import shuffle

C_Lambda = 0.03 
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10 
PHI = []

input_size = 18
drop_out = 0.1
first_dense_layer_nodes  = 512
second_dense_layer_nodes = 2

def preprocess():
    hofd = pd.read_csv('HumanObserved-Features-Data\HumanObserved-Features-Data.csv').iloc[:,1:]
    hofd_s = pd.read_csv('HumanObserved-Features-Data\same_pairs.csv')
    hofd_d = pd.read_csv('HumanObserved-Features-Data\diffn_pairs.csv')

    rec_cnt = len(hofd_s)
    hofd_d = hofd_d.iloc[random.sample(range(len(hofd_d)),rec_cnt),:]

    #print(hofd.head())#print(hofd_s.head())#print(hofd_d.head())
    #read.csv returns dataframe
    #iloc is used for slicing dataset #print(hofd.iloc[0:5,0:1])

    ##Concatenation##
    ##human observed features data - same##
    inter = hofd_s.merge(hofd,left_on='img_id_A',right_on='img_id',how='left')
    inter = inter.merge(hofd,left_on='img_id_B',right_on='img_id',how='left',suffixes=['A','B'])
    #https://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns
    cols = inter.columns.tolist()
    t = cols[2]
    cols = cols[0:2] + cols[4:13] + cols[14:]
    cols.append(t)
    hofd_same_concat = inter[cols]
    print(hofd_same_concat.columns)
    #print(hofd_same_concat.head())
    #hofd_same_concat.to_csv('hofd_same_concat.csv')
    print(len(hofd_same_concat))

    ##human observed features data - different##
    inter = hofd_d.merge(hofd,left_on='img_id_A',right_on='img_id',how='left')
    inter = inter.merge(hofd,left_on='img_id_B',right_on='img_id',how='left',suffixes=['A','B'])
    #https://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns
    cols = inter.columns.tolist()
    t = cols[2]
    cols = cols[0:2] + cols[4:13] + cols[14:]
    cols.append(t)
    hofd_diff_concat = inter[cols]
    print(hofd_diff_concat.columns)
    #print(hofd_diff_concat.head())
    #hofd_diff_concat.to_csv('hofd_diff_concat.csv')
    print(len(hofd_diff_concat))

    frames = [hofd_same_concat,hofd_diff_concat]
    hofd_concat = pd.concat(frames)

    hofd_concat = shuffle(hofd_concat)
    hofd_concat.to_csv('hofd_concat.csv')

    del hofd_same_concat,hofd_diff_concat

    ##Subtraction##
    ##human observed features data - same##
    inter1 = hofd_s.merge(hofd,left_on='img_id_A',right_on='img_id',how='left')
    cols = inter1.columns.tolist()
    t = cols[2]
    cols = cols[0:2] + cols[4:]
    cols.append(t)
    inter1 = inter1[cols]
    #print(inter1.columns)

    inter2 = hofd_s.merge(hofd,left_on='img_id_B',right_on='img_id',how='left')
    cols = inter2.columns.tolist()
    t = cols[2]
    cols = cols[0:2] + cols[4:]
    cols.append(t)
    inter2 = inter2[cols]
    #print(inter2.columns)

    #https://stackoverflow.com/questions/38419286/subtracting-multiple-columns-and-appending-results-in-pandas-dataframe
    tmp = inter1.iloc[:,2:-1] - inter2.iloc[:,2:-1].values #subtraction
    #print(tmp.columns)
    #print(inter1.iloc[:5,2:-1])
    #print(inter2.iloc[:5,2:-1])
    #print(tmp.head())
    t = hofd_s.iloc[:,:-1] #target
    t1 = pd.concat([t.reset_index(drop=True), tmp], axis=1) 
    t1['target']=1
    hofd_same_subtract = t1
    print(hofd_same_subtract.columns)
    #print(hofd_same_subtract.head())
    #print(inter2.columns) #to get column names
    #print(inter2.dtypes) #to get data types of each column
    #hofd_same_subtract.to_csv('hofd_same_subtract.csv')
    print(len(hofd_same_subtract))

    ##human observed features data - different##
    inter1 = hofd_d.merge(hofd,left_on='img_id_A',right_on='img_id',how='left')
    cols = inter1.columns.tolist()
    t = cols[2]
    cols = cols[0:2] + cols[4:]
    cols.append(t)
    inter1 = inter1[cols]
    #print(inter1.columns)

    inter2 = hofd_d.merge(hofd,left_on='img_id_B',right_on='img_id',how='left')
    cols = inter2.columns.tolist()
    t = cols[2]
    cols = cols[0:2] + cols[4:]
    cols.append(t)
    inter2 = inter2[cols]
    #print(inter2.columns)

    #https://stackoverflow.com/questions/38419286/subtracting-multiple-columns-and-appending-results-in-pandas-dataframe
    tmp = inter1.iloc[:,2:-1] - inter2.iloc[:,2:-1].values
    #print(inter1.iloc[:5,2:-1])
    #print(inter2.iloc[:5,2:-1])
    #print(tmp.head())
    t = hofd_d.iloc[:,:-1]
    t1 = pd.concat([t.reset_index(drop=True), tmp], axis=1)
    t1['target']=0
    hofd_diff_subtract = t1
    print(hofd_diff_subtract.columns)
    #print(hofd_diff_subtract.head())
    #print(inter2.columns) #to get column names
    #print(inter2.dtypes) #to get data types of each column
    #hofd_diff_subtract.to_csv('hofd_diff_subtract.csv')
    print(len(hofd_diff_subtract))

    frames = [hofd_same_subtract,hofd_diff_subtract]
    hofd_subtract = pd.concat(frames)

    '''
    x = list(range(len(hofd_subtract)))
    random.shuffle(x)
    hofd_subtract = hofd_subtract.iloc[x,:]
    '''
    hofd_subtract=shuffle(hofd_subtract)
    hofd_subtract.to_csv('hofd_subtract.csv')

    del hofd_same_subtract,hofd_diff_subtract
    #####################################################################################################################

    ##GSC##
    gsc = pd.read_csv('GSC-Features-Data\GSC-Features.csv')
    gsc_s = pd.read_csv('GSC-Features-Data\same_pairs.csv')
    gsc_d = pd.read_csv('GSC-Features-Data\diffn_pairs.csv')

    rec_cnt = 1500
    gsc_s = gsc_s.iloc[random.sample(range(len(gsc_s)),rec_cnt),:]
    gsc_d = gsc_d.iloc[random.sample(range(len(gsc_d)),rec_cnt),:]

    ##Concatenation##
    ##GSC data - same##
    inter = gsc_s.merge(gsc,left_on='img_id_A',right_on='img_id',how='left')
    inter = inter.merge(gsc,left_on='img_id_B',right_on='img_id',how='left',suffixes=['A','B'])
    #https://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns
    cols = inter.columns.tolist()
    t = cols[2]
    cols = cols[0:2] + cols[4:516] + cols[517:]
    cols.append(t)
    gsc_same_concat = inter[cols]
    print(gsc_same_concat.columns)
    #print(gsc_same_concat.head())
    #gsc_same_concat.to_csv('gsc_same_concat.csv')
    print(len(gsc_same_concat))

    ##GSC data - different##
    inter = gsc_d.merge(gsc,left_on='img_id_A',right_on='img_id',how='left')
    inter = inter.merge(gsc,left_on='img_id_B',right_on='img_id',how='left',suffixes=['A','B'])
    #https://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns
    cols = inter.columns.tolist()
    t = cols[2]
    cols = cols[0:2] + cols[4:516] + cols[517:]
    cols.append(t)
    gsc_diff_concat = inter[cols]
    print(gsc_diff_concat.columns)
    #print(gsc_diff_concat.head())
    #print(len(gsc_diff_concat))
    #gsc_diff_concat.to_csv('gsc_diff_concat.csv')
    print(len(gsc_diff_concat))

    frames = [gsc_same_concat,gsc_diff_concat]
    gsc_concat = pd.concat(frames)

    '''
    x = list(range(len(gsc_concat)))
    random.shuffle(x)
    gsc_concat = gsc_concat.iloc[x,:]
    '''
    gsc_concat = shuffle(gsc_concat)
    gsc_concat.to_csv('gsc_concat.csv')

    del gsc_same_concat,gsc_diff_concat

    ##Subtraction##
    ##GSC data - same##
    inter1 = gsc_s.merge(gsc,left_on='img_id_A',right_on='img_id',how='left')
    cols = inter1.columns.tolist()
    t = cols[2]
    cols = cols[0:2] + cols[4:]
    cols.append(t)
    inter1 = inter1[cols]
    #print(inter1.columns)

    inter2 = gsc_s.merge(gsc,left_on='img_id_B',right_on='img_id',how='left')
    cols = inter2.columns.tolist()
    t = cols[2]
    cols = cols[0:2] + cols[4:]
    cols.append(t)
    inter2 = inter2[cols]
    #print(inter2.columns)

    #https://stackoverflow.com/questions/38419286/subtracting-multiple-columns-and-appending-results-in-pandas-dataframe
    tmp = inter1.iloc[:,2:-1] - inter2.iloc[:,2:-1].values #subtraction
    #print(tmp.columns)
    #print(inter1.iloc[:5,2:-1])
    #print(inter2.iloc[:5,2:-1])
    #print(tmp.head())
    t = gsc_s.iloc[:,:-1] #except target
    t1 = pd.concat([t.reset_index(drop=True), tmp], axis=1)
    t1['target']=1
    gsc_same_subtract = t1
    #print(gsc_same_subtract.head())
    print(gsc_same_subtract.columns) #to get column names
    #print(inter2.dtypes) #to get data types of each column
    #gsc_same_subtract.to_csv('gsc_same_subtract.csv')
    print(len(gsc_same_subtract))

    ##GSC data - different##
    inter1 = gsc_d.merge(gsc,left_on='img_id_A',right_on='img_id',how='left')
    cols = inter1.columns.tolist()
    t = cols[2]
    cols = cols[0:2] + cols[4:]
    cols.append(t)
    inter1 = inter1[cols]
    #print(inter1.columns)

    inter2 = gsc_d.merge(gsc,left_on='img_id_B',right_on='img_id',how='left')
    cols = inter2.columns.tolist()
    t = cols[2]
    cols = cols[0:2] + cols[4:]
    cols.append(t)
    inter2 = inter2[cols]
    #print(inter2.columns)

    #https://stackoverflow.com/questions/38419286/subtracting-multiple-columns-and-appending-results-in-pandas-dataframe
    tmp = inter1.iloc[:,2:-1] - inter2.iloc[:,2:-1].values #subtraction
    #print(inter1.iloc[:5,2:-1])
    #print(inter2.iloc[:5,2:-1])
    #print(tmp.head())
    t = gsc_d.iloc[:,:-1] #except target
    t1 = pd.concat([t.reset_index(drop=True), tmp], axis=1)
    t1['target']=0
    gsc_diff_subtract = t1
    print(gsc_diff_subtract.columns)
    #gsc_diff_subtract.to_csv('gsc_diff_subtract.csv')
    print(len(gsc_diff_subtract))

    frames = [gsc_same_subtract,gsc_diff_subtract]
    gsc_subtract = pd.concat(frames)

    '''
    x = list(range(len(gsc_subtract)))
    random.shuffle(x)
    gsc_subtract = gsc_subtract.iloc[x,:]
    '''
    gsc_subtract = shuffle(gsc_subtract)
    gsc_subtract.to_csv('gsc_subtract.csv')

    del gsc_same_subtract,gsc_diff_subtract


def get_model():
    
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    
    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def nn(filename):
    global input_size

    RawTarget = GetTargetVector(filename)
    RawData   = GenerateRawData(filename) #transpose of data as matrix
    print(RawData.shape)

    input_size = RawData.shape[0]
    
    model = get_model()

    validation_data_split = 0.2
    num_epochs = 10000
    model_batch_size = 128
    tb_batch_size = 32
    early_patience = 100

    tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

    # Process Dataset
    processedData = np.transpose(GenerateTrainingDataMatrix(RawData,80))
    processedLabel = np_utils.to_categorical(np.array(GenerateTrainingTarget(RawTarget,80)),2)

    
    history = model.fit(processedData
                        , processedLabel
                        , validation_split=validation_data_split
                        , epochs=num_epochs
                        , batch_size=model_batch_size
                        , callbacks = [tensorboard_cb,earlystopping_cb]
                       )

    #get_ipython().run_line_magic('matplotlib', 'inline')
    df = pd.DataFrame(history.history)
    #df.plot(subplots=True, grid=True, figsize=(10,15))

    wrong   = 0
    right   = 0

    processedTestLabel = np_utils.to_categorical(np.array(GenerateValTargetVector(RawTarget, 20, (len(processedData)))),2)
    processedTestData    = np.transpose(GenerateValData(RawData,ValidationPercent, (len(processedData))))

    predictedTestLabel = []

    for i,j in zip(processedTestData,processedTestLabel):
        y = model.predict(np.array(i).reshape(-1,input_size))
        predictedTestLabel.append(y.argmax())
        
        if j.argmax() == y.argmax():
            right = right + 1
        else:
            wrong = wrong + 1

    print("Errors: " + str(wrong), " Correct :" + str(right))

    print("Testing Accuracy: " + str(right/(right+wrong)*100))

def sigmoid(x):
    #t = np.dot(np.transpose(w),np.transpose(x))
    return 1/(1+np.exp(-x))

def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f: 
        reader = csv.reader(f)
        header = next(reader,None) #to skip headers
        for row in reader:
            t.append(int(row[-1])) 
    #print("Raw Training Generated..")
    return t

def GenerateRawData(filePath): 
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        header = next(reader,None) #to skip headers
        #print(header)
        for row in reader:
            dataRow = []
            tmp = row[3:-1]
            for column in tmp:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   

    #print('dim x : ' + str(len(dataMatrix))) #3000
    #print("dim y : " , str(len(dataMatrix[0]))) #1025
    varVect = []
    for i in range(len(dataMatrix[0])): 
        vct = []
        for j in range(len(dataMatrix)): 
            vct.append(dataMatrix[j][i])    
        if np.var(vct)==0: #variance of each feature
            varVect.append(i)

    dataMatrix = np.delete(dataMatrix, varVect, axis=1)
    
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
def GenerateBigSigma(Data, MuMatrix,TrainingPercent): #Raw data 41Xn, MuMatrix 10X41
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
        BigSigma[j][j] = varVect[j]+0.2 #reassigning diagonal values in zero matrix
    BigSigma = np.dot(200,BigSigma) #multiplying each covariance with 200 #why?#
    #print ("BigSigma Generated..")
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

def GetErms_log(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - sigmoid(VAL_TEST_OUT[i])),2) #(computed - actual)^2
        if(int(np.around(sigmoid(VAL_TEST_OUT[i]), 0)) == ValDataAct[i]): #if integer round off of computed value = actual value
            counter+=1                                           #increment counter by 1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT))) #Values matched divided by total values in percentage
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT)))) #accuracy and RMS value of error

def linear_reg(filename):
    # ## Fetch and Prepare Dataset

    RawTarget = GetTargetVector(filename)
    RawData   = GenerateRawData(filename) #transpose of data as matrix


    # ## Prepare Training Data

    TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
    TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
    print(TrainingTarget.shape)
    print(TrainingData.shape)


    # ## Prepare Validation Data

    ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
    ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
    print(ValDataAct.shape)
    print(ValData.shape)


    # ## Prepare Test Data

    TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
    print(ValDataAct.shape)
    print(ValData.shape)


    # ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData)) #finding M clusters in training data
    Mu = kmeans.cluster_centers_ #Getting cluster centers #for each centroid 41 values ie.. it finds 10 centroids for each feature
    BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent) #returns bigsigma for training set 41X41
    TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent) #55k X 10
    W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) #10X1 weight vector obtained
    TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) #using same bigsigma as in training set
    VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100) #using same bigsigma as in training set

    # ## Finding Erms on training, validation and test set 

    TR_TEST_OUT  = GetValTest(TRAINING_PHI,W) #returns target output value for the train dataset
    VAL_TEST_OUT = GetValTest(VAL_PHI,W) #returns target output value for the validation dataset
    TEST_OUT     = GetValTest(TEST_PHI,W) #returns target output value for the test dataset

    TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
    ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
    TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))
    '''
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
    '''
    # ## Gradient Descent solution for Linear Regression

    print ('----------------------------------------------------')
    print ('--------------Please Wait for 2 mins!----------------')
    print ('----------------------------------------------------')

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
        
        #print('-----------------TrainingData Accuracy---------------------')
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) #computes target output values for given phi matrix and weights
        Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget) #gets training set accuracy given computed and actual target values
        L_Erms_TR.append(float(Erms_TR.split(',')[1])) #takes ERMS value and discards accuracy from output of previous line
        
        #print('-----------------ValidationData Accuracy---------------------')
        VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) #computes target output values for given phi matrix and weights
        Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct) #gets validation set accuracy given computed and actual target values
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))#takes ERMS value and discards accuracy from output of previous line
        
        #print('-----------------TestingData Accuracy---------------------')
        TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) #computes target output values for given phi matrix and weights
        Erms_Test = GetErms(TEST_OUT,TestDataAct) #gets test set accuracy given computed and actual target values
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))#takes ERMS value and discards accuracy from output of previous line

    print ('----------Gradient Descent Solution--------------------')
    print(filename[:-4])
    print ("M = " + str(M) + "\nLambda  = " + str(La) + "\neta = " + str(learningRate))
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

def log_reg(filename):
    # ## Fetch and Prepare Dataset

    RawTarget = GetTargetVector(filename)
    RawData   = GenerateRawData(filename) #transpose of data as matrix


    # ## Prepare Training Data

    TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
    TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
    print(TrainingTarget.shape)
    print(TrainingData.shape)


    # ## Prepare Validation Data

    ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
    ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
    print(ValDataAct.shape)
    print(ValData.shape)


    # ## Prepare Test Data

    TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
    print(ValDataAct.shape)
    print(ValData.shape)


    # ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData)) #finding M clusters in training data
    Mu = kmeans.cluster_centers_ #Getting cluster centers #for each centroid 41 values ie.. it finds 10 centroids for each feature
    BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent) #returns bigsigma for training set 41X41
    TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent) #55k X 10
    #W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) #10X1 weight vector obtained
    TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) #using same bigsigma as in training set
    VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100) #using same bigsigma as in training set


    # ## Gradient Descent solution for Linear Regression

    print ('----------------------------------------------------')
    print ('--------------Please Wait for 2 mins!----------------')
    print ('----------------------------------------------------')

    W_Now        = np.random.random((TRAINING_PHI.shape[1],1)) #randomly 220 times the weights computed using closed form method
    La           = 2 #Lambda value for SGD
    learningRate = 0.1
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    W_Mat        = []

    for i in range(0,400):
        
        #print ('---------Iteration: ' + str(i) + '--------------')
        x = sigmoid(GetValTest(TRAINING_PHI,np.transpose(W_Now)))
        y = np.subtract(TrainingTarget,x)
        Delta_E_D     = np.dot(np.transpose(TRAINING_PHI),np.transpose(y))/TrainingTarget.shape[0]#-(target-computed)* phi #negative as change must be in opposite direction of gradient
        La_Delta_E_W  = np.dot(La,W_Now) #(lambda) * Ew
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)  #Total delta E = delta Ed + delta Ew  
        Delta_W       = -np.dot(learningRate,Delta_E) #changind weights based on their partial differential values times learning rate
        W_T_Next      = W_Now + Delta_W #The change in weights are subtracted from previous weights
        W_Now         = W_T_Next #updating weights
        
        #print('-----------------TrainingData Accuracy---------------------')
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,np.transpose(W_T_Next)) #computes target output values for given phi matrix and weights
        Erms_TR       = GetErms_log(np.transpose(TR_TEST_OUT),TrainingTarget) #gets training set accuracy given computed and actual target values
        L_Erms_TR.append(float(Erms_TR.split(',')[1])) #takes ERMS value and discards accuracy from output of previous line
        
        #print('-----------------ValidationData Accuracy---------------------')
        VAL_TEST_OUT  = GetValTest(VAL_PHI,np.transpose(W_T_Next)) #computes target output values for given phi matrix and weights
        Erms_Val      = GetErms_log(np.transpose(VAL_TEST_OUT),ValDataAct) #gets validation set accuracy given computed and actual target values
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))#takes ERMS value and discards accuracy from output of previous line
        
        #print('-----------------TestingData Accuracy---------------------')
        TEST_OUT      = GetValTest(TEST_PHI,np.transpose(W_T_Next)) #computes target output values for given phi matrix and weights
        Erms_Test = GetErms_log(np.transpose(TEST_OUT),TestDataAct) #gets test set accuracy given computed and actual target values
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))#takes ERMS value and discards accuracy from output of previous line

    print ('----------Gradient Descent Solution--------------------')
    print(filename[:-4])
    print ("M = " + str(M) + "\nLambda  = " + str(La) + "\neta = " + str(learningRate))
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

preprocess()

linear_reg('hofd_concat.csv')
linear_reg('hofd_subtract.csv')
linear_reg('gsc_concat.csv')
linear_reg('gsc_subtract.csv')

log_reg('hofd_concat.csv')
log_reg('hofd_subtract.csv')
log_reg('gsc_concat.csv')
log_reg('gsc_subtract.csv')

input_size = 18
nn('hofd_concat.csv')
input_size = 9
nn('hofd_subtract.csv')
input_size = 512
nn('gsc_concat.csv')
input_size = 1024
nn('gsc_subtract.csv')
