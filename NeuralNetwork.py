
# coding: utf-8

# ## Logic Based FizzBuzz Function [Software 1.0]

# In[27]:


import pandas as pd
#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)

def fizzbuzz(n):
    
    # Logic Explanation
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'


# ## Create Training and Testing Datasets in CSV Format

# In[28]:


def createInputCSV(start,end,filename):
    
    # Why list in Python?
    '''Lists are indexed,ordered and allows duplicates. 
       So, it is useful as follows: 
       1) 1-1 association of inputdata and outputdata (which is to be added in a dataframe as it is ordered).
       2) Output data is categorical. So, it has lot of duplicate values. 
       A set can't accomodate duplicates.So can't be used.
       A tuple can't be used to dynamically add elements as it is unchangeable. So, can't be used.
    '''
    inputData   = []
    outputData  = []
    
    # Why do we need training Data?
    ''' The neural networks extracts the features of data and computes the weights in the neural network by processing the 
        training data.
        The combination of weights with minimum loss, found after processing the sample data is used as the final model to
        predict the output of test data.
    '''
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))
    
    # Why Dataframe?
    '''Dataframes are used to create tables by feeding lists of equal length as its columns.
       Here 2 columns "input" and "output" are created in the table.
       This table is stored in csv format.
    '''
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")


# ## Processing Input and Label Data

# In[29]:


def processData(dataset):
    
    # Why do we have to process?
    ''' Input is processed i.e.. converted to its 10 digit binary representation to study the common patterns 
        in the binary representation for numbers which are divisible by 3,5 and both.
        The output is processed i.e.. in this cases enumerated to so that final output can simply be a 4 digit binary
        pattern which represents in which category the input data belongs.
    '''
    data   = dataset['input'].values
    labels = dataset['label'].values
    
    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)
    
    return processedData, processedLabel


# In[30]:


def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:
        
        # Why do we have number 10?
        '''The max input value is 1000. 1000 can be represented by 10 digits in binary representation
           as 2^10=1024 is the max number that can be represented with 10 digit binary representation.
        '''
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData)


# In[31]:


from keras.utils import np_utils

def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel),4)


# ## Model Definition

# In[32]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard

import numpy as np

input_size = 10
drop_out = 0.1
first_dense_layer_nodes  = 512
second_dense_layer_nodes = 4

def get_model():
    
    # Why do we need a model?
    '''A model is used to define the layout and functionality of neurons in the neural network.'''

    # Why use Dense layer and then activation?
    '''The operations can also be done in a single step by passing the activation function as a 
       parameter to the Dense() function.
       But this approach is better as it can be used to retrieve the outputs of the last layer 
       (before activation) out of such defined model.
       Reference: https://stackoverflow.com/questions/40866124/difference-between-dense-and-activation-layer-in-keras
    '''
    
    # Why use sequential model with layers?
    '''Since no re-use of the layers is required for this task, sequential model is used.'''
   
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    # Why dropout?
    '''It is to avoid overfitting of the model on the training dataset by removing a proportional amount of data.'''

    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    
    # Why Softmax?
    ''' To get the most probably category of the input based on the values in the output neurons.
        It is most useful in finding category.
    '''
    
    model.summary()
    
    # Why use categorical_crossentropy?
    '''Because the purpose is to determine the category of the input, categorical_crossentropy is used as the
       loss function.
    '''

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# # <font color='blue'>Creating Training and Testing Datafiles</font>

# In[33]:


# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')


# # <font color='blue'>Creating Model</font>

# In[34]:


model = get_model()


# # <font color = blue>Run Model</font>

# In[35]:


validation_data_split = 0.2
num_epochs = 10000
model_batch_size = 128
tb_batch_size = 32
early_patience = 100

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Read Dataset
dataset = pd.read_csv('training.csv')

# Process Dataset
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )


# # <font color = blue>Training and Validation Graphs</font>

# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# # <font color = blue>Testing Accuracy [Software 2.0]</font>

# In[37]:


def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


# In[38]:


wrong   = 0
right   = 0

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))
    
    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))

# Please input your UBID and personNumber 
testDataInput = testData['input'].tolist()
testDataLabel = testData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "baalajiv")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50287414")

predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')

