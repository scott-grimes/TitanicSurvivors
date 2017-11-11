import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

#normalize a feature between 0 and 1
def Norm(df,feature):
    max_value = df[feature].max()
    min_value = df[feature].min()
    return (df[feature] - min_value) / (max_value - min_value)
    
def getTitle(fullName):
    start = fullName.find(",")+2
    end = fullName[start::].find(".")+start
    return fullName[start:end]
    
def titleToValue(x):
    if(x == 'Miss'):
        return 0.0
    if(x== 'Mrs'):
        return 0.2
    if(x== 'Master'):
        return 0.4
    if(x == 'Mr'):
        return 0.8
    return 1.0
    
#parse data from the CSV, output an array of inputs and the expected output
def parseData(rawData):
    #Normalizing data
    rawData['Sex'].replace(['male','female'],[0.0,1.0],inplace=True) # Code Female as 0, Male as 1
    
    rawData['Pclass'].replace([1,2,3],[0.0,.5,1.0],inplace=True)     # Upperclass 0, Middleclass .5, Lowerclass 1
    rawData['Pclass'].fillna(.5,inplace=True)
    
    rawData['Age'] = Norm(rawData,'Age')
    rawData['Age'].fillna(.5,inplace=True)
    
    rawData['SibSp'] = Norm(rawData,'SibSp')
    rawData['SibSp'].fillna(0.0,inplace=True)
    
    rawData['Parch'] = Norm(rawData,'Parch')
    rawData['Parch'].fillna(0.0,inplace=True)
    
    rawData['Fare'] = Norm(rawData,'Fare')
    rawData['Fare'].fillna(0.0,inplace=True)
    
    
    rawData['Name'] = rawData['Name'].apply(lambda x: getTitle(x))
    
    #Determine what titles exist
    #print(rawData['Name'].value_counts())
    rawData['Name'] = rawData['Name'].apply(lambda x: titleToValue(x))
   
    
    
    #Build our Input and Output 
    inputs = []
    outputs = []
    for index,person in rawData.iterrows():
        #title = getTitle(person['Name'])
        inputs.append([person['Sex'],person['Pclass'],person['Age'],person['SibSp'],person['Parch'],person['Fare'],person['Name']])
        if 'Survived' in rawData:
            outputs.append(person['Survived'])
    return inputs, outputs
        
#Test the Accuracy of a given network
def test(testInput, testOutput, network):
    correct = 0
    incorrect = 0
    for i in range(0,len(testInput)):
        answer = network.predict([testInput[i]])[0]
        actual = testOutput[i]
        if answer == actual:
            correct+=1
        else:
            incorrect+=1
           
    return(correct/(correct+incorrect)*100)

#Predict the survival of a person
def predict(inputs,network):
    prediction = []
    for i in range(0,len(inputs)):
        prediction.append(network.predict([inputs[i]])[0])
    return prediction

# Read in data from csv. Each passenger has the following information (one passenger per row)
raw_training = pd.read_csv("../input/train.csv")
raw_testing = pd.read_csv("../input/test.csv")

#Create our Neural Net with 2 hidden layers
clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(8, 5), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

#Fetch our training data
trainingInputs, trainingOutputs = parseData(raw_training)

#Fit our Neural Net weights using the training data
clf.fit(trainingInputs, trainingOutputs)
print("Training Accuracy: "+str(test(trainingInputs,trainingOutputs,clf))+"%")

#Fetch our testing data
testingInputs, testingOutputes = parseData(raw_testing)

print("Predictions:")
print(predict(testingInputs,clf))
    
#Accuracy: 84.3%


