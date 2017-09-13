import pandas as pd 
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Read in data from csv. Each passenger has the following information (one passenger per row)
data = pd.read_csv("train.csv")

#normalize a feature between 0 and 1
def Norm(df,feature):
    max_value = df[feature].max()
    min_value = df[feature].min()
    return (df[feature] - min_value) / (max_value - min_value)

#Normalizing data
data['Sex'].replace(['female','male'],[0.0,1.0],inplace=True) # Code Female as 0, Male as 1
data['Pclass'].replace([1,2,3],[0.0,.5,1.0],inplace=True)     # Upperclass 0, Middleclass .5, Lowerclass 1
data['Age'] = Norm(data,'Age')
data['Age'].fillna(.5,inplace=True)
data['SibSp'] = Norm(data,'SibSp')
data['Parch'] = Norm(data,'Parch')
data['Fare'] = Norm(data,'Fare')

#Build our Input and Output 
testInput = []
testOutput = []
for index,person in data.iterrows():
    testInput.append([person['Sex'],person['Pclass'],person['Age'],person['SibSp'],person['Parch'],person['Fare']])
    testOutput.append(person['Survived'])

#Create our Neural Net with 2 hidden layers
clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(10, 5), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
       
#Fit our Neural Net weights
clf.fit(testInput, testOutput)


#Test the Accuracy
def test(testInput, testOutput):
    correct = 0
    incorrect = 0
    for i in range(0,len(testInput)):
        answer = clf.predict([testInput[i]])[0]
        actual = testOutput[i]
        if answer == actual:
            correct+=1
        else:
            incorrect+=1
           
    return(correct/(correct+incorrect)*100)

print("Accuracy: "+str(test(testInput,testOutput))+"%")

#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html



