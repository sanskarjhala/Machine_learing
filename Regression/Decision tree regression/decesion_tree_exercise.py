import pandas as pd
df = pd.read_csv('titanic.csv')
df.head()

target = df.Survived
target.head()
inputs = df.drop(['Survived' , 'PassengerId' , 'Name' , 'SibSp' , 'Ticket' , 'Parch'  , "Cabin" , 'Embarked'],axis='columns')
inputs

mean_age = inputs['Age'].mean()
mean_age
inputs['Age'] = inputs['Age'].fillna(mean_age)
inputs['Age'] = inputs['Age'].round(2)
inputs

from sklearn.preprocessing import LabelEncoder
inputs['Fare'] = inputs['Fare'].round(2)
le_Sex = LabelEncoder()
inputs

from sklearn import tree
model = tree.DecisionTreeClassifier()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

model.fit(X_train ,y_train)

model.score(X_test , y_test)