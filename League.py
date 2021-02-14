import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv("League_of_Legends.csv")

#Visulising The dataset

blue_kills= dataset['blueKills']
red_kills = dataset['redKills']

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
both_kills = blue_kills + red_kills
g_id = dataset['gameId']
ax.scatter(both_kills,g_id)
#plt.show()

#Doing Feature Engineering

blue_win = dataset['blueWins']
dataset.drop(labels=['blueWins'],axis=1,inplace=True)
dataset.insert(20,'blueWins',blue_win)

#Splitting data into train and test data

blue_train = dataset.iloc[: , 1:20].values
blue_win_train = dataset.iloc[:,20].values
red_test = dataset.iloc[: , 21:].values

#Doing Feature Scalling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
blue_train = sc.fit_transform(blue_train) 
red_test = sc.transform(red_test)

#Training The Model

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(blue_train,blue_win_train)

#Pridicting The Winners of Red Team

y_pred = classifier.predict(red_test)
redwins = y_pred
dataset.insert(40,'redWins',redwins)

#Checking The Accuracy of The model

from sklearn.metrics import accuracy_score
ac = accuracy_score(red_test, y_pred)
print(ac)