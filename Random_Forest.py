import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#import csv to pandas dataframe
income_data=pd.read_csv("income.csv",header=0, delimiter = ", ")
#print(income_data.iloc[0])

#Converting age String to Inrt
income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)
income_data["race-int"] = income_data["race"].apply(lambda row: 0 if row == "White" else 1)

# divide the results and data for prediction
labels = income_data[["income"]]
data = income_data[["capital-gain","capital-loss","hours-per-week","sex-int","age","race-int"]]

#Split the training and testing data using train_test_split
train_data,test_data,train_labels,test_labels = train_test_split(data,labels,random_state =1)

# Create a random forest classifier
forest = RandomForestClassifier(random_state=1)
forest.fit(train_data,train_labels) #Fit training data to random forest
print(forest.feature_importances_)

print(forest.score(test_data, test_labels)) 
print(income_data["race"].value_counts())