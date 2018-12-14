import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 500)

data_train = pd.read_csv("resources/train.csv", low_memory=False)
data_test = pd.read_csv("resources/test.csv", low_memory=False)
data_test_labels = pd.read_csv("resources/test_labels.csv", low_memory=False)

# show number of rows for all the data sets
print("Total data_train rows: {0}".format(len(data_train)))
print("Total data_test rows: {0}".format(len(data_test)))
print("Total data_test_labels rows: {0}".format(len(data_test_labels)))

# show headers for all the data sets
print("headers for data_train: {0}".format(list(data_train)))
print("headers for data_test: {0}".format(list(data_test)))
print("headers for data_test_labels: {0}".format(list(data_test_labels)))

# show the first few rows for all the data sets
print(data_train.head())
print(data_test.head())
print(data_test_labels.head())

# show the stats on comment length. the shortest is 6 and longest is 5000.
print(data_train["comment_text"].str.len().describe())

# create a seventh class to reflect comment_text that does not belong to any of the six classes.
# data_train['unk'] = 1 - data_train[data_train.columns[2: ]].max(axis=1)

# check to ensure the 'unk' class is added correctly.
print(data_train.head())
print(data_train.tail())

# check for null values in the training set.
print(data_train.isnull().any())
# alternatively
print(data_train.isnull().sum())

# number of training examples for each class
class_num = data_train.iloc[:, 2:].sum()

print('training examples for each class')
print(class_num)

# check for correlation between the classes
cor = data_train.corr()
plt.figure(figsize=(8, 8))
sns.heatmap(cor, annot=True, center=True, square=True)
plt.show()


