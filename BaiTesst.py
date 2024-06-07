import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score, roc_curve, recall_score

### I Data Cleaning
# Reading dataset
train_data = pd.read_csv('churn-bigml-80.csv')
test_data = pd.read_csv('churn-bigml-20.csv')

# Check for missing values
missing_values = train_data.isnull().sum()
missing_values = test_data.isnull().sum()

# Delete rows missing values
train_data.dropna()
test_data.dropna()

# Remove duplicate rows
train_data = train_data.drop_duplicates()
test_data = test_data.drop_duplicates()


### II: EDA: Represents the correlation between the target data('churn') column and the remaining columns
#Checking the information of the data
train_data.info()
test_data.info()

# Check data types
print(train_data.dtypes)

# Displaying the column names of the dataset.
train_data.columns
test_data.columns

# Check the balance of the target column ('Churn')
train_data['Churn'].value_counts()
test_data['Churn'].value_counts()

# Display basic statistics
train_data.describe()
test_data.describe()

# Analyze customer loss rate with "International plan"
intl_plan_churn = train_data[train_data['International plan'] == 'Yes']['Churn'].mean()
no_intl_plan_churn = train_data[train_data['International plan'] == 'No']['Churn'].mean()
plt.bar(['International plan', 'No International plan'], [intl_plan_churn, no_intl_plan_churn])
plt.xlabel('International plan')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate with International plan')
plt.show()

# Analyze customer loss rate with "Voice mail plan"
voice_mail_plan_churn = train_data[train_data['Voice mail plan'] == 'Yes']['Churn'].mean()
no_voice_mail_plan_churn = train_data[train_data['Voice mail plan'] == 'No']['Churn'].mean()
plt.bar(['Voice mail plan', 'No Voice mail plan'], [voice_mail_plan_churn, no_voice_mail_plan_churn])
plt.xlabel('Voice mail plan')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate with Voice mail plan')
plt.show()

# Analyze customer loss rate based on "Number vmail messages"
vmail_messages_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
vmail_messages_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
vmail_messages_churn = []
for i in range(len(vmail_messages_bins) - 1):
    mask = (train_data['Number vmail messages'] >= vmail_messages_bins[i]) & (train_data['Number vmail messages'] < vmail_messages_bins[i+1])
    vmail_messages_churn.append(train_data[mask]['Churn'].mean())
plt.bar(vmail_messages_labels, vmail_messages_churn)
plt.xlabel('Number of voice messages')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Number vmail messages"')
plt.show()

# Analyze customer loss rate based on "Total day minutes"
day_minutes_bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
day_minutes_labels = ['0-100', '101-200', '201-300', '301-400', '401-500', '501-600', '601-700', '701-800', '801-900', '901-1000']
day_minutes_churn = []
for i in range(len(day_minutes_bins) - 1):
    mask = (train_data['Total day minutes'] >= day_minutes_bins[i]) & (train_data['Total day minutes'] < day_minutes_bins[i+1])
    day_minutes_churn.append(train_data[mask]['Churn'].mean())
plt.bar(day_minutes_labels, day_minutes_churn)
plt.xlabel('Total day minutes')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Total day minutes"')
plt.show()

# Analyze customer loss rate based on "Total day calls"
day_calls_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
day_calls_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
day_calls_churn = []
for i in range(len(day_calls_bins) - 1):
    mask = (train_data['Total day calls'] >= day_calls_bins[i]) & (train_data['Total day calls'] < day_calls_bins[i+1])
    day_calls_churn.append(train_data[mask]['Churn'].mean())
plt.bar(day_calls_labels, day_calls_churn)
plt.xlabel('Total day calls')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Total day calls')
plt.show()

# Analyze customer loss rate based on "Total day charge"
day_charge_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
day_charge_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
day_charge_churn = []
for i in range(len(day_charge_bins) - 1):
    mask = (train_data['Total day charge'] >= day_charge_bins[i]) & (train_data['Total day charge'] < day_charge_bins[i+1])
    day_charge_churn.append(train_data[mask]['Churn'].mean())
plt.bar(day_charge_labels, day_charge_churn)
plt.xlabel('Total day charge')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Total day charge"')
plt.show()

# Analyze customer loss rate based on "Total eve minutes"
eve_minutes_bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
eve_minutes_labels = ['0-100', '101-200', '201-300', '301-400', '401-500', '501-600', '601-700', '701-800', '801-900', '901-1000']
eve_minutes_churn = []
for i in range(len(eve_minutes_bins) - 1):
    mask = (train_data['Total eve minutes'] >= eve_minutes_bins[i]) & (train_data['Total eve minutes'] < eve_minutes_bins[i+1])
    eve_minutes_churn.append(train_data[mask]['Churn'].mean())
plt.bar(eve_minutes_labels, eve_minutes_churn)
plt.xlabel('Total eve minutes')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Total eve minutes"')
plt.show()

# Analyze customer loss rate is based on "Total eve calls"
eve_calls_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
eve_calls_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
eve_calls_churn = []
for i in range(len(eve_calls_bins) - 1):
    mask = (train_data['Total eve calls'] >= eve_calls_bins[i]) & (train_data['Total eve calls'] < eve_calls_bins[i+1])
    eve_calls_churn.append(train_data[mask]['Churn'].mean())
plt.bar(eve_calls_labels, eve_calls_churn)
plt.xlabel('Total eve calls')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate dbased on "Total eve calls"')
plt.show()

# Analyze customer loss rate is based on "Total eve charge"
eve_charge_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
eve_charge_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
eve_charge_churn = []
for i in range(len(eve_charge_bins) - 1):
    mask = (train_data['Total eve charge'] >= eve_charge_bins[i]) & (train_data['Total eve charge'] < eve_charge_bins[i+1])
    eve_charge_churn.append(train_data[mask]['Churn'].mean())
plt.bar(eve_charge_labels, eve_charge_churn)
plt.xlabel('Total eve charge')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Total eve charge"')
plt.show()

# Analyze customer loss rate is based on "Total night minutes"
night_minutes_bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
night_minutes_labels = ['0-100', '101-200', '201-300', '301-400', '401-500', '501-600', '601-700', '701-800', '801-900', '901-1000']
night_minutes_churn = []
for i in range(len(night_minutes_bins) - 1):
    mask = (train_data['Total night minutes'] >= night_minutes_bins[i]) & (train_data['Total night minutes'] < night_minutes_bins[i+1])
    night_minutes_churn.append(train_data[mask]['Churn'].mean())
plt.bar(night_minutes_labels, night_minutes_churn)
plt.xlabel('Total night minutes')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Total night minutes"')
plt.show()

# Analyze customer loss rate based on "Total night calls"
night_calls_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
night_calls_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
night_calls_churn = []
for i in range(len(night_calls_bins) - 1):
    mask = (train_data['Total night calls'] >= night_calls_bins[i]) & (train_data['Total night calls'] < night_calls_bins[i+1])
    night_calls_churn.append(train_data[mask]['Churn'].mean())
plt.bar(night_calls_labels, night_calls_churn)
plt.xlabel('Total night calls')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Total night calls"')
plt.show()

# Analyze customer loss rate is based on "Total night charge"
night_charge_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
night_charge_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50']
night_charge_churn = []
for i in range(len(night_charge_bins) - 1):
    mask = (train_data['Total night charge'] >= night_charge_bins[i]) & (train_data['Total night charge'] < night_charge_bins[i+1])
    night_charge_churn.append(train_data[mask]['Churn'].mean())
plt.bar(night_charge_labels, night_charge_churn)
plt.xlabel('Total night charge')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Total night charge"')
plt.show()

# Analyze customer loss rate is based on "Total intl minutes"
intl_minutes_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
intl_minutes_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
intl_minutes_churn = []
for i in range(len(intl_minutes_bins) - 1):
    mask = (train_data['Total intl minutes'] >= intl_minutes_bins[i]) & (train_data['Total intl minutes'] < intl_minutes_bins[i+1])
    intl_minutes_churn.append(train_data[mask]['Churn'].mean())
plt.bar(intl_minutes_labels, intl_minutes_churn)
plt.xlabel('Total intl minutes')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Total intl minutes"')
plt.show()

# Analyze customer loss rate based on "Total intl calls"
intl_calls_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
intl_calls_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50']
intl_calls_churn = []
for i in range(len(intl_calls_bins) - 1):
    mask = (train_data['Total intl calls'] >= intl_calls_bins[i]) & (train_data['Total intl calls'] < intl_calls_bins[i+1])
    intl_calls_churn.append(train_data[mask]['Churn'].mean())
plt.bar(intl_calls_labels, intl_calls_churn)
plt.xlabel('Total intl calls')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Total intl calls"')
plt.show()

# Analyze customer loss rate is based on "Total intl charge"
intl_charge_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
intl_charge_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
intl_charge_churn = []
for i in range(len(intl_charge_bins) - 1):
    mask = (train_data['Total intl charge'] >= intl_charge_bins[i]) & (train_data['Total intl charge'] < intl_charge_bins[i+1])
    intl_charge_churn.append(train_data[mask]['Churn'].mean())
plt.bar(intl_charge_labels, intl_charge_churn)
plt.xlabel('Total intl charge')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Total intl charge"')
plt.show()

# Analyze customer loss rate based on "Customer service calls"
service_calls_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
service_calls_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']
service_calls_churn = []

for i in range(len(service_calls_bins) - 1):
    mask = (train_data['Customer service calls'] >= service_calls_bins[i]) & (train_data['Customer service calls'] < service_calls_bins[i+1])
    churn_rate = train_data[mask]['Churn'].mean()
    service_calls_churn.append(churn_rate)
mask = (train_data['Customer service calls'] >= service_calls_bins[-1])
churn_rate = train_data[mask]['Churn'].mean()
service_calls_churn.append(churn_rate) 
plt.bar(service_calls_labels, service_calls_churn)
plt.xlabel('Customer service calls')
plt.ylabel('Customer loss rate')
plt.title('Customer loss rate based on "Customer service calls"')
plt.show()

### III: Decision Tree Classifier
#Encode categorical features
columns_to_encode = [ 'International plan', 'Voice mail plan', 'Churn']
encoder = LabelEncoder()
for column in columns_to_encode:
    train_data[column] = encoder.fit_transform(train_data[column])
for column in columns_to_encode:
    test_data[column] = encoder.fit_transform(test_data[column])


# Split data to training
X_train = train_data.drop(labels=['State', 'Account length', 'Area code','Churn'], axis=1)
y_train = train_data['Churn']

# Split data to testing
X_test = test_data.drop(labels=['State', 'Account length', 'Area code','Churn'], axis=1)
y_test = test_data['Churn']
# Fit the model on train data 
DT = DecisionTreeClassifier().fit(X_train,y_train)

# Predict on train 
train_preds3 = DT.predict(X_train)

# Predict on test
test_preds3 = DT.predict(X_test)
# Accuracy on train
print("")
print("Model accuracy on train is: ", accuracy_score(y_train, train_preds3))

# Accuracy on test
print("Model accuracy on test is: ", accuracy_score(y_test, test_preds3))

# We got good accuracy which means our model is performing quite well 
# ROC 
print("ROC score on train is: ", roc_auc_score(y_train, train_preds3))
print("ROC score on test is: ", roc_auc_score(y_test, test_preds3))
print('-'*50)

# Confusion matrix
print("Confusion_matrix train is: ", confusion_matrix(y_train, train_preds3))
print("Confusion_matrix test is: ", confusion_matrix(y_test, test_preds3))
print('Wrong predictions out of total')
print('-'*50)

# Wrong Predictions made.
print((y_test !=test_preds3).sum(),'/',((y_test == test_preds3).sum()+(y_test != test_preds3).sum()))
print('-'*50)

# Kappa Score
print('KappaScore is: ', metrics.cohen_kappa_score(y_test,test_preds3)) 
