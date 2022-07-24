#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn --upgrade --quiet')
get_ipython().system('pip install opendatasets --upgrade --quiet')
get_ipython().system('pip install pandas --quiet')
get_ipython().system('pip install plotly matplotlib seaborn --quiet')
get_ipython().system('pip install numpy --quiet')
get_ipython().system('pip install pyarrow --quiet')


# In[2]:


import opendatasets as od
import os
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib


# In[3]:


dataset_url = 'https://www.kaggle.com/jsphyg/weather-dataset-rattle-package'
od.download(dataset_url)


# In[4]:


data_dir = './weather-dataset-rattle-package'
os.listdir(data_dir)


# In[5]:


train_csv = data_dir + '/weatherAUS.csv'


# In[6]:


raw_df = pd.read_csv(train_csv)
raw_df


# In[7]:


raw_df.info()


# In[8]:


raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)


# In[9]:


px.histogram(raw_df, x='Location', title='Location vs. Rainy Days', color='RainToday')


# In[10]:


px.histogram(raw_df, 
             x='Temp3pm', 
             title='Temperature at 3 pm vs. Rain Tomorrow', 
             color='RainTomorrow')


# In[11]:


px.histogram(raw_df, 
             x='RainTomorrow', 
             color='RainToday', 
             title='Rain Tomorrow vs. Rain Today')


# In[12]:


px.scatter(raw_df.sample(2000), 
           title='Min Temp. vs Max Temp.',
           x='MinTemp', 
           y='MaxTemp', 
           color='RainToday')


# In[13]:


px.scatter(raw_df.sample(2000), 
           title='Temp (3 pm) vs. Humidity (3 pm)',
           x='Temp3pm',
           y='Humidity3pm',
           color='RainTomorrow')


# In[14]:


use_sample = False


# In[15]:


sample_fraction = 0.1
if use_sample:
    raw_df = raw_df.sample(frac=sample_fraction).copy()


# In[16]:


train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)


# In[17]:


print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)


# In[18]:


plt.title('No. of Rows per Year')
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year);


# In[19]:


year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]


# In[20]:


print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)


# In[21]:


train_df


# In[22]:


val_df


# In[23]:


test_df


# In[24]:


input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'


# In[25]:


print(input_cols)


# In[26]:


target_col


# In[27]:


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()\

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

train_inputs


# In[28]:


train_targets


# In[29]:


numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


# In[30]:


train_inputs[numeric_cols].describe()


# In[31]:


train_inputs[categorical_cols].nunique()


# In[32]:


imputer = SimpleImputer(strategy = 'mean')


# In[33]:


raw_df[numeric_cols].isna().sum()


# In[34]:


train_inputs[numeric_cols].isna().sum()


# In[35]:


imputer.fit(raw_df[numeric_cols])


# In[36]:


list(imputer.statistics_)


# In[37]:


train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])


# In[38]:


train_inputs[numeric_cols].isna().sum()


# In[39]:


raw_df[numeric_cols].describe()


# In[40]:


scaler = MinMaxScaler()


# In[41]:


scaler.fit(raw_df[numeric_cols])


# In[42]:


print('Minimum:')
list(scaler.data_min_)


# In[43]:


print('Maximum:')
list(scaler.data_max_)


# In[44]:


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


# In[45]:


train_inputs[numeric_cols].describe()


# In[46]:


raw_df[categorical_cols].nunique()


# In[47]:


encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[48]:


encoder.fit(raw_df[categorical_cols])


# In[49]:


encoder.categories_


# In[50]:


encoded_cols = list(encoder.get_feature_names(categorical_cols))
print(encoded_cols)


# In[51]:


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


# In[52]:


pd.set_option('display.max_columns', None)
test_inputs


# In[53]:


print('train_inputs:', train_inputs.shape)
print('train_targets:', train_targets.shape)
print('val_inputs:', val_inputs.shape)
print('val_targets:', val_targets.shape)
print('test_inputs:', test_inputs.shape)
print('test_targets:', test_targets.shape)


# In[54]:


train_inputs.to_parquet('train_inputs.parquet')
val_inputs.to_parquet('val_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')


# In[55]:


get_ipython().run_cell_magic('time', '', "pd.DataFrame(train_targets).to_parquet('train_targets.parquet')\npd.DataFrame(val_targets).to_parquet('val_targets.parquet')\npd.DataFrame(test_targets).to_parquet('test_targets.parquet')\n")


# In[56]:


get_ipython().run_cell_magic('time', '', "\ntrain_inputs = pd.read_parquet('train_inputs.parquet')\nval_inputs = pd.read_parquet('val_inputs.parquet')\ntest_inputs = pd.read_parquet('test_inputs.parquet')\n\ntrain_targets = pd.read_parquet('train_targets.parquet')[target_col]\nval_targets = pd.read_parquet('val_targets.parquet')[target_col]\ntest_targets = pd.read_parquet('test_targets.parquet')[target_col]\n")


# In[57]:


print('train_inputs:', train_inputs.shape)
print('train_targets:', train_targets.shape)
print('val_inputs:', val_inputs.shape)
print('val_targets:', val_targets.shape)
print('test_inputs:', test_inputs.shape)
print('test_targets:', test_targets.shape)


# In[58]:


val_inputs


# In[59]:


val_targets


# # Training a Logistic Regression Model
# 
# Logistic regression is a commonly used technique for solving binary classification problems. In a logistic regression model: 
# 
# - we take linear combination (or weighted sum of the input features) 
# - we apply the sigmoid function to the result to obtain a number between 0 and 1
# - this number represents the probability of the input being classified as "Yes"
# - instead of RMSE, the cross entropy loss function is used to evaluate the results
# 
# 
# Here's a visual summary of how a logistic regression model is structured ([source](http://datahacker.rs/005-pytorch-logistic-regression-in-pytorch/)):
# 
# 
# <img src="https://i.imgur.com/YMaMo5D.png" width="480">
# 
# The sigmoid function applied to the linear combination of inputs has the following formula:
# 
# <img src="https://i.imgur.com/sAVwvZP.png" width="400">
# 
# To train a logistic regression model, we can use the `LogisticRegression` class from Scikit-learn.

# In[60]:


model = LogisticRegression(solver='liblinear')


# In[61]:


model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)


# `model.fit` uses the following workflow for training the model ([source](https://www.deepnetts.com/blog/from-basic-machine-learning-to-deep-learning-in-5-minutes.html)):
# 
# 1. We initialize a model with random parameters (weights & biases).
# 2. We pass some inputs into the model to obtain predictions.
# 3. We compare the model's predictions with the actual targets using the loss function.  
# 4. We use an optimization technique (like least squares, gradient descent etc.) to reduce the loss by adjusting the weights & biases of the model
# 5. We repeat steps 1 to 4 till the predictions from the model are good enough.
# 
# 
# <img src="https://www.deepnetts.com/blog/wp-content/uploads/2019/02/SupervisedLearning.png" width="480">

# In[62]:


print(numeric_cols + encoded_cols)


# In[63]:


print(model.coef_.tolist())


# In[64]:


print(model.intercept_)


# ## Making Predictions and Evaluating the Model
# 
# We can now use the trained model to make predictions on the training, test 

# In[65]:


X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


# In[66]:


train_preds = model.predict(X_train)
train_preds


# In[67]:


train_targets


# In[68]:


train_probs = model.predict_proba(X_train)
train_probs


# In[69]:


model.classes_


# In[70]:


accuracy_score(train_targets, train_preds)


# The model achieves an accuracy of 85.1% on the training set. We can visualize the breakdown of correctly and incorrectly classified inputs using a confusion matrix.
# 
# <img src="https://i.imgur.com/UM28BCN.png" width="480">

# In[71]:


confusion_matrix(train_targets, train_preds, normalize='true')


# In[72]:


def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    
    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name));
    
    return preds


# In[73]:


train_preds = predict_and_plot(X_train, train_targets, 'Training')


# In[74]:


val_preds = predict_and_plot(X_val, val_targets, 'Validatiaon')


# In[75]:


test_preds = predict_and_plot(X_test, test_targets, 'Test')


# In[76]:


def random_guess(inputs):
    return np.random.choice(["No", "Yes"], len(inputs))


def all_no(inputs):
    return np.full(len(inputs), "No")


# In[77]:


accuracy_score(test_targets, random_guess(X_test))


# In[78]:


accuracy_score(test_targets, all_no(X_test))


# ## Making Predictions on a Single Input
# 
# Once the model has been trained to a satisfactory accuracy, it can be used to make predictions on new data. Consider the following dictionary containing data collected from the Katherine weather department today.

# In[79]:


new_input = {'Date': '2021-06-19',
             'Location': 'Katherine',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}


# In[80]:


new_input_df = pd.DataFrame([new_input])

new_input_df


# In[81]:


new_input_df[numeric_cols] = imputer.transform(new_input_df[numeric_cols])
new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])
new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols])


# In[82]:


X_new_input = new_input_df[numeric_cols + encoded_cols]
X_new_input


# In[83]:


prediction = model.predict(X_new_input)[0]
prediction


# In[84]:


prob = model.predict_proba(X_new_input)[0]
prob


# In[85]:


def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob


# In[86]:


new_input = {'Date': '2021-06-19',
             'Location': 'Launceston',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}


# In[87]:


predict_input(new_input)


# In[88]:


raw_df.Location.unique()


# In[89]:


aussie_rain = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}


# In[90]:


joblib.dump(aussie_rain, 'aussie_rain.joblib')


# In[91]:


aussie_rain2 = joblib.load('aussie_rain.joblib')


# In[92]:


test_preds2 = aussie_rain2['model'].predict(X_test)
accuracy_score(test_targets, test_preds2)


# ### Data Preprocessing

# In[93]:


import opendatasets as od
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Download the dataset
od.download('https://www.kaggle.com/jsphyg/weather-dataset-rattle-package')
raw_df = pd.read_csv('weather-dataset-rattle-package/weatherAUS.csv')
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# Create training, validation and test sets
year = pd.to_datetime(raw_df.Date).dt.year
train_df, val_df, test_df = raw_df[year < 2015], raw_df[year == 2015], raw_df[year > 2015]

# Create inputs and targets
input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'
train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()
test_inputs, test_targets = test_df[input_cols].copy(), test_df[target_col].copy()

# Identify numeric and categorical columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()[:-1]
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

# Impute missing numerical values
imputer = SimpleImputer(strategy = 'mean').fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# Scale numeric features
scaler = MinMaxScaler().fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(raw_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

# Save processed data to disk
train_inputs.to_parquet('train_inputs.parquet')
val_inputs.to_parquet('val_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')
pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
pd.DataFrame(val_targets).to_parquet('val_targets.parquet')
pd.DataFrame(test_targets).to_parquet('test_targets.parquet')

# Load processed data from disk
train_inputs = pd.read_parquet('train_inputs.parquet')
val_inputs = pd.read_parquet('val_inputs.parquet')
test_inputs = pd.read_parquet('test_inputs.parquet')
train_targets = pd.read_parquet('train_targets.parquet')[target_col]
val_targets = pd.read_parquet('val_targets.parquet')[target_col]
test_targets = pd.read_parquet('test_targets.parquet')[target_col]


# ### Model Training and Evaluation

# In[94]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Select the columns to be used for training/prediction
X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

# Create and train the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, train_targets)

# Generate predictions and probabilities
train_preds = model.predict(X_train)
train_probs = model.predict_proba(X_train)
accuracy_score(train_targets, train_preds)

# Helper function to predict, compute accuracy & plot confustion matrix
def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name));    
    return preds

# Evaluate on validation and test set
val_preds = predict_and_plot(X_val, val_targets, 'Validation')
test_preds = predict_and_plot(X_test, test_targets, 'Test')

# Save the trained model & load it back
aussie_rain = {'model': model, 'imputer': imputer, 'scaler': scaler, 'encoder': encoder,
               'input_cols': input_cols, 'target_col': target_col, 'numeric_cols': numeric_cols,
               'categorical_cols': categorical_cols, 'encoded_cols': encoded_cols}
joblib.dump(aussie_rain, 'aussie_rain.joblib')
aussie_rain2 = joblib.load('aussie_rain.joblib')


# ### Prediction on Single Inputs 

# In[95]:


def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

new_input = {'Date': '2021-06-19',
             'Location': 'Launceston',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}

predict_input(new_input)


# In[ ]:




