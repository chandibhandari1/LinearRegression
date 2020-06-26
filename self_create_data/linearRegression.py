'''
Here we use our generated data:
This example is just to give concepts to the students how we can transform the data and process
as we have done in training process: here lebel encoding and One-hot encoding
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import pickle
import sys

# read the data
df =pd.read_excel("sales_data_nan.xlsx")
print(df.head())
# understanding the data
print('shape of the data is ',df.shape)


# since the productId is int64 which should be categorical
df['productId']=df['productId'].astype('category')
print(df.dtypes)

# Checking for null values
print('There are null values in the following columns: \n', df.isnull().sum())

# For now lets replace the null values with mean, median and mode
df.sales1.fillna(df.sales1.mean(), inplace =True)
df.sales2.fillna(df.sales2.median(), inplace =True)
df.sales3.fillna(df.sales3.mean(), inplace =True)
df.sales4.fillna(df.sales4.mean(), inplace =True)
df.deman_level.fillna(df.deman_level.value_counts().index[0], inplace=True)
df.best_seller.fillna(df.best_seller.mode().iloc[0], inplace=True)

# Making sure to check the null values
print('There are null values in the following columns: \n', df.isnull().sum())

# Handling the categorical variables: Lable encoding fo deman_level and one_hot encoding for best_seller
label_en = LabelEncoder()
# fit label encoder
label_en.fit(df.deman_level)
df['deman_level'] =label_en.transform(df.deman_level)

# # OR Directly Transform
# df['deman_level'] =label_en.fit_transform(df.deman_level)


#%%%%%%%%%%%%%%%%%%%%%%% Pickle the label encoder
filehandler = open("le.obj","wb")
pickle.dump(label_en,filehandler)
filehandler.close()

# Now drop the productId column
df.drop(['productId'],axis=1, inplace=True)
print(df.head().T)
# one hot encoding for best seller
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df[['best_seller']])
enc_df = pd.DataFrame(enc.transform(df[['best_seller']]).toarray(), columns=['A', 'D', 'L','M'])
# join with main data on key values
pro_data= df.join(enc_df)
pro_data.drop(['best_seller'], axis=1, inplace = True)
print(pro_data.head().T)

# now pickling one-hot encoding
with open('one_hot.pkl', 'wb') as output_file:
    pickle.dump(enc, output_file)

# OR Directly transform using one hot encoding by get dummies
# pd_get dummies for easy methond
#pro_data = pd.get_dummies(df, prefix='bs')

print(pro_data.head())

# Segregation of target and predictors
X = pro_data[pro_data.loc[:,pro_data.columns!='target'].columns]
y =pro_data['target']

# Train and test split
xtrain, xtest,ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

# modeling
linReg = LinearRegression()
linReg.fit(xtrain, ytrain)

# prediction
prediction_train = linReg.predict(xtrain)
prediction_test = linReg.predict(xtest)


# Get the mean square error
MSE = mean_squared_error(ytest, prediction_test)
print(f'Mean square Error: \n{MSE}')

# Checking the R-square error
r_square = r2_score(ytest, prediction_test)
print('R -square value: {}'.format(r_square))

# accuracy
print(linReg.score(xtrain, ytrain)*100, '% Train Accuracy')
print(linReg.score(xtest, ytest)*100, '% Test Accuracy')
# Now Visualizing the result with actual values
plt.plot(prediction_test,'.' ,ytest, '*')
plt.title('plot the training with train_pred')
plt.xlabel('pred')
plt.ylabel('ytest')
plt.show()


# dump the model for API call
pickle.dump(linReg, open('model_linReg.pkl','wb'))


####Testing the model is working or not

# Lets load the label encoder
file = open("le.obj",'rb')
label_en_loaded = pickle.load(file)
file.close()

pred_data =[253.0  ,  28.0 ,  309.0,   602.0, 'veryLow', 'A']
# Let's encode the data
print(label_en_loaded.transform(['veryLow']))
pred_data[4] =label_en_loaded.transform([pred_data[4]])
# application of one encoding
with open("one_hot.pkl",'rb') as f:
    oneHot_en_loaded = pickle.load(f)
#
print('This is one:',oneHot_en_loaded.transform([['A']]).toarray())
arr = oneHot_en_loaded.transform([[pred_data[5]]]).toarray()
print(arr)
pred_data.remove('A')
pred_data[5:5]=arr[0]
print(pred_data)

# lets load the model to check it is working or not
lr_model = pickle.load(open('model_linReg.pkl', 'rb'))

pr = lr_model.predict([pred_data])
print('predicted value: ', pr)
