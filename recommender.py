import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('investment_data_large.csv')
print(df.head(2))

dataset_features = ['Name', 'Email', 'Risk Tolerance', 'Investment Timeline (Years)',
'Financial Goals', 'Investment Type', 'Expected Return (%)', 'Risk Level'] 

df['Risk Tolerance'] = df['Risk Tolerance'].map({'Low' : 0, 'Medium' : 1, 'High' : 2})
df['Risk Level'] = df['Risk Level'].map({'Low' : 0, 'Medium' : 1, 'High' : 2})

df['Financial Goals'] = df['Financial Goals'].map({
    'Wealth Accumulation' : 0,
    "Children's Education" : 1,
    'Buying a House' : 2,
    'Retirement Savings' : 3,
    'Travel Fund' : 4,
    'Emergency Fund' : 5
})

df.rename(columns={'Expected Return (%)' : 'Expected Return'}, inplace=True)

df.rename(columns={'Investment Timeline (Years)' : 'Investment Timeline'}, inplace=True)

X = df[['Risk Level', 'Investment Timeline', 'Financial Goals']]
y = df['Expected Return']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

def recommend_stocks(risk, timeline, goals):
   risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
   goals_mapping = {'Wealth Accumulation': 0, "Children's Education": 1, 'Buying a House': 2,
                    'Retirement Savings' : 3, 'Travel Fund' : 4, 'Emergency Fund' : 5}
   
   input_data = pd.DataFrame({
       'Risk Level': [risk_mapping[risk]],
        'Investment Timeline': [timeline],
        'Financial Goals': [goals_mapping[goals]]
   })

   predicted_return = model.predict(input_data)
   return f"Recommended Expected Return: {predicted_return[0]:.2f}%"

predicted_return = recommend_stocks('Low', 6 , 'Emergency Fund')
print(predicted_return.strip('%'))