import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('housing-barcelona.csv')

df['price'] = df['price'].astype(str).str.replace('€', '', regex=False)
df['price'] = df['price'].str.replace('.', '', regex=False).astype(float)

df['room'] = df['room'].astype(str).str.extract('(\d+)').astype(float)
df['space'] = df['space'].astype(str).str.extract('(\d+)').astype(float)

df['room'] = df['room'].fillna(df['room'].median())
df['space'] = df['space'].fillna(df['space'].mean())
df['price'] = df['price'].fillna(df['price'].median())

if df['subarea'].isnull().any():
    df['subarea'] = df['subarea'].fillna(df['subarea'].mode()[0])

if 'name' in df.columns:
    df = df.drop(columns=['name'])

le = LabelEncoder()
df['subarea'] = le.fit_transform(df['subarea'].astype(str))

df = df.dropna(subset=['price'])

X = df[['room', 'space', 'subarea']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

x_coeff = model.coef_[0]
y_coeff = model.coef_[1]
z_coeff = model.coef_[2]
constanta = model.intercept_

print(f"x (room): {x_coeff}")
print(f"y (space): {y_coeff}")
print(f"z (subarea): {z_coeff}")
print(f"Constant (b): {constanta}")
