# Big Mart Sales Prediction: A Regression Problem

## Introduction

Sales prediction is a crucial aspect of retail business management, and accurate forecasting allows businesses to optimize inventory, plan promotions, and improve overall efficiency. In this project, we will focus on predicting sales per product in a retail store, particularly using data from Big Mart.

## Problem Statement

The objective is to predict the sales of each product in the store, considering various features such as product characteristics, store information, and historical sales data. The target label for our regression problem is `Item_Outlet_Sales`.

## Data and Libraries

Let's start by importing the necessary libraries for our analysis. The code utilizes `pyforest` for lazy imports, covering a wide range of commonly used data science libraries like Pandas, NumPy, Matplotlib, Seaborn, and more.

```python
from pyforest import *
lazy_imports()
# ... (import statements)
import warnings
warnings.filterwarnings("ignore")

# Reading the training and testing data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
```

## Exploratory Data Analysis (EDA)

### Dataset Overview

Let's begin with a brief overview of the dataset:

```python
print("The size of training data", train_data.shape)
print("The shape of testing data", test_data.shape)
train_data.describe().T
```

The training data has 8523 entries and 12 features, while the testing data contains 5681 entries and 11 features.

### Missing Values

Checking for missing values in the dataset:

```python
train_data.isnull().sum()
test_data.isnull().sum()
```

There are missing values in the `Item_Weight` and `Outlet_Size` columns.

### Data Visualization

Exploring the data distribution and patterns using visualizations:

```python
# Distribution plots
for i in train_data.describe().columns:
    sns.distplot(train_data[i].dropna())
    plt.show()

# Box plots
for i in train_data.describe().columns:
    sns.boxplot(train_data[i].dropna())
    plt.show()

# Count plot for item types
plt.figure(figsize=(15, 10))
sns.countplot(x='Item_Type', data=train_data)
plt.xticks(rotation=90)
plt.show()
```

Visualizations provide insights into the distribution of numerical features and the count of different item types.

### Feature Analysis

Analyzing features such as `Outlet_Size`, `Outlet_Location_Type`, and `Outlet_Type`:

```python
plt.figure(figsize=(10, 8))
sns.countplot(x='Outlet_Size', data=train_data)

train_data.Outlet_Size.value_counts()

plt.figure(figsize=(10, 8))
sns.countplot(x='Outlet_Location_Type', data=train_data)

train_data.Outlet_Location_Type.value_counts()

plt.figure(figsize=(10, 8))
sns.countplot(x='Outlet_Type', data=train_data)

train_data.Outlet_Type.value_counts()
```

### Correlation Analysis

Understanding the correlation between features:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.xlabel("Item Weight")
plt.ylabel("Item Outlet sales")
plt.title("Item Weights Vs Outlet Sales")
sns.scatterplot(x="Item_Weight", y="Item_Outlet_Sales", hue="Item_Type", size="Item_Weight", data=train_data)

plt.figure(figsize=(12, 10))
plt.xlabel("Item Visibility")
plt.ylabel("Maximum Retail Price")
plt.title("Item Visibility Vs Maximum Retail Price")
plt.plot(train_data["Item_Visibility"], train_data["Item_MRP"], ".", alpha=0.3)
plt.show()

plt.figure(figsize=(10, 8))
plt.xlabel = "Item Visibility"
plt.ylabel = "Item Outlet Sales"
plt.title = "Item Visibility Vs Item Outlet Sales"
sns.scatterplot(x="Item_Visibility", y="Item_Outlet_Sales", hue="Item_Type", size="Item_Weight", data=train_data)
```

### Data Preprocessing

Handling missing values and feature engineering:

```python
# Replacing missing values in Item Weight
Item_weight_mean = df['Item_Weight'].mean()
df['Item_Weight'] = df['Item_Weight'].replace(np.nan, Item_weight_mean)

# Replacing missing values in Outlet Size
df['Outlet_Size'].fillna('Medium', inplace=True)

# Handling zero values in Item Visibility
df['Item_Visibility'].fillna(df['Item_Visibility'].median(), inplace=True)

# Creating a new feature - Outlet Years
df['Outlet_Years'] = 2009 - df['Outlet_Establishment_Year']

# Categorizing item types
df['New_Item_type'] = df['Item_Identifier'].apply(lambda x: x[0:2])
df['New_Item_type'] = df['New_Item_type'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks

'})

# Label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'New_Item_type', 'Outlet_Type']
le = LabelEncoder()

for i in var_mod:
    df[i] = le.fit_transform(df[i])
```

## Model Building

Now, let's build regression models using linear regression, decision tree regression, random forest regression, and XGBoost regression:

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Preparing data
X = df.drop(columns=['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Identifier'])
y = df['Item_Outlet_Sales']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Decision Tree Regression
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Random Forest Regression
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# XGBoost Regression
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
```

## Model Evaluation

Evaluating the models using Mean Squared Error (MSE):

```python
print("Linear Regression MSE:", mean_squared_error(y_test, lr_pred))
print("Decision Tree Regression MSE:", mean_squared_error(y_test, dt_pred))
print("Random Forest Regression MSE:", mean_squared_error(y_test, rf_pred))
print("XGBoost Regression MSE:", mean_squared_error(y_test, xgb_pred))
```

## Conclusion

In this project, we explored the Big Mart sales dataset, performed extensive exploratory data analysis, and built regression models to predict item outlet sales. The models were evaluated using Mean Squared Error, providing insights into their performance. Further improvements can be made through hyperparameter tuning, ensemble methods, and refining data preprocessing steps.

---
