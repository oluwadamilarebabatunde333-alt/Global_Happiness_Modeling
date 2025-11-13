#Healthy Lifestyle Project Work

import pandas as pd
df = pd.read_csv(r"C:\Users\user\Downloads\Healthy-Lifestyle (2021).csv")
print(df)
print("shape:", df.shape)
print("columns:",df.columns.tolist())
print(df.isnull().sum())
df = df.dropna(subset=["Sunshine hours (City)","Annual avg. hours worked"])
print(df.isnull().sum())
df['Obesity levels (Country)']= ( #cleaning the obesity column
    df['Obesity levels (Country)']
     .astype(str)
     .str.replace("%","",regex=False)
     .str.strip()
    .astype(float)
     )
df["Cost of a bottle of water (City)"] = ( #cleaning cost of bottle water
    df["Cost of a bottle of water (City)"]
    .astype(str)
    .str.replace("Â","",regex=True)
    .str.replace("£", "", regex=True)
    .str.strip()
    .astype(float)
)
df["Cost of a monthly gym membership (City)"] = (
    df["Cost of a monthly gym membership (City)"]
    .astype(str)
    .str.replace("Â","",regex=False)
    .str.replace("£","",regex=False)
    .str.strip()
    .astype(float)
)
print(df["Cost of a monthly gym membership (City)"])
print(df.isnull().sum())
print(df.info())
df['Pollution (Index score)  (City)'] = (
    df['Pollution (Index score)  (City)']
    .astype(str)
    .str.replace(",","", regex = False)
    .replace("-",None)
    .str.strip()
    .astype(float)
)
df["Annual avg. hours worked"]=(
    df["Annual avg. hours worked"]
    .astype(str)
    .str.replace(",","")
    .replace("-",None)
    .str.strip()
    .astype(float)
)
print(df.info())
df["Annual avg. hours worked"] = ( #filling NAn with mean value
    df["Annual avg. hours worked"]
    .fillna(df["Annual avg. hours worked"].mean())
)
df['Pollution (Index score)  (City)'] = (
    df['Pollution (Index score)  (City)']
    .fillna(df['Pollution (Index score)  (City)'].mean())
)
print(df['Pollution (Index score)  (City)'])
print(df["Annual avg. hours worked"])
print(df.info())
print(df.isnull().sum())
print(df.describe())

correlation_matrix = (df.select_dtypes #relationship between column variables
(include=["number"]).corr())
print(correlation_matrix)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(
    correlation_matrix,
    annot = True,
    cmap = "coolwarm_r",
    fmt="2f",
    linewidths=0.5,
    annot_kws={"size":7}
)
plt.title("Correlation Heatmap of Healthy Lifestyles", fontsize = 14, pad  = 12)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()
cols = [
"Sunshine hours (City)",
    "Cost of a bottle of water (City)",
    "Obesity levels (Country)",
    "Life expectancy (years)  (Country)",
    "Pollution (Index score)  (City)",
    "Annual avg. hours worked",
    "Cost of a monthly gym membership (City)",
    "Happiness levels (Country)"
]
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
for i in range(len(cols)):
    plt.subplot(3,3,i+1)
    sns.histplot(df[cols[i]],kde=True)
    plt.title(cols[i])
    plt.tight_layout(pad=5.0)
plt.suptitle("Distribution of Healthy Lifestyle Indicators", fontsize=14)
plt.show()

import matplotlib.pyplot as plt #visualization using scatter plots
import seaborn as sns

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.scatter(df["Pollution (Index score)  (City)"],
     df["Happiness levels (Country)"], color="coral")
plt.xlabel("Pollution (Index score)  (City)")
plt.ylabel("Happiness levels (Country)")
plt.title("Pollution vs Happiness", fontsize = 10)


plt.subplot(1,3,2)
plt.scatter(df["Sunshine hours (City)"],
    df["Happiness levels (Country)"],color="goldenrod")
plt.xlabel("Sunshine hours (City)")
plt.ylabel("Happiness levels (Country)")
plt.title("Sunshine hours vs Happiness level", fontsize=10)

plt.subplot(1,3,3)
plt.scatter(df["Cost of a monthly gym membership (City)"],
    df["Happiness levels (Country)"], color="red")
plt.xlabel("Cost of a monthly gym membership (City)")
plt.ylabel("Happiness levels (Country)")
plt.title("Cost of Monthly Gym vs Happiness Levels",fontsize=10)
plt.tight_layout()
plt.show()

import pandas as pd     #Model Development
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

x = df.drop(columns=["Happiness levels (Country)", "City"])
y = df["Happiness levels (Country)"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 42)
print(x_train,x_test,y_train,y_test)

scaler = StandardScaler() #scaling the features
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

lin_reg = LinearRegression() #Linear regression
lin_reg.fit(x_train_scaled, y_train)
ridge = Ridge(alpha = 0.1) #Ridge regression
ridge.fit(x_train_scaled, y_train)
lasso = Lasso(alpha = 0.1) #Lasso regression
lasso.fit(x_train_scaled, y_train)

y_pred_lin = lin_reg.predict(x_test_scaled) #making predictions
y_pred_ridge = ridge.predict(x_test_scaled)
y_pred_lasso = lasso.predict(x_test_scaled)

def evaluate_model(y_test,y_pred, model_name):
    print(f"\n {model_name} Performance:")
    print("MAE:", round (mean_absolute_error(y_test,y_pred),4))
    print("MSE:", round(mean_squared_error(y_test, y_pred),4))
    print("R^2 Score:", r2_score(y_test, y_pred))

evaluate_model(y_test, y_pred_lin, "Linear Regression")
evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
evaluate_model(y_test, y_pred_lasso, "Lasso Regression")

import matplotlib.pyplot as plt
models = ["Linear","Ridge","Lasso"]
r2_scores = [
    r2_score(y_test, y_pred_lin),
    r2_score(y_test, y_pred_ridge),
    r2_score(y_test, y_pred_lasso)
]
plt.bar(models, r2_scores, color = ["skyblue", "salmon", "lightgreen"])
plt.ylabel("R2 Scores")
plt.title("Model Comparison")
plt.show()

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor #using other models
from sklearn.svm import SVR

rf = RandomForestRegressor(random_state=42) #random forest
rf.fit(x_train_scaled, y_train)
gbr = GradientBoostingRegressor(random_state=42) #gradientboostingregressor
gbr.fit(x_train_scaled,y_train)
svr = SVR(kernel="rbf") #support vector machine
svr.fit(x_train_scaled, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
models = {
    "Linear Regression": lin_reg,
    "Ridge Regression": ridge,
    "Lasso Regression": lasso,
    "Random Forest": rf,
    "Gradient Boosting":gbr,
    "Support Vector Regressor":svr,
}
for name, model in models.items():
    y_pred = model.predict(x_test_scaled)
    print(f"\n {name} Performance:")
    print("MAE:", round (mean_absolute_error(y_test,y_pred),4))
    print("MSE:", round (mean_squared_error (y_test, y_pred),4))
    print("R^2 Score:", round (r2_score(y_test,y_pred),4))

import matplotlib.pyplot as plt #R_2 Plot
r2_scores = [r2_score (y_test, model.predict(x_test_scaled)) for model in models.values()]
plt.figure(figsize=(8,5))
plt.bar(models.keys(), r2_scores, color=["skyblue","salmon","lightgreen","violet","gold","teal"])
plt.xticks(rotation = 45,  ha = "right")
plt.ylabel("R^2_score")
plt.title("Model Performance Comparison")

plt.tight_layout()
plt.show()

import numpy as np
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.bar(range(len(importances)), importances[indices], color="skyblue")
plt.xticks(range(len(importances)), x.columns[indices], rotation = 45, ha = "right")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()











