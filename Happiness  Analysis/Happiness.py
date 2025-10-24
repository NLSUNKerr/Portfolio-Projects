import numpy as np 
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

Data = pd.read_csv('/Users/skandermellah/Desktop/Data Analysis/Data.csv')
Happiness_Data = pd.read_excel('/Users/skandermellah/Desktop/Data Analysis/Happiness.xlsx')

Happiness_Data.columns =  ['Year','Country','Happiness_Score']
Data.rename (columns = {'country_name':'Country', 'year':'Year'}, inplace = True)
income_mapping = {
    'Low income': 0,
    'Lower middle income': 1,
    'Upper middle income': 2,
    'High income': 3
}
Data['income_group_code'] = Data['income_group'].map(income_mapping)
Data = Data.drop(columns=['income_group'])

Happiness_Data = Happiness_Data [(Happiness_Data ['Year'] >= 2011) & (Happiness_Data['Year'] <= 2020)]
Data = Data [(Data['Year'] >= 2011) & (Data['Year'] <= 2020)]

DataSet = Data.merge (Happiness_Data, on = ['Year', 'Country'], how ='inner')
DataSet = DataSet.sort_values(by='Country')
file_path = "/Users/skandermellah/Desktop/Data Analysis/final_macro_dataset3.csv"
DataSet.to_csv(file_path, index=False)
# OLS
df_2018 = DataSet [DataSet["Year"] == 2018]

Y = df_2018 [["Happiness_Score"]]
X = df_2018 [["gdp_usd", "inflation_rate", "unemployment_rate","fdi_pct_gdp", "healthcare_capacity_index", "digital_connectivity_index","governance_quality_index"]]

OLS = sm.OLS(Y, X, missing='drop')
resultsOLS = OLS.fit()
print(resultsOLS.summary())
#Preparing Data for ML techniques

Exclude = ['Year', 'Country', 'Happiness_Score', 'country_code', 'region', 'currency_unit', 'years_since_2000','years_since_century']

MLVariables = DataSet[[col for col in DataSet.columns if col not in exclude]].dropna()

Y_ML = DataSet["Happiness_Score"].loc[MLVariables.index]

Variables_train, Variables_test, y_train, y_test = train_test_split(
    MLVariables, Y_ML, test_size=0.2, random_state=42
)

scaler = StandardScaler()
Variables_train = scaler.fit_transform(Variables_train)
Variables_test = scaler.transform(Variables_test)
#Lasso

Lasso = LassoCV(cv=10, random_state=0, max_iter = 100000)
Lasso.fit(Variables_train,y_train)

y_pred = Lasso.predict(Variables_test)

print("Best alpha (lambda):", Lasso.alpha_)

print("Test MSE:", mean_squared_error(y_test, y_pred))

LassoResults = pd.DataFrame({
    'Variable': MLVariables.columns,
    'Lasso_Coefficient': Lasso.coef_
})

# Filter to only non-zero coefficients
LassoResults = LassoResults[LassoResults['Lasso_Coefficient'] != 0]

LassoResults = LassoResults.reindex(LassoResults['Lasso_Coefficient'].abs().sort_values(ascending=False).index)

# Display the result
print(LassoResults)

# Optional: Sort by absolute value of coefficient
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)
plt.xlabel("Actual Happiness Score")
plt.ylabel("Predicted Happiness Score")
plt.title("Actual vs. Predicted Happiness Score (Lasso Regression)")
plt.grid(True)
plt.show()

#Random Forest

Grid_Atributes = {'n_estimators':[100, 500, 1000],'max_features':['sqrt', 'log2' ] }

RandomForest = RandomForestRegressor(random_state=42, oob_score = True)

grid_search = GridSearchCV(estimator=RandomForest, param_grid=Grid_Atributes, cv=10,  scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)

grid_search.fit(Variables_train, y_train)

Best_RandomForest = grid_search.best_estimator_
y_pred = Best_RandomForest.predict(Variables_test)

MSE = mean_squared_error(y_test, y_pred)

print("MSE:", mse)

Importance_Matrix = pd.Series(Best_RandomForest.feature_importances_, index=MLVariables.columns)
Top_Features = Importance_Matrix.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
top_features.plot(kind='barh')
plt.xlabel("Feature Importance")
plt.title("Top 15 Important Features - Random Forest")
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)
plt.xlabel("Actual Happiness Score")
plt.ylabel("Predicted Happiness Score")
plt.title("Actual vs Predicted (Random Forest)")
plt.grid(True)
plt.show()
