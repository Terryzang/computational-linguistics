import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap


pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))

# Read data
alpha = 169.59
train = 'Study1_agency'
test = 'Study2_agency'
df1 = pd.read_csv(f'{train}.csv')
df2 = pd.read_csv(f'{test}.csv')

X1 = df1.iloc[:, :-1]
Y1 = df1.iloc[:, -1]
X2 = df2.iloc[:, :-1]
Y2 = df2.iloc[:, -1]

# 2. standard scaler
scaler = StandardScaler()
X_scaled1 = scaler.fit_transform(X1)
X_scaled2 = scaler.transform(X2)

# 3. Training Ridge Regression
ridge = Ridge(alpha=alpha)
ridge.fit(X_scaled1, Y1)

# 4. Testing
Y_pred = ridge.predict(X_scaled2)

# 5. Evaluate the performance of the model
r2 = r2_score(Y2, Y_pred)
mse = mean_squared_error(Y2, Y_pred)
mae = mean_absolute_error(Y2, Y_pred)
pearson_r, pearson_p = pearsonr(Y2, Y_pred)


print(f"\nR²: {r2:.5f}")
print(f"MSE: {mse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"Pearson r: {pearson_r:.5f} (p = {pearson_p:.5f})")

# 6. Using SHAP for explanation
explainer = shap.LinearExplainer(ridge, X_scaled1, feature_perturbation="interventional")

# 5. Calculate the SHAP values
shap_values = explainer.shap_values(X_scaled2)
feature_names = ['total words', 'First-Person Pronoun', 'Negation Word', 'Positive Word', 'Negative Word',
                     'Tokens per sentence', 'Dependency tree depth', 'Attributives', 'Adverbials',
                     'Positive emotion', 'Negative emotion', 'Neutral emotion']
shap.summary_plot(shap_values, X_scaled2, feature_names=feature_names)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])

print("\n[SHAP Importance ranking]:")
for name, score in importance:
    print(f"{name}: {score:.3f}")