import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
from sklearn.model_selection import KFold

# Reading Data
df = pd.read_csv(f'Study1_agency.csv')
alphas = np.logspace(2.229, 2.2294, 10)
print(df.describe().T)

# todo Calculate the correlation coefficient matrix and the p-value matrix
def corr_sig(df):
    cols = df.columns
    n = len(cols)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
    p_matrix = pd.DataFrame(np.ones((n, n)), columns=cols, index=cols)

    for i in range(n):
        for j in range(i, n):
            corr, p = pearsonr(df[cols[i]], df[cols[j]])
            corr_matrix.iloc[i, j] = corr
            corr_matrix.iloc[j, i] = corr
            p_matrix.iloc[i, j] = p
            p_matrix.iloc[j, i] = p
    return corr_matrix, p_matrix

# Create annotations with prominent marks
def corr_star(corr_matrix, p_matrix):
    stars = p_matrix.copy()
    stars[:] = ''
    stars[p_matrix < 0.001] = '***'
    stars[(p_matrix < 0.01) & (p_matrix >= 0.001)] = '**'
    stars[(p_matrix < 0.05) & (p_matrix >= 0.01)] = '*'

    annot = corr_matrix.round(3).astype(str) + stars
    # Set the values on the main diagonal to empty strings
    for i in range(len(corr_matrix)):
        annot.iloc[i, i] = ''
    return annot

# using DataFrame df
corr, pvals = corr_sig(df)
annot = corr_star(corr, pvals)
# Use an upper triangular mask to cover the main diagonal and the lower part.
mask = np.triu(np.ones(corr.shape), k=0)  # Upper
# mask = np.tril(np.ones(corr.shape), k=0)  # Lower

# Visualization
plt.figure(figsize=(16, 10))
sns.heatmap(
    corr,
    mask=mask,
    annot=annot,
    fmt='',
    cmap='RdBu_r',
    vmin=-1, vmax=1,
    cbar=True,
    annot_kws={"size": 10}
)
plt.title('Correlation Matrix with Significance')
plt.tight_layout()
plt.savefig(f'Heatmap.jpg', dpi=300)
plt.show()


# todo Multivariate Linear Regression
model = smf.ols('SE~feature1+feature2+feature3+feature4+feature5+feature8+feature10'
                '+feature15+feature16+feature19+feature20+feature21', data=df)

model = model.fit()
print(model.summary())

# todo Calculate the VIF (variance inflation factor)
def VIF_calculate(df_all, y_name):
    x_cols = df.columns.to_list()
    x_cols.remove(y_name)

    def vif(df_exog, exog_name):
        exog_use = list(df_exog.columns)
        exog_use.remove(exog_name)
        model = smf.ols(f"{exog_name}~{'+'.join(list(exog_use))}", data=df_exog).fit()
        return 1. / (1. - model.rsquared)

    df_vif = pd.DataFrame()
    for x in x_cols:
        df_vif.loc['VIF', x] = vif(df_all[x_cols], x)

    df_vif.loc['tolerance'] = 1 / df_vif.loc['VIF']
    df_vif = df_vif.T.sort_values('VIF', ascending=False)
    df_vif.loc['mean_vif'] = df_vif.mean()
    return df_vif

print(VIF_calculate(df, 'SE'))

# todo ridge regression
# 1. Set features and labels
y = df['SE']
X = df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5',  'feature8', 'feature10',
        'feature15', 'feature16', 'feature19', 'feature20', 'feature21']]

# 2. Standardized features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# # 3. Use cross-validation to find the optimal alpha (the adjustable range)
ridge_cv = RidgeCV(alphas=alphas, store_cv_values=False, cv=KFold(n_splits=10, shuffle=True, random_state=42))
ridge_cv.fit(X_scaled, y)

# 4. Printing result
print(f"Best alpha: {ridge_cv.alpha_:.3f}")
print(f"R² score: {ridge_cv.score(X_scaled, y):.3f}")
print(f"Coefficients:")
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': ridge_cv.coef_
})
print(coef_df)

# # 5. If you need to assess the prediction error:
# y_pred = ridge_cv.predict(X_scaled)
# mse = mean_squared_error(y, y_pred)
# print(f"Mean Squared Error: {mse:.3f}")

# 6. Visualized alpha-R map
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

alphas = np.logspace(-3, 4, 100)
r2_scores = []

for a in alphas:
    ridge = Ridge(alpha=a)
    scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring='r2')
    r2_scores.append(scores.mean())

plt.figure(figsize=(8,5),dpi=256)
plt.plot(alphas, r2_scores)
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Cross-validated R²')
plt.title('Alpha vs R² (Cross-Validation)')
plt.axvline(x=ridge_cv.alpha_, color='red', linestyle='--', label=f'Best alpha={ridge_cv.alpha_:.2f}')
plt.legend()
plt.tight_layout()
plt.show()