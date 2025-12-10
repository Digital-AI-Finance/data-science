# Lesson Memories

This file captures key insights from each lesson for quick reference and future recall.

---

## L21: Linear Regression

**Problem:** How to measure systematic risk for portfolio construction?

**Solution:** CAPM beta via linear regression on stock vs market returns

**Key Formulas:**
- OLS: minimize sum of squared residuals
- Beta = Cov(stock, market) / Var(market)
- Prediction: y_hat = intercept + slope * x

**Code Pattern:**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
beta = model.coef_[0]
alpha = model.intercept_
```

**Finance Insight:**
- Beta > 1: Aggressive (TSLA ~1.8) - amplifies market moves
- Beta = 1: Market-tracking (index funds)
- Beta < 1: Defensive (JNJ ~0.7) - dampens volatility
- Alpha > 0: Outperformance after risk adjustment

**Assumptions to Check:**
- Linearity (relationship is linear)
- Independence (observations don't influence each other)
- Homoscedasticity (constant error variance)
- Normality (residuals normally distributed)

**Builds On:** L13-L16 (statistics), L05-L06 (DataFrames)

**Leads To:** L22 (regularization), L23 (metrics), L24 (factor models)

---

## L12: Time Series

**Problem:** How to analyze and manipulate financial time series data?

**Solution:** Pandas datetime indexing, resampling, and rolling window calculations

**Key Formulas:**
- Returns: pct_change() = (P_t - P_{t-1}) / P_{t-1}
- Rolling mean: rolling(window).mean()
- Resampling: resample('M').last() for monthly

**Code Pattern:**
```python
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df['returns'] = df['price'].pct_change()
df['ma_20'] = df['price'].rolling(20).mean()
monthly = df.resample('M').agg({'price': 'last', 'volume': 'sum'})
```

**Finance Insight:**
- Resampling: Convert daily to weekly/monthly for longer-term analysis
- Rolling windows: Moving averages smooth noise, reveal trends
- Shift/lag: Create features for time series forecasting
- Autocorrelation: Measure persistence in returns

**Builds On:** L05-L06 (DataFrames), L08 (operations)

**Leads To:** L13-L16 (statistics), L21 (regression)

---

## L17: Matplotlib Basics

**Problem:** How to create professional financial visualizations?

**Solution:** Matplotlib's object-oriented API with customization

**Key Patterns:**
- Line plots: Price series, cumulative returns
- Bar charts: Sector comparison, returns by period
- Histograms: Return distributions
- Scatter plots: Risk vs return, correlations

**Code Pattern:**
```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dates, prices, color='#0066CC', linewidth=2, label='Price')
ax.set_title('Stock Price', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
```

**Finance Insight:**
- Price + volume charts: Standard for technical analysis
- Drawdown charts: Visualize risk and recovery periods
- Rolling Sharpe: Monitor risk-adjusted performance over time
- Cumulative returns: Compare strategy performance

**Builds On:** L05-L12 (data manipulation)

**Leads To:** L18 (Seaborn), L19 (multi-panel), L20 (storytelling)

---

## L22: Regularization

**Problem:** How to prevent overfitting when you have many features?

**Solution:** Ridge (L2) and Lasso (L1) penalties shrink coefficients

**Key Formulas:**
- Ridge: minimize ||y - Xb||^2 + lambda * ||b||^2
- Lasso: minimize ||y - Xb||^2 + lambda * ||b||_1
- Elastic Net: combines both penalties

**Code Pattern:**
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X_train, y_train)
```

**Finance Insight:**
- Ridge: Keep all factors, shrink coefficients (factor models)
- Lasso: Automatic feature selection (find key predictors)
- Cross-validation: Use to find optimal lambda
- Coefficient paths: Visualize shrinkage as lambda increases

**Builds On:** L21 (linear regression)

**Leads To:** L23 (metrics), L24 (factor models)

---

## L23: Regression Metrics

**Problem:** How to evaluate and compare regression model performance?

**Solution:** MSE, RMSE, MAE, R-squared with proper interpretation

**Key Formulas:**
- MSE = (1/n) * sum((y - y_hat)^2)
- RMSE = sqrt(MSE) - same units as y
- MAE = (1/n) * sum(|y - y_hat|) - robust to outliers
- R^2 = 1 - SS_res / SS_tot - proportion of variance explained

**Code Pattern:**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

**Finance Insight:**
- RMSE vs MAE: If RMSE >> MAE, you have outliers
- R^2 in finance: Often low (0.01-0.05) is still useful
- Time series CV: Use expanding or rolling windows
- Residual analysis: Check for patterns, heteroscedasticity

**Builds On:** L21-L22 (regression, regularization)

**Leads To:** L24 (factor models), L25 (classification)

---

## L24: Factor Models

**Problem:** How to decompose stock returns into systematic risk factors?

**Solution:** Fama-French multi-factor models via multiple regression

**Key Formulas:**
- CAPM: R_i - R_f = alpha + beta * (R_m - R_f)
- FF3: + SMB (size) + HML (value)
- FF5: + RMW (profitability) + CMA (investment)

**Code Pattern:**
```python
# Load Fama-French factors
X = ff_data[['Mkt-RF', 'SMB', 'HML']]
y = stock_returns - ff_data['RF']

model = LinearRegression().fit(X, y)
alpha = model.intercept_  # Risk-adjusted return
betas = model.coef_  # Factor exposures
```

**Finance Insight:**
- Alpha: Manager skill after accounting for factor exposure
- Factor loadings: Stock's sensitivity to each factor
- SMB positive: Tilts toward small caps
- HML positive: Tilts toward value stocks

**Builds On:** L21-L23 (regression, regularization, metrics)

**Leads To:** L25-L28 (classification methods)

---

*Template for future lessons:*

```markdown
## LXX: Topic Name

**Problem:** [One sentence problem statement]

**Solution:** [One sentence solution]

**Key Formulas:**
- [Formula 1]
- [Formula 2]

**Code Pattern:**
[Essential code snippet]

**Finance Insight:**
- [Key insight 1]
- [Key insight 2]

**Builds On:** [Prerequisites]

**Leads To:** [Next topics]
```
