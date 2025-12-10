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
