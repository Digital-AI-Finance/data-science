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

## L25: Logistic Regression

**Problem:** How to predict binary outcomes (default/no default, buy/sell)?

**Solution:** Logistic regression outputs probabilities via sigmoid function

**Key Formulas:**
- Sigmoid: P(y=1) = 1 / (1 + exp(-z))
- z = b0 + b1*x1 + b2*x2 + ...
- Log-odds: ln(p/(1-p)) = z

**Code Pattern:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)
```

**Finance Insight:**
- Credit scoring: Predict default probability
- Trade signals: Classify buy/sell opportunities
- Threshold tuning: Adjust 0.5 cutoff based on costs
- Odds ratio: exp(coef) = change in odds per unit change in x

**Builds On:** L21-L24 (regression)

**Leads To:** L26 (decision trees), L27 (classification metrics)

---

## L26: Decision Trees

**Problem:** How to create interpretable classification rules?

**Solution:** Recursive binary splits based on feature thresholds

**Key Concepts:**
- Gini impurity: 2p(1-p), measures node purity
- Information gain: Reduction in entropy after split
- Pruning: Limit depth to prevent overfitting

**Code Pattern:**
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

tree = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
plot_tree(tree, feature_names=X.columns, filled=True)
```

**Finance Insight:**
- Credit decisioning: Explainable approval rules
- Risk bucketing: Automatic threshold discovery
- Random forests: Ensemble of trees reduces variance
- Feature importance: Which variables drive splits

**Builds On:** L25 (logistic regression)

**Leads To:** L27 (metrics), L28 (class imbalance)

---

## L27: Classification Metrics

**Problem:** How to evaluate binary classifiers beyond accuracy?

**Solution:** Confusion matrix, precision, recall, F1, ROC-AUC

**Key Formulas:**
- Precision = TP / (TP + FP) - quality of positive predictions
- Recall = TP / (TP + FN) - coverage of actual positives
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- AUC: Area under ROC curve (0.5 = random, 1.0 = perfect)

**Code Pattern:**
```python
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_true, y_pred))
auc = roc_auc_score(y_true, y_proba)
```

**Finance Insight:**
- Fraud detection: High recall (catch all fraud) vs precision tradeoff
- Credit scoring: AUC is industry standard metric
- Threshold selection: ROC curve shows all tradeoffs
- Cost-sensitive: Weight FP and FN differently

**Builds On:** L25-L26 (classification models)

**Leads To:** L28 (class imbalance)

---

## L28: Class Imbalance

**Problem:** How to handle rare events (fraud, default, churn)?

**Solution:** Resampling, class weights, and appropriate metrics

**Key Techniques:**
- Oversampling: SMOTE creates synthetic minority samples
- Undersampling: Reduce majority class
- Class weights: Penalize majority class errors less

**Code Pattern:**
```python
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
model = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
```

**Finance Insight:**
- Fraud: 0.1% fraud rate means accuracy is meaningless
- Use precision-recall curve, not ROC, for imbalanced data
- Stratified CV: Maintain class proportions in folds
- Business cost: FN (missed fraud) often costlier than FP

**Builds On:** L27 (classification metrics)

**Leads To:** L29-L32 (unsupervised learning)

---

## L29: K-Means Clustering

**Problem:** How to discover natural groupings in data?

**Solution:** K-Means partitions data into k clusters by minimizing within-cluster variance

**Key Concepts:**
- Elbow method: Plot inertia vs k, find "elbow"
- Silhouette score: Measure cluster separation (-1 to 1)
- Initialization: k-means++ for better starting centroids

**Code Pattern:**
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
```

**Finance Insight:**
- Customer segmentation: Group by behavior, not demographics
- Stock clustering: Find similar return patterns
- Regime detection: Market states (bull, bear, sideways)
- Feature engineering: Cluster membership as new feature

**Builds On:** L11 (NumPy), L31 (PCA for visualization)

**Leads To:** L30 (hierarchical clustering)

---

## L30: Hierarchical Clustering

**Problem:** How to visualize nested cluster structure?

**Solution:** Agglomerative clustering builds tree (dendrogram) from bottom up

**Key Concepts:**
- Linkage: single, complete, average, Ward
- Dendrogram: Visual tree showing merge order
- Cutting: Choose height to get desired number of clusters

**Code Pattern:**
```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

Z = linkage(X, method='ward')
dendrogram(Z)
clusters = fcluster(Z, t=4, criterion='maxclust')
```

**Finance Insight:**
- Asset classification: Build taxonomy of investments
- Portfolio construction: Group correlated assets
- No k required: Dendrogram shows all possible clusterings
- Interpretable: Can explain why assets grouped together

**Builds On:** L29 (K-Means)

**Leads To:** L31 (PCA)

---

## L31: PCA

**Problem:** How to reduce dimensionality while preserving information?

**Solution:** Principal Component Analysis finds orthogonal directions of maximum variance

**Key Concepts:**
- Eigenvalues: Variance explained by each component
- Scree plot: Visualize explained variance
- Loadings: How original features contribute to PCs

**Code Pattern:**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(X_scaled)
X_pca = pca.transform(X_scaled)
explained = pca.explained_variance_ratio_
```

**Finance Insight:**
- Factor analysis: First PC often tracks market
- Risk decomposition: Identify principal risk factors
- Visualization: Project high-dim data to 2D/3D
- Noise reduction: Keep top PCs, discard rest

**Builds On:** L11 (linear algebra concepts)

**Leads To:** L32 (ML pipeline)

---

## L32: ML Pipeline

**Problem:** How to organize preprocessing and modeling into reproducible workflows?

**Solution:** sklearn Pipeline chains transformers and estimators

**Key Components:**
- Pipeline: Sequential steps (preprocess -> model)
- ColumnTransformer: Different transforms for different columns
- GridSearchCV: Hyperparameter search with CV

**Code Pattern:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
```

**Finance Insight:**
- Prevent leakage: Fit scaler only on training data
- Reproducibility: Same pipeline for train and production
- Model selection: Compare multiple algorithms systematically
- Deployment: Save entire pipeline with joblib

**Builds On:** L21-L31 (all ML techniques)

**Leads To:** L33-L36 (deep learning)

---

## L33: Perceptron

**Problem:** How do neural networks learn decision boundaries?

**Solution:** Perceptron learns linear boundary via iterative weight updates

**Key Concepts:**
- Neuron: weighted sum + activation
- Learning rule: w = w + lr * (y - y_hat) * x
- XOR problem: Single perceptron cannot solve non-linear

**Code Pattern:**
```python
from sklearn.linear_model import Perceptron

perceptron = Perceptron(max_iter=1000).fit(X_train, y_train)
weights = perceptron.coef_
bias = perceptron.intercept_
```

**Finance Insight:**
- Foundation for neural networks
- Linear separability: Works when classes can be divided by hyperplane
- Historical: First neural network model (1958)

**Builds On:** L25 (logistic regression concepts)

**Leads To:** L34 (MLP, multiple layers)

---

## L34: MLP Activations

**Problem:** How to learn non-linear patterns?

**Solution:** Multi-layer perceptron with non-linear activations

**Key Activations:**
- ReLU: max(0, x) - default for hidden layers
- Sigmoid: 1/(1+exp(-x)) - output for binary
- Softmax: exp(x_i)/sum(exp(x)) - output for multiclass

**Code Pattern:**
```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                    activation='relu',
                    max_iter=500).fit(X_train, y_train)
```

**Finance Insight:**
- Universal approximator: Can learn any function
- Hidden layers: Automatic feature extraction
- Vanishing gradients: ReLU solves sigmoid's problem
- Architecture design: Start simple, add complexity if needed

**Builds On:** L33 (perceptron)

**Leads To:** L35 (backpropagation), L36 (overfitting)

---

## L35: Backpropagation

**Problem:** How do neural networks learn weights?

**Solution:** Backpropagation computes gradients via chain rule

**Key Concepts:**
- Forward pass: Compute predictions
- Loss function: MSE (regression), cross-entropy (classification)
- Backward pass: Compute gradients layer by layer
- Gradient descent: Update weights in direction of steepest descent

**Code Pattern:**
```python
# Conceptual (PyTorch-style)
loss = criterion(output, target)
loss.backward()  # Compute gradients
optimizer.step()  # Update weights
```

**Finance Insight:**
- Learning rate: Too high = unstable, too low = slow
- Mini-batch: Balance computation and gradient noise
- Adam optimizer: Adaptive learning rates per parameter
- Early stopping: Monitor validation loss to prevent overfit

**Builds On:** L34 (MLP architecture)

**Leads To:** L36 (overfitting prevention)

---

## L36: Overfitting Prevention

**Problem:** How to prevent neural networks from memorizing training data?

**Solution:** Dropout, early stopping, regularization

**Key Techniques:**
- Dropout: Randomly zero neurons during training
- Early stopping: Stop when validation loss increases
- L2 regularization: Weight decay penalty
- Data augmentation: Create more training examples

**Code Pattern:**
```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    alpha=0.01,  # L2 regularization
    early_stopping=True,
    validation_fraction=0.1
).fit(X_train, y_train)
```

**Finance Insight:**
- Small datasets: Finance often has limited data, overfit risk high
- Ensemble: Combine multiple models (bagging, boosting)
- Cross-validation: Essential for model selection
- Simplicity: Simpler models often generalize better

**Builds On:** L34-L35 (MLP, backprop)

**Leads To:** L37-L40 (NLP applications)

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
