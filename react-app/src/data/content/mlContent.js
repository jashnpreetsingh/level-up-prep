// Comprehensive ML Content for Study Modal
// Each topic has DEEP explanations with intuition, derivations, and the "why" behind everything

export const mlContent = {
    "Linear Regression": `
# Linear Regression

## 1. The Fundamental Question

Imagine you have data: house sizes and their prices. You want to predict the price of a new house given its size. **How do you find the best line through your data points?**

This is the essence of linear regression. But more importantly: **why a line?** And **how do we define "best"?**

### Why Linear?

The assumption of linearity isn't arbitrary—it's the simplest possible relationship. Before jumping to complex models, we start here because:

1. **Interpretability**: Each coefficient tells you exactly how much the output changes per unit input
2. **Occam's Razor**: The simplest explanation is often the best
3. **Foundation**: Every complex model (neural networks, ensemble methods) builds on these principles

The model assumes:
$$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_n x_n + \\epsilon$$

But **what does each term mean intuitively?**

- $\\beta_0$ (intercept): The baseline value when all inputs are zero
- $\\beta_i$ (coefficients): How much $y$ changes when $x_i$ increases by 1, **holding everything else constant**
- $\\epsilon$ (error): Everything the model can't explain—randomness, unmeasured factors, reality's complexity

---

## 2. Defining "Best": The Loss Function

### The Core Problem

You have infinitely many lines you could draw through your data. **How do you pick THE line?**

You need a **loss function**—a number that tells you how "wrong" your line is. A good line minimizes this number.

### Why Sum of SQUARED Errors?

The most common choice is the **Sum of Squared Errors (SSE)**:

$$SSE = \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$

But **why squares?** Why not absolute values? Why not cubes? This choice has deep reasons:

**Reason 1: Differentiability**
Squared errors are smooth everywhere. Absolute errors have a "kink" at zero, making calculus harder. We need smooth derivatives for optimization.

**Reason 2: Penalizing Large Errors More**
Squaring makes big errors MUCH worse than small ones. An error of 10 contributes 100 to the loss, while an error of 1 contributes just 1. This says: "I'd rather have many small errors than a few catastrophic ones."

**Reason 3: Connection to Probability (The Deep Reason)**
If we believe errors are **Gaussian (normally distributed)**, then minimizing squared error is equivalent to **Maximum Likelihood Estimation**. This isn't a coincidence—it's the mathematical foundation of why this works.

### Why the Gaussian Assumption Works

In the real world, errors often ARE approximately Gaussian. Why? The **Central Limit Theorem**: when many small, independent factors contribute to the error, their sum tends toward a normal distribution.

Think about it: a house price is affected by hundreds of factors (neighborhood quality, age, condition, market timing, etc.). Even if each factor has a weird distribution, their combined effect tends toward normality.

---

## 3. Finding the Optimal Solution

### The Calculus Approach

We want to minimize:
$$SSE = \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 = \\sum_{i=1}^{n} (y_i - \\mathbf{x}_i^T\\boldsymbol{\\beta})^2$$

In matrix form (which is cleaner for derivation):
$$SSE = (\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})^T(\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})$$

**Step 1: Expand the expression**
$$SSE = \\mathbf{y}^T\\mathbf{y} - 2\\boldsymbol{\\beta}^T\\mathbf{X}^T\\mathbf{y} + \\boldsymbol{\\beta}^T\\mathbf{X}^T\\mathbf{X}\\boldsymbol{\\beta}$$

**Step 2: Take the derivative and set to zero**

Using matrix calculus rules:
- $\\frac{\\partial}{\\partial \\boldsymbol{\\beta}}(\\boldsymbol{\\beta}^T\\mathbf{a}) = \\mathbf{a}$
- $\\frac{\\partial}{\\partial \\boldsymbol{\\beta}}(\\boldsymbol{\\beta}^T\\mathbf{A}\\boldsymbol{\\beta}) = 2\\mathbf{A}\\boldsymbol{\\beta}$ (when A is symmetric)

$$\\frac{\\partial SSE}{\\partial \\boldsymbol{\\beta}} = -2\\mathbf{X}^T\\mathbf{y} + 2\\mathbf{X}^T\\mathbf{X}\\boldsymbol{\\beta} = 0$$

**Step 3: Solve for β**

$$\\mathbf{X}^T\\mathbf{X}\\boldsymbol{\\beta} = \\mathbf{X}^T\\mathbf{y}$$

$$\\boxed{\\boldsymbol{\\beta} = (\\mathbf{X}^T\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{y}}$$

This is the **Normal Equation**. It's beautiful because:
1. It's a **closed-form solution** (no iteration needed)
2. It's **unique** (when $\\mathbf{X}^T\\mathbf{X}$ is invertible)
3. It's **guaranteed to be the global minimum** (because SSE is convex)

### When Does This Break?

The inverse $(\\mathbf{X}^T\\mathbf{X})^{-1}$ only exists when the matrix is **invertible**. This fails when:

1. **Multicollinearity**: Features are linearly dependent (e.g., both "age in years" and "age in months")
2. **More features than samples**: $p > n$ makes the matrix rank-deficient

---

## 4. The Probabilistic View: Why MLE = OLS

This is where the magic happens. Let's derive linear regression from **probability theory**.

### The Generative Story

Assume each observation follows:
$$y_i = \\mathbf{x}_i^T\\boldsymbol{\\beta} + \\epsilon_i \\quad \\text{where} \\quad \\epsilon_i \\sim \\mathcal{N}(0, \\sigma^2)$$

This means: "The true relationship is linear, plus some Gaussian noise."

### Deriving the Likelihood

Given this model, what's the probability of observing our data?

For a single point:
$$P(y_i | \\mathbf{x}_i, \\boldsymbol{\\beta}) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\left(-\\frac{(y_i - \\mathbf{x}_i^T\\boldsymbol{\\beta})^2}{2\\sigma^2}\\right)$$

For all points (assuming independence):
$$L(\\boldsymbol{\\beta}) = \\prod_{i=1}^{n} P(y_i | \\mathbf{x}_i, \\boldsymbol{\\beta})$$

### The Log-Likelihood

Taking the log (which is easier to work with):
$$\\log L(\\boldsymbol{\\beta}) = -\\frac{n}{2}\\log(2\\pi\\sigma^2) - \\frac{1}{2\\sigma^2}\\sum_{i=1}^{n}(y_i - \\mathbf{x}_i^T\\boldsymbol{\\beta})^2$$

**The key insight**: Maximizing this is equivalent to **minimizing the sum of squared errors**!

The first term is constant (doesn't depend on β), and the second term is just SSE scaled by a constant. This proves:

> **Under Gaussian noise assumptions, Maximum Likelihood Estimation gives the same answer as Ordinary Least Squares.**

This is why least squares isn't arbitrary—it has deep probabilistic justification.

---

## 5. Gradient Descent: When Closed-Form Fails

For very large datasets, computing $(\\mathbf{X}^T\\mathbf{X})^{-1}$ is expensive ($O(p^3)$ for $p$ features). Gradient descent offers an iterative alternative.

### The Intuition

Imagine you're blindfolded on a hilly landscape, trying to find the lowest point. You feel the slope beneath your feet and take a step downhill. Repeat until you stop descending.

### The Update Rule

$$\\boldsymbol{\\beta}_{t+1} = \\boldsymbol{\\beta}_t - \\eta \\nabla_{\\boldsymbol{\\beta}} SSE$$

Where:
- $\\eta$ is the **learning rate** (step size)
- $\\nabla_{\\boldsymbol{\\beta}} SSE = -2\\mathbf{X}^T(\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})$ is the gradient

### Why Does This Work?

The gradient points in the direction of **steepest increase**. By taking the negative, we walk toward the minimum. For convex functions (like SSE), this is guaranteed to find the global minimum eventually.

### Implementation

\`\`\`python
import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights to small random values
        # (zeros work too, but random helps with symmetry breaking)
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for iteration in range(self.n_iters):
            # Forward pass: compute predictions
            y_pred = X @ self.weights + self.bias
            
            # Compute loss for monitoring
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)
            
            # Backward pass: compute gradients
            # d(SSE)/d(weights) = -2 * X^T * (y - y_pred) / n
            dw = -(2 / n_samples) * (X.T @ (y - y_pred))
            db = -(2 / n_samples) * np.sum(y - y_pred)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias
\`\`\`

---

## 6. The Assumptions and Why They Matter

Linear regression makes four key assumptions (remembered as **LINE**):

### Linearity
**What it means**: The true relationship between X and y is linear.

**Why it matters**: If the real relationship is curved, a line will systematically miss the pattern.

**How to check**: Plot residuals vs. fitted values. Random scatter = good. Curved pattern = linearity violated.

**What to do if violated**: Add polynomial features, use a non-linear model, or apply transformations.

### Independence
**What it means**: Errors are uncorrelated with each other.

**Why it matters**: If today's error predicts tomorrow's error (common in time series), your standard errors will be wrong, making hypothesis tests invalid.

**How to check**: Durbin-Watson test, or plot residuals over time.

### Normality
**What it means**: Errors follow a Gaussian distribution.

**Why it matters**: Mostly for **inference** (confidence intervals, p-values). For prediction, this is less critical.

**How to check**: Q-Q plot of residuals.

### Equal Variance (Homoscedasticity)
**What it means**: The spread of errors is constant across all values of X.

**Why it matters**: If variance increases with X, your model is less reliable for high-X predictions.

**How to check**: Plot residuals vs. fitted values. Look for "cone" shapes.

---

## 7. Interview Deep Dives

> **Q: You've fit a linear regression. The R² is 0.95. Is this a good model?**

**A**: Not necessarily! High R² can be misleading:
- **Overfitting**: Too many features can inflate R² while hurting generalization
- **Domain context**: In physics, 0.95 might be poor. In social sciences, it might be excellent.
- **Check residuals**: Even with high R², patterns in residuals indicate problems

Always validate on held-out data, not just training R².

> **Q: Why might you choose gradient descent over the normal equation?**

**A**: 
- **Scale**: Normal equation is $O(p^3)$. With millions of features, this is prohibitive.
- **Memory**: Normal equation requires storing $\\mathbf{X}^T\\mathbf{X}$, which is $p \\times p$.
- **Online learning**: Gradient descent can update with new data. Normal equation requires recomputation.
- **Regularization**: Some regularizers are easier with iterative methods.

> **Q: What happens to coefficients when features are highly correlated?**

**A**: They become **unstable**. Small changes in data cause large swings in coefficients. The model may still predict well, but individual coefficients become uninterpretable. The matrix $\\mathbf{X}^T\\mathbf{X}$ becomes nearly singular, making the inverse ill-conditioned.

**Solution**: Use regularization (Ridge/Lasso), remove redundant features, or use PCA.

---

## 8. Key Takeaways

1. **Linear regression minimizes squared error**, which is equivalent to maximum likelihood under Gaussian noise
2. **The normal equation gives a closed-form solution**, but gradient descent scales better
3. **Assumptions matter for inference**, but are less critical for pure prediction
4. **Multicollinearity doesn't hurt predictions**, but destroys interpretability
5. **Always plot residuals**—they reveal what R² hides
`,

    "Logistic Regression": `
# Logistic Regression

## 1. From Lines to Probabilities

Linear regression predicts a continuous value. But what if we want to predict **yes or no**? Spam or not spam? Tumor is malignant or benign?

### The Problem with Linear Classification

You might think: "Just use linear regression and threshold at 0.5." But this fails:

1. **Predictions can go beyond [0,1]**: A line can predict -0.3 or 1.7, which don't make sense as probabilities
2. **Outliers distort the boundary**: One extreme point can shift your decision threshold dramatically
3. **Equal sensitivity everywhere**: A point far from the boundary affects the line as much as one near it

We need a model that:
- Outputs values between 0 and 1 (valid probabilities)
- Has a natural threshold interpretation
- Links to a proper probabilistic framework

### The Sigmoid Solution

Enter the **sigmoid function**:

$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$

This function:
- Takes any real number as input
- Outputs a value in $(0, 1)$
- Is smooth and differentiable
- Has a beautiful derivative: $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$

**Why this specific function?** It's not arbitrary. The sigmoid emerges naturally when you model log-odds as a linear function—we'll derive this from first principles.

---

## 2. The Probabilistic Foundation

### Bernoulli Distribution

For binary outcomes, we use the **Bernoulli distribution**:
$$P(y | p) = p^y (1-p)^{1-y}$$

- When $y = 1$: $P = p$
- When $y = 0$: $P = 1 - p$

Our goal: model $p$ as a function of features $\\mathbf{x}$.

### The Odds and Log-Odds

**Odds** = probability of success / probability of failure = $\\frac{p}{1-p}$

If you have a 75% chance of winning, your odds are $\\frac{0.75}{0.25} = 3:1$.

**Log-odds (logit)** = $\\log\\left(\\frac{p}{1-p}\\right)$

**Key insight**: Log-odds range from $-\\infty$ to $+\\infty$, just like a linear function!

### The Logistic Model

We assume:
$$\\log\\left(\\frac{p}{1-p}\\right) = \\mathbf{w}^T\\mathbf{x} + b$$

Solving for $p$:
$$\\frac{p}{1-p} = e^{\\mathbf{w}^T\\mathbf{x} + b}$$
$$p = \\frac{e^{\\mathbf{w}^T\\mathbf{x} + b}}{1 + e^{\\mathbf{w}^T\\mathbf{x} + b}} = \\frac{1}{1 + e^{-(\\mathbf{w}^T\\mathbf{x} + b)}} = \\sigma(\\mathbf{w}^T\\mathbf{x} + b)$$

**The sigmoid wasn't chosen arbitrarily—it emerges from assuming log-odds are linear.**

---

## 3. Maximum Likelihood Estimation

### Why Not Least Squares?

Can we minimize $(y - \\sigma(\\mathbf{w}^T\\mathbf{x}))^2$? Technically yes, but:

1. **Non-convex**: The loss surface has many local minima
2. **Probabilistically wrong**: It doesn't correspond to the correct likelihood

### The Correct Loss Function

Given our Bernoulli assumption, the likelihood of observing all data is:
$$L(\\mathbf{w}) = \\prod_{i=1}^{n} p_i^{y_i} (1-p_i)^{1-y_i}$$

Taking the log:
$$\\log L = \\sum_{i=1}^{n} \\left[ y_i \\log(p_i) + (1-y_i) \\log(1-p_i) \\right]$$

We want to maximize this, which is equivalent to minimizing the **negative log-likelihood**:

$$\\mathcal{L}(\\mathbf{w}) = -\\sum_{i=1}^{n} \\left[ y_i \\log(p_i) + (1-y_i) \\log(1-p_i) \\right]$$

This is called **binary cross-entropy loss**.

### Why Cross-Entropy Works

**Intuition**: Cross-entropy measures how "surprised" you are by the true labels given your predictions.

- If you predict $p = 0.99$ and $y = 1$: loss $= -\\log(0.99) \\approx 0.01$ (tiny)
- If you predict $p = 0.99$ and $y = 0$: loss $= -\\log(0.01) \\approx 4.6$ (huge!)

The loss **explodes** when you're confidently wrong, which is exactly the behavior we want.

---

## 4. The Gradient Derivation

### Computing the Gradient

For a single data point, the loss is:
$$\\ell = -y \\log(p) - (1-y) \\log(1-p)$$

where $p = \\sigma(z)$ and $z = \\mathbf{w}^T\\mathbf{x}$.

Using chain rule:
$$\\frac{\\partial \\ell}{\\partial \\mathbf{w}} = \\frac{\\partial \\ell}{\\partial p} \\cdot \\frac{\\partial p}{\\partial z} \\cdot \\frac{\\partial z}{\\partial \\mathbf{w}}$$

**Step 1**: $\\frac{\\partial \\ell}{\\partial p} = -\\frac{y}{p} + \\frac{1-y}{1-p}$

**Step 2**: $\\frac{\\partial p}{\\partial z} = \\sigma(z)(1 - \\sigma(z)) = p(1-p)$

**Step 3**: $\\frac{\\partial z}{\\partial \\mathbf{w}} = \\mathbf{x}$

Combining:
$$\\frac{\\partial \\ell}{\\partial \\mathbf{w}} = \\left(-\\frac{y}{p} + \\frac{1-y}{1-p}\\right) \\cdot p(1-p) \\cdot \\mathbf{x}$$

Simplifying:
$$= \\left(-y(1-p) + (1-y)p\\right) \\mathbf{x} = (p - y)\\mathbf{x}$$

**Remarkably simple**: The gradient is $(\\text{prediction} - \\text{truth}) \\times \\text{features}$.

This is identical in form to linear regression's gradient! The sigmoid's derivative "cancels" nicely with the cross-entropy derivative.

---

## 5. Decision Boundaries

### What the Model Learns

The prediction is:
$$p = \\sigma(\\mathbf{w}^T\\mathbf{x} + b)$$

We classify as positive when $p > 0.5$, which means:
$$\\sigma(\\mathbf{w}^T\\mathbf{x} + b) > 0.5$$
$$\\mathbf{w}^T\\mathbf{x} + b > 0$$

This is a **hyperplane** in feature space. Everything on one side is class 0, everything on the other is class 1.

### Geometric Interpretation

- $\\mathbf{w}$ is the **normal vector** to the decision boundary
- $b$ is the **offset** from the origin
- Points far from the boundary get extreme probabilities (near 0 or 1)
- Points near the boundary get probabilities near 0.5

---

## 6. Regularization: Preventing Overconfidence

### The Problem

When classes are separable, logistic regression can push $\\mathbf{w}$ to infinity to make predictions more extreme. This leads to **overconfident predictions** and **poor generalization**.

### L2 Regularization (Ridge)

Add a penalty on weight magnitude:
$$\\mathcal{L}_{regularized} = \\mathcal{L} + \\lambda \\|\\mathbf{w}\\|_2^2$$

**Effect**: Shrinks all weights toward zero, but never exactly to zero.

**Interpretation**: Bayesian prior that weights come from a Gaussian distribution centered at zero.

### L1 Regularization (Lasso)

$$\\mathcal{L}_{regularized} = \\mathcal{L} + \\lambda \\|\\mathbf{w}\\|_1$$

**Effect**: Drives some weights exactly to zero, performing **feature selection**.

**Why does L1 create sparsity?** The L1 penalty has "corners" at zero, so the optimal solution often lands exactly at these corners.

---

## 7. Implementation with Intuition

\`\`\`python
import numpy as np

class LogisticRegression:
    """
    Logistic regression with gradient descent.
    
    The key insight: despite the sigmoid and cross-entropy,
    the gradient has the same form as linear regression:
    gradient = (prediction - truth) * features
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.0):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.lambda_reg = lambda_reg  # L2 regularization strength
        
    def sigmoid(self, z):
        # Clip to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            # Forward pass
            linear = X @ self.weights + self.bias
            predictions = self.sigmoid(linear)
            
            # Gradient (with L2 regularization)
            dw = (1/n_samples) * (X.T @ (predictions - y))
            dw += (self.lambda_reg / n_samples) * self.weights  # L2 term
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_proba(self, X):
        """Return probability of class 1."""
        return self.sigmoid(X @ self.weights + self.bias)
    
    def predict(self, X, threshold=0.5):
        """Return class predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)
\`\`\`

---

## 8. Interview Deep Dives

> **Q: Why cross-entropy instead of MSE for classification?**

**A**: Two reasons:

1. **Convexity**: Cross-entropy with sigmoid is convex (has unique minimum). MSE with sigmoid is non-convex (multiple local minima).

2. **Gradient behavior**: With MSE, when you're confidently wrong, the gradient is small (sigmoid is flat). With cross-entropy, being confidently wrong produces **huge gradients**, forcing rapid correction.

> **Q: How do you handle imbalanced classes (99% vs 1%)?**

**A**: Multiple strategies:

1. **Class weights**: Penalize misclassifying the minority class more heavily
2. **Resampling**: Oversample minority (SMOTE) or undersample majority
3. **Threshold adjustment**: Don't use 0.5; optimize threshold on validation set
4. **Different metrics**: Use precision-recall, F1, or AUC instead of accuracy

> **Q: What does the coefficient mean in logistic regression?**

**A**: Each coefficient represents the change in **log-odds** for a one-unit increase in that feature.

If $w_j = 0.3$, then increasing $x_j$ by 1 multiplies the odds by $e^{0.3} \\approx 1.35$ (35% increase in odds).

---

## 9. Key Takeaways

1. **Logistic regression models log-odds as linear**, which naturally produces the sigmoid
2. **Cross-entropy is the correct loss** because it corresponds to maximum likelihood under Bernoulli assumptions
3. **The gradient has the same form as linear regression**: $(\\text{prediction} - \\text{truth}) \\times \\text{features}$
4. **The decision boundary is a hyperplane**, but the model outputs probabilities, not hard predictions
5. **Regularization prevents overconfident predictions** and enables feature selection (L1)
`,

    "Bias-Variance Tradeoff": `
# The Bias-Variance Tradeoff

## 1. The Central Dilemma of Machine Learning

Every machine learning practitioner faces this tension: make your model **complex enough to capture the pattern**, but **simple enough to not memorize noise**.

This isn't just practical advice—it's a fundamental mathematical truth about learning from data.

### The Error You Can't Escape

When we train a model, we care about **how well it predicts on NEW data**, not how well it memorizes training data. This generalization error can be decomposed into three irreducible sources:

$$\\text{Expected Error} = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Noise}$$

Let's understand each term deeply.

---

## 2. Bias: The Error of Simplification

### What It Is

**Bias** is the error from using a model that's too simple to capture the true pattern.

Imagine the true relationship is quadratic ($y = x^2$), but you fit a straight line. No matter how much data you have, a line will NEVER capture the curve. This systematic error is bias.

### Mathematical Definition

$$\\text{Bias}[\\hat{f}(x)] = E[\\hat{f}(x)] - f(x)$$

In words: the difference between what your model predicts ON AVERAGE (across many possible training sets) and the true underlying function.

### Intuition

A high-bias model has **strong assumptions** that may not match reality:
- Linear regression assumes linear relationships
- A decision tree with max_depth=1 can only make one split

### Signs of High Bias

- Training error is high
- Training and validation error are similar (both bad)
- Model is "underfitting"—too simple for the data

---

## 3. Variance: The Error of Instability

### What It Is

**Variance** is the error from a model that's too sensitive to the specific training data.

Train a decision tree with no depth limit on dataset A, then on a slightly different dataset B. If the trees look completely different and make different predictions, that's high variance.

### Mathematical Definition

$$\\text{Variance}[\\hat{f}(x)] = E\\left[(\\hat{f}(x) - E[\\hat{f}(x)])^2\\right]$$

In words: how much your model's predictions fluctuate across different possible training sets.

### Intuition

A high-variance model:
- Captures every quirk of the training data (including noise)
- Changes dramatically with different training samples
- Has too many parameters relative to the data

### Signs of High Variance

- Training error is low, but validation error is high
- The gap between training and validation error is large
- Model is "overfitting"—memorizing rather than learning

---

## 4. The Mathematical Proof

This is one of the most beautiful results in ML. Let's prove it.

### Setup

- True function: $f(x)$
- Our model: $\\hat{f}(x)$ (random because it depends on training data)
- Data is generated as: $y = f(x) + \\epsilon$ where $E[\\epsilon] = 0$ and $Var(\\epsilon) = \\sigma^2$

### Goal

Show that:
$$E[(y - \\hat{f}(x))^2] = \\text{Bias}^2 + \\text{Variance} + \\sigma^2$$

### Proof

Let $\\bar{f}(x) = E[\\hat{f}(x)]$ (the expected prediction).

$$E[(y - \\hat{f})^2] = E[(f + \\epsilon - \\hat{f})^2]$$

Add and subtract $\\bar{f}$:
$$= E[(f - \\bar{f} + \\bar{f} - \\hat{f} + \\epsilon)^2]$$

Expand (using independence of $\\epsilon$ from the model):
$$= (f - \\bar{f})^2 + E[(\\hat{f} - \\bar{f})^2] + E[\\epsilon^2]$$

The cross terms vanish because:
- $E[\\epsilon] = 0$
- $E[\\hat{f} - \\bar{f}] = 0$ by definition

$$= \\underbrace{(f - \\bar{f})^2}_{\\text{Bias}^2} + \\underbrace{E[(\\hat{f} - \\bar{f})^2]}_{\\text{Variance}} + \\underbrace{\\sigma^2}_{\\text{Noise}}$$

**QED.** This is not an approximation—it's an exact decomposition.

---

## 5. The Tradeoff in Action

### Why It's a Tradeoff

As you increase model complexity:
- **Bias decreases**: The model can capture more complex patterns
- **Variance increases**: More parameters means more sensitivity to training data

The optimal model is at the sweet spot where their SUM is minimized.

### Visualizing with Model Complexity

| Complexity | Example Model | Bias | Variance |
|------------|---------------|------|----------|
| Very Low | Predict mean | Very High | Very Low |
| Low | Linear regression | High | Low |
| Medium | 5-layer neural net | Medium | Medium |
| High | 100-layer neural net | Low | High |
| Very High | 1-NN (memorization) | Zero | Very High |

### The Reality

With modern deep learning and regularization, we've learned to control variance even with complex models. But the fundamental tradeoff remains—we just have better tools to navigate it.

---

## 6. Practical Strategies

### Reducing Bias (Model is Too Simple)

1. **Increase model complexity**: More layers, more trees, polynomial features
2. **Reduce regularization**: Let the model use its capacity
3. **Feature engineering**: Give the model better inputs
4. **Try ensemble methods**: Boosting specifically targets bias

### Reducing Variance (Model is Too Complex)

1. **Get more training data**: The most effective but often expensive solution
2. **Regularization**: L1, L2, dropout, early stopping
3. **Simplify the model**: Fewer parameters, shallower trees
4. **Ensemble averaging**: Bagging averages out variance
5. **Cross-validation**: Get reliable estimates of true performance

---

## 7. The Double Descent Phenomenon

### Breaking the Classical Picture

Modern deep learning has revealed something surprising: as model complexity increases BEYOND the interpolation threshold (where training error hits zero), test error can actually **decrease again**.

This creates a "double descent" curve:
1. Classical regime: U-shaped test error
2. Interpolation threshold: Peak variance
3. Overparameterized regime: Test error decreases again

### Why It Happens

In the overparameterized regime, the model finds "simpler" interpolating functions (with smaller norm). This implicit regularization counteracts the expected variance explosion.

This doesn't invalidate the bias-variance tradeoff—it shows that our understanding of "complexity" needs refinement.

---

## 8. Interview Deep Dives

> **Q: How do you know if your model is underfitting vs. overfitting?**

**A**: Compare training and validation errors:
- **Both high**: Underfitting (high bias)
- **Training low, validation high**: Overfitting (high variance)
- **Both low, similar**: Good fit

Also plot learning curves (error vs. training set size). If they converge high, you have bias problems. If there's a persistent gap, you have variance problems.

> **Q: You have a model with high variance. You can't get more data. What do you do?**

**A**: In order of effectiveness:
1. **Regularization** (L2, dropout, early stopping)
2. **Ensemble averaging** (bagging, random forests)
3. **Reduce model complexity** (fewer layers, lower poly degree)
4. **Data augmentation** (if applicable—creates "virtual" data)
5. **Feature selection** (remove noisy/irrelevant features)

> **Q: Why does bagging reduce variance but boosting reduces bias?**

**A**: 
- **Bagging** trains independent models and averages them. Averaging reduces variance by a factor of n (for independent estimators). It doesn't help bias because averaging biased models keeps the bias.
- **Boosting** trains sequential models where each focuses on errors of the previous. This systematically reduces error, which primarily comes from bias in the component models.

---

## 9. Key Takeaways

1. **Expected error = Bias² + Variance + Noise**—this is exact, not approximate
2. **Bias is systematic error** from model assumptions; **Variance is sensitivity** to training data
3. **Simple models have high bias, low variance**; the reverse for complex models
4. **The goal is to minimize their SUM**, not either one alone
5. **Modern deep learning** has new tools (regularization, overparameterization) but the tradeoff remains fundamental
`,

    "SVM & Kernels": `
# Support Vector Machines & The Kernel Trick

## 1. The Geometric Insight

Most classifiers find ANY boundary that separates classes. SVMs ask a different question: **what's the BEST boundary?**

### Maximum Margin

Imagine two groups of points, perfectly separable by a line. There are infinitely many lines that work. Which should you choose?

**SVM's answer**: The line with the **largest margin**—the widest possible "street" between the classes.

**Why is this smart?**

1. **Robustness**: A wide margin means small perturbations don't cause misclassification
2. **Generalization**: Margin maximization is a form of regularization (from statistical learning theory, the expected error is bounded by a function of the margin)
3. **Uniqueness**: There's only ONE maximum-margin hyperplane, no arbitrary choices

### The Support Vectors

A beautiful property: the optimal hyperplane depends only on a few "critical" points—those sitting on the edge of the margin. These are the **support vectors**.

Why does this matter? Even if you have millions of data points, only a handful might be support vectors. This makes SVMs memory-efficient at prediction time.

---

## 2. The Mathematical Formulation

### Hard Margin SVM

For linearly separable data with labels $y_i \\in \\{-1, +1\\}$:

The decision boundary is: $\\mathbf{w}^T\\mathbf{x} + b = 0$

We want:
- Points with $y_i = +1$ to satisfy $\\mathbf{w}^T\\mathbf{x}_i + b \\geq 1$
- Points with $y_i = -1$ to satisfy $\\mathbf{w}^T\\mathbf{x}_i + b \\leq -1$

Combined: $y_i(\\mathbf{w}^T\\mathbf{x}_i + b) \\geq 1$ for all $i$

### Why the Margin is $\\frac{2}{\\|\\mathbf{w}\\|}$

The distance from a point $\\mathbf{x}$ to the hyperplane $\\mathbf{w}^T\\mathbf{x} + b = 0$ is:
$$\\text{distance} = \\frac{|\\mathbf{w}^T\\mathbf{x} + b|}{\\|\\mathbf{w}\\|}$$

The margin is the distance from the hyperplane to the nearest points. By our constraints, the nearest points have $|\\mathbf{w}^T\\mathbf{x} + b| = 1$.

So margin = $\\frac{1}{\\|\\mathbf{w}\\|}$ on each side, total margin = $\\frac{2}{\\|\\mathbf{w}\\|}$.

### The Optimization Problem

Maximize margin = Minimize $\\|\\mathbf{w}\\|$ = Minimize $\\frac{1}{2}\\|\\mathbf{w}\\|^2$

$$\\min_{\\mathbf{w}, b} \\frac{1}{2}\\|\\mathbf{w}\\|^2 \\quad \\text{subject to} \\quad y_i(\\mathbf{w}^T\\mathbf{x}_i + b) \\geq 1$$

This is a **convex quadratic program** with linear constraints—guaranteed to have a unique global optimum.

---

## 3. The Dual Problem and Why It Matters

### Lagrangian Formulation

Introduce Lagrange multipliers $\\alpha_i \\geq 0$ for each constraint:

$$L(\\mathbf{w}, b, \\boldsymbol{\\alpha}) = \\frac{1}{2}\\|\\mathbf{w}\\|^2 - \\sum_{i=1}^{n} \\alpha_i[y_i(\\mathbf{w}^T\\mathbf{x}_i + b) - 1]$$

### Taking Derivatives

Setting $\\frac{\\partial L}{\\partial \\mathbf{w}} = 0$:
$$\\mathbf{w} = \\sum_{i=1}^{n} \\alpha_i y_i \\mathbf{x}_i$$

This is profound: **the optimal $\\mathbf{w}$ is a linear combination of the training points!**

Setting $\\frac{\\partial L}{\\partial b} = 0$:
$$\\sum_{i=1}^{n} \\alpha_i y_i = 0$$

### The Dual Problem

Substituting back, we get the **dual problem**:

$$\\max_{\\boldsymbol{\\alpha}} \\sum_{i=1}^{n} \\alpha_i - \\frac{1}{2}\\sum_{i,j} \\alpha_i \\alpha_j y_i y_j (\\mathbf{x}_i^T \\mathbf{x}_j)$$

subject to: $\\alpha_i \\geq 0$ and $\\sum_i \\alpha_i y_i = 0$

**Key Observation**: The data appears only in dot products $(\\mathbf{x}_i^T \\mathbf{x}_j)$!

### Support Vectors Revealed

By the **KKT conditions**:
- If $\\alpha_i > 0$: the constraint is **tight** (point is on the margin)
- If $\\alpha_i = 0$: the point is not contributing

Points with $\\alpha_i > 0$ are the support vectors. All other points could be removed without changing the solution.

---

## 4. The Kernel Trick: The Deep Magic

### The Problem with Non-Linear Data

What if your data isn't linearly separable? A classic example: points arranged in two concentric circles.

**Naive Solution**: Map to a higher-dimensional space where it IS separable.

For example, $(x_1, x_2) \\mapsto (x_1, x_2, x_1^2 + x_2^2)$. In 3D, the circles become vertically separated!

**The Problem**: If you have 1000 features and use degree-2 polynomials, you get ~500,000 features. Computing this is expensive.

### The Insight

Remember: the dual only uses dot products $\\mathbf{x}_i^T \\mathbf{x}_j$.

What if we could compute $\\phi(\\mathbf{x}_i)^T \\phi(\\mathbf{x}_j)$ without explicitly computing $\\phi$?

### Kernel Functions

A **kernel** is a function $K(\\mathbf{x}, \\mathbf{x}') = \\phi(\\mathbf{x})^T\\phi(\\mathbf{x}')$ that computes the inner product in a high-dimensional space **implicitly**.

**The magic**: Some kernels correspond to INFINITE-dimensional feature spaces, yet can be computed in finite time!

### Common Kernels and Their Feature Spaces

| Kernel | Formula | Feature Space |
|--------|---------|---------------|
| Linear | $\\mathbf{x}^T\\mathbf{x}'$ | Original ($d$ dimensions) |
| Polynomial | $(\\mathbf{x}^T\\mathbf{x}' + c)^p$ | All monomials up to degree $p$ |
| RBF (Gaussian) | $\\exp(-\\gamma\\|\\mathbf{x} - \\mathbf{x}'\\|^2)$ | INFINITE dimensions! |
| Sigmoid | $\\tanh(\\alpha \\mathbf{x}^T\\mathbf{x}' + c)$ | Similar to neural networks |

### The RBF Kernel: Why It Works

The Gaussian/RBF kernel:
$$K(\\mathbf{x}, \\mathbf{x}') = \\exp(-\\gamma\\|\\mathbf{x} - \\mathbf{x}'\\|^2)$$

It can be shown (via Taylor expansion) that this corresponds to an infinite-dimensional feature space of Gaussian basis functions.

**Interpretation**:
- Similar points ($\\|\\mathbf{x} - \\mathbf{x}'\\|$ small): $K \\approx 1$
- Dissimilar points ($\\|\\mathbf{x} - \\mathbf{x}'\\|$ large): $K \\approx 0$

The parameter $\\gamma$ controls the "radius of influence" of each point.

---

## 5. Soft Margin: Handling Non-Separable Data

### The Reality

Real data is messy. Even the "best" kernel won't perfectly separate noisy data.

**Solution**: Allow some misclassifications, but penalize them.

### The Formulation

Introduce **slack variables** $\\xi_i \\geq 0$:

$$\\min_{\\mathbf{w}, b, \\boldsymbol{\\xi}} \\frac{1}{2}\\|\\mathbf{w}\\|^2 + C\\sum_{i=1}^{n} \\xi_i$$

subject to: $y_i(\\mathbf{w}^T\\mathbf{x}_i + b) \\geq 1 - \\xi_i$

### The C Parameter

$C$ controls the tradeoff:
- **Large C**: Heavily penalize margin violations → narrow margin, few violations
- **Small C**: Accept violations → wide margin, more violations

This is another bias-variance knob:
- Large C → low bias, high variance (could overfit)
- Small C → high bias, low variance (may underfit)

---

## 6. Practical Considerations

### When To Use SVMs

**SVMs excel when:**
- Feature dimension is high relative to samples
- Data is clean with clear margin
- You need non-linear boundaries but want interpretability
- You have moderate-sized datasets (1k-100k samples)

**SVMs struggle when:**
- Dataset is very large (training is $O(n^2)$ to $O(n^3)$)
- Features are noisy or data is inherently overlapping
- You need probability estimates (SVMs output margins, not probabilities)

### Implementation Tips

\`\`\`python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ALWAYS scale features for SVM (kernels depend on distances)
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
])

# Grid search for hyperparameters
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.1, 1]
}
\`\`\`

---

## 7. Key Takeaways

1. **SVMs maximize the margin**, which provides strong generalization guarantees
2. **The solution depends only on support vectors**, making it memory-efficient
3. **The kernel trick enables non-linear boundaries** without explicit high-dimensional computation
4. **RBF kernel corresponds to infinite dimensions** but is computed in constant time
5. **The C parameter trades off margin width vs. violations** (bias-variance tradeoff)
`,

    "Neural Net Backprop": `
# Neural Networks & Backpropagation

## 1. Why Neural Networks?

Linear models are interpretable but limited—they can only learn linear boundaries. What if we could **compose many simple functions** to learn complex ones?

### The Universal Approximation Theorem

A neural network with a single hidden layer (and enough neurons) can approximate **any continuous function** to arbitrary accuracy.

This is remarkable: by stacking simple linear transformations with non-linear activations, we get universal function approximators.

### The Architecture

A feedforward neural network:
1. Takes an input $\\mathbf{x}$
2. Applies alternating linear transformations and non-linear activations
3. Outputs a prediction $\\hat{y}$

For a 2-layer network:
$$\\hat{y} = \\sigma_2(\\mathbf{W}_2 \\cdot \\sigma_1(\\mathbf{W}_1 \\cdot \\mathbf{x} + \\mathbf{b}_1) + \\mathbf{b}_2)$$

---

## 2. Why Non-Linearity is Crucial

### The Problem Without It

Consider two linear layers:
$$\\mathbf{h} = \\mathbf{W}_1\\mathbf{x}$$
$$\\mathbf{y} = \\mathbf{W}_2\\mathbf{h} = \\mathbf{W}_2\\mathbf{W}_1\\mathbf{x} = \\mathbf{W}'\\mathbf{x}$$

Without non-linearity, any number of layers collapses to a single linear transformation! Depth becomes useless.

### Activation Functions

Activation functions introduce the non-linearity that makes deep networks powerful:

**ReLU** (Rectified Linear Unit):
$$\\text{ReLU}(z) = \\max(0, z)$$
- Simple, fast, works well in practice
- Derivative is 0 or 1 (no vanishing gradient)
- "Dead neurons" can occur (always output 0)

**Sigmoid**:
$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$
- Squashes to (0, 1)—good for probabilities
- Saturates at extremes (vanishing gradient)
- Use only for output layer in binary classification

**Tanh**:
$$\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$$
- Squashes to (-1, 1)
- Zero-centered (unlike sigmoid)
- Still has saturation issues

---

## 3. The Forward Pass

### Layer by Layer Computation

For layer $l$:
$$\\mathbf{z}^{(l)} = \\mathbf{W}^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}$$
$$\\mathbf{a}^{(l)} = g^{(l)}(\\mathbf{z}^{(l)})$$

Where:
- $\\mathbf{z}$ = pre-activation (linear combination)
- $\\mathbf{a}$ = activation (after non-linearity)
- $g$ = activation function

### Example: A 3-Layer Network

\`\`\`
Input x → [W1] → z1 → ReLU → a1 → [W2] → z2 → ReLU → a2 → [W3] → z3 → softmax → ŷ
\`\`\`

Each arrow is a simple operation. The magic is in how we learn the weights.

---

## 4. Backpropagation: The Core Algorithm

### The Problem

We have a loss function $L(\\hat{y}, y)$. We want $\\frac{\\partial L}{\\partial W}$ for every weight to do gradient descent.

A network has MILLIONS of weights. Computing each gradient separately would be impossibly slow.

### The Insight: Chain Rule

For a composition of functions $f(g(h(x)))$:
$$\\frac{df}{dx} = \\frac{df}{dg} \\cdot \\frac{dg}{dh} \\cdot \\frac{dh}{dx}$$

Neural networks are just compositions! We can compute all gradients efficiently by propagating derivatives backward through the network.

### The Algorithm

**Step 1: Forward Pass**
Compute all activations layer by layer, storing intermediate values.

**Step 2: Backward Pass**
Starting from the output:
1. Compute $\\frac{\\partial L}{\\partial \\hat{y}}$ (how loss changes with output)
2. Propagate backward through each layer using chain rule
3. Store gradients for each weight

### Mathematical Details

For the output layer with softmax + cross-entropy:
$$\\frac{\\partial L}{\\partial \\mathbf{z}^{(L)}} = \\hat{\\mathbf{y}} - \\mathbf{y}$$

For hidden layers:
$$\\frac{\\partial L}{\\partial \\mathbf{z}^{(l)}} = (\\mathbf{W}^{(l+1)})^T \\frac{\\partial L}{\\partial \\mathbf{z}^{(l+1)}} \\odot g'(\\mathbf{z}^{(l)})$$

For weights:
$$\\frac{\\partial L}{\\partial \\mathbf{W}^{(l)}} = \\frac{\\partial L}{\\partial \\mathbf{z}^{(l)}} (\\mathbf{a}^{(l-1)})^T$$

---

## 5. Implementation from Scratch

\`\`\`python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list like [input_dim, hidden1, hidden2, output_dim]
        """
        self.weights = []
        self.biases = []
        
        # Initialize weights with Xavier initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i])
            w *= np.sqrt(2.0 / layer_sizes[i])  # He initialization for ReLU
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def forward(self, X):
        """Forward pass, storing activations for backprop."""
        self.activations = [X]
        self.z_values = []
        
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ a + b
            self.z_values.append(z)
            
            if i == len(self.weights) - 1:
                a = self.softmax(z)  # Output layer
            else:
                a = self.relu(z)     # Hidden layers
            self.activations.append(a)
        
        return a
    
    def backward(self, y):
        """Backward pass, computing gradients."""
        m = y.shape[1]  # batch size
        grads_w = []
        grads_b = []
        
        # Output layer: softmax + cross-entropy gradient
        dz = self.activations[-1] - y
        
        # Propagate backward through layers
        for i in reversed(range(len(self.weights))):
            dW = (1/m) * dz @ self.activations[i].T
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)
            
            grads_w.insert(0, dW)
            grads_b.insert(0, db)
            
            if i > 0:  # Not input layer
                dz = (self.weights[i].T @ dz) * self.relu_derivative(self.z_values[i-1])
        
        return grads_w, grads_b
    
    def update_weights(self, grads_w, grads_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]
\`\`\`

---

## 6. Why Initialization Matters

### The Problem of Vanishing/Exploding Gradients

If weights start too small: gradients shrink exponentially through layers → vanishing gradient
If weights start too large: gradients grow exponentially → exploding gradient

### Xavier Initialization

For tanh/sigmoid: $W \\sim \\mathcal{N}(0, \\frac{1}{n_{in}})$

**Intuition**: Keep variance of activations roughly constant across layers.

### He Initialization

For ReLU: $W \\sim \\mathcal{N}(0, \\frac{2}{n_{in}})$

**Why factor of 2?** ReLU zeros out half the activations, so we need larger weights to compensate.

---

## 7. Key Takeaways

1. **Non-linearity is essential**: Without it, deep networks are just linear models
2. **Backpropagation is efficient**: It computes ALL gradients in a single backward pass using the chain rule
3. **The gradient has a beautiful form**: output layer is (prediction - truth), hidden layers propagate this backward
4. **Initialization matters**: Use He for ReLU, Xavier for tanh/sigmoid
5. **Depth enables compositional learning**: Each layer builds on abstractions from the previous
`,

    "Decision Trees (Info Gain)": `
# Decision Trees & Information Theory

## 1. The Intuition: Playing 20 Questions

A decision tree is like playing 20 Questions optimally. You want to ask questions that **maximally reduce uncertainty** about the answer.

If someone is thinking of an animal, asking "Is it a mammal?" is better than "Is it a blue whale?" The first question eliminates ~half the possibilities; the second barely narrows things down.

**This is the core insight**: at each split, choose the feature and threshold that **reduce uncertainty the most**.

---

## 2. Quantifying Uncertainty: Entropy

### What is Entropy?

**Entropy** measures the "surprise" or "uncertainty" in a distribution. It comes from information theory.

$$H(S) = -\\sum_{c=1}^{C} p_c \\log_2(p_c)$$

where $p_c$ is the proportion of class $c$ in set $S$.

### Why This Formula?

**Intuition**: How many yes/no questions do you need to identify a sample?

- If all samples are class A: 0 bits (no uncertainty)
- If 50-50 between A and B: 1 bit (one yes/no question)
- If uniform across 8 classes: 3 bits ($\\log_2(8)$)

### Examples

| Distribution | Entropy |
|--------------|---------|
| [100%, 0%] | 0 bits (pure) |
| [50%, 50%] | 1 bit (maximum for binary) |
| [75%, 25%] | 0.81 bits |
| [25%, 25%, 25%, 25%] | 2 bits |

---

## 3. Information Gain: The Splitting Criterion

### The Goal

We want to split data so that the **child nodes are purer** than the parent.

**Information Gain** = Entropy before split - Weighted entropy after split

$$IG(S, A) = H(S) - \\sum_{v \\in Values(A)} \\frac{|S_v|}{|S|} H(S_v)$$

### Why Weighted Average?

If a split creates one pure node of 10 samples and one messy node of 990 samples, that's not a good split—the messy node dominates.

We weight by size to account for this.

### Example Walkthrough

Parent node: 50 spam, 50 not spam → $H = 1$ bit

Split on "contains 'FREE'":
- Yes (30 samples): 28 spam, 2 not spam → $H \\approx 0.35$ bits
- No (70 samples): 22 spam, 48 not spam → $H \\approx 0.89$ bits

Weighted entropy after: $\\frac{30}{100}(0.35) + \\frac{70}{100}(0.89) = 0.73$ bits

**Information Gain = 1 - 0.73 = 0.27 bits**

---

## 4. Gini Impurity: The Alternative

### Definition

$$Gini(S) = 1 - \\sum_{c=1}^{C} p_c^2$$

### Intuition

Gini measures the probability that a randomly chosen sample would be **incorrectly classified** if we randomly labeled it according to the class distribution.

### Gini vs Entropy

| Property | Gini | Entropy |
|----------|------|---------|
| Range | [0, 0.5] for binary | [0, 1] for binary |
| Computation | Slightly faster (no log) | Slightly slower |
| Behavior | Prefers bigger partitions | More balanced splits |
| In practice | Default in sklearn | ID3 algorithm |

**In practice**: They give similar results. Gini is slightly faster.

---

## 5. Building the Tree: The Recursive Algorithm

\`\`\`python
def build_tree(data, depth=0, max_depth=None):
    # Base cases
    if all_same_class(data) or (max_depth and depth >= max_depth):
        return LeafNode(majority_class(data))
    
    if no_features_left(data):
        return LeafNode(majority_class(data))
    
    # Find best split
    best_feature, best_threshold = find_best_split(data)
    
    if no_information_gain(best_feature):
        return LeafNode(majority_class(data))
    
    # Recursive construction
    left_data, right_data = split(data, best_feature, best_threshold)
    
    node = DecisionNode(best_feature, best_threshold)
    node.left = build_tree(left_data, depth + 1, max_depth)
    node.right = build_tree(right_data, depth + 1, max_depth)
    
    return node
\`\`\`

---

## 6. Overfitting and Pruning

### Why Trees Overfit

A fully grown tree can create a leaf for EVERY training sample, achieving 0 training error. But it will fail spectacularly on new data—it memorized, not learned.

### Pre-Pruning (Early Stopping)

- **max_depth**: Limit tree depth
- **min_samples_split**: Require minimum samples to split
- **min_samples_leaf**: Require minimum samples in leaves
- **max_features**: Consider only a subset of features

### Post-Pruning (Reduced Error Pruning)

1. Grow full tree
2. Evaluate performance on validation set
3. For each non-leaf node, try replacing subtree with leaf
4. Keep change if validation accuracy improves

---

## 7. Random Forests: Wisdom of Crowds

### The Problem with Single Trees

Single trees have **high variance**—small changes in data cause big changes in the tree.

### The Solution: Ensemble Averaging

Train many trees on different random subsets, then vote.

$$\\hat{y} = \\text{mode}\\{T_1(x), T_2(x), ..., T_B(x)\\}$$

### Two Key Randomizations

1. **Bootstrap sampling**: Each tree sees a random subset of data (with replacement)
2. **Feature randomization**: Each split considers a random subset of features

### Why This Works

The variance of an average is:
$$Var\\left(\\frac{1}{B}\\sum T_i\\right) = \\frac{\\sigma^2}{B} + \\frac{B-1}{B}\\rho\\sigma^2$$

Feature randomization **decorrelates** trees ($\\rho \\to 0$), making the variance term shrink!

---

## 8. Gradient Boosting: Learning from Mistakes

### The Idea

Instead of averaging independent trees, build them **sequentially**, where each tree corrects the errors of the previous ensemble.

$$F_m(x) = F_{m-1}(x) + \\gamma_m h_m(x)$$

where $h_m$ is trained on the **residuals** (or negative gradients).

### Why It Works

Each tree focuses on where the model is **currently wrong**. Early trees handle the easy patterns; later trees refine the hard cases.

### XGBoost Innovations

1. **Regularized objective**: Penalize tree complexity
2. **Second-order gradients**: Use Newton's method, not just gradients
3. **Efficient implementation**: Column subsampling, cache-aware access

---

## 9. Key Takeaways

1. **Entropy/Gini measure impurity**—we split to reduce them
2. **Information Gain selects the best feature**—the one that reduces uncertainty most
3. **Trees overfit easily**—use pruning or ensembles
4. **Random Forests reduce variance** via averaging + decorrelation
5. **Gradient Boosting reduces bias** via sequential error correction
`,

    "K-Means & EM": `
# K-Means Clustering & Expectation-Maximization

## 1. The Clustering Problem

You have data with no labels. You want to discover natural groupings.

**K-Means asks**: If I had to summarize this data with $K$ representative points (centroids), where should they be?

---

## 2. K-Means: The Algorithm

### The Objective

Minimize **within-cluster sum of squared distances**:

$$J = \\sum_{k=1}^{K} \\sum_{x \\in C_k} \\|x - \\mu_k\\|^2$$

where $\\mu_k$ is the centroid of cluster $k$.

### The Algorithm

\`\`\`
1. Initialize K centroids randomly
2. Repeat until convergence:
   a. ASSIGN: Each point to its nearest centroid
   b. UPDATE: Move each centroid to the mean of its assigned points
\`\`\`

### Why Does This Work?

Each step is **guaranteed to decrease or maintain J**.

**Assignment step**: Holding centroids fixed, the optimal assignment is to the nearest centroid (by definition of "minimize distance").

**Update step**: Holding assignments fixed, the optimal centroid is the mean (calculus: minimize sum of squared distances).

Since J is bounded below by 0 and decreases monotonically, it must converge.

---

## 3. The Initialization Problem

### Why It Matters

K-Means finds a **local minimum**, not the global one. Different starting points give different solutions.

**Bad initialization** can lead to:
- Empty clusters
- Clusters split across natural groups
- Suboptimal objective value

### K-Means++ Initialization

The smart way to initialize:

1. Choose first centroid uniformly at random
2. For each subsequent centroid, choose point $x$ with probability:
$$P(x) \\propto D(x)^2$$
where $D(x)$ is the distance to the nearest existing centroid.

**Why this works**: Points far from existing centroids are more likely to be chosen, spreading out the initial centroids.

**Guarantee**: K-Means++ is $O(\\log K)$-competitive with the optimal solution!

---

## 4. Limitations of K-Means

### Assumes Spherical Clusters

K-Means uses Euclidean distance, which creates spherical decision boundaries. Elongated or curved clusters will be split poorly.

### Hard Assignments

Each point belongs to exactly one cluster. What if a point is on the boundary?

### Must Specify K

How do you know how many clusters exist?

---

## 5. Gaussian Mixture Models: Soft Clustering

### The Generative Model

Assume data is generated by a **mixture of Gaussians**:

$$p(x) = \\sum_{k=1}^{K} \\pi_k \\mathcal{N}(x | \\mu_k, \\Sigma_k)$$

where:
- $\\pi_k$ = mixing coefficient (probability of cluster $k$)
- $\\mu_k$ = mean of cluster $k$
- $\\Sigma_k$ = covariance matrix of cluster $k$

### Soft Assignments

Instead of "point belongs to cluster 3", we have "point has 70% probability of cluster 3, 25% cluster 1, 5% cluster 2".

This is called the **responsibility**:
$$\\gamma_{nk} = P(z_n = k | x_n) = \\frac{\\pi_k \\mathcal{N}(x_n | \\mu_k, \\Sigma_k)}{\\sum_j \\pi_j \\mathcal{N}(x_n | \\mu_j, \\Sigma_j)}$$

---

## 6. The EM Algorithm: The Beautiful General Solution

### The Problem

We want to maximize the likelihood:
$$\\log p(X | \\theta) = \\sum_n \\log \\sum_k \\pi_k \\mathcal{N}(x_n | \\mu_k, \\Sigma_k)$$

The **sum inside the log** makes this hard to optimize directly.

### The Solution: Iterate Between Two Steps

**E-Step (Expectation)**: Compute responsibilities using current parameters
$$\\gamma_{nk} = \\frac{\\pi_k \\mathcal{N}(x_n | \\mu_k, \\Sigma_k)}{\\sum_j \\pi_j \\mathcal{N}(x_n | \\mu_j, \\Sigma_j)}$$

**M-Step (Maximization)**: Update parameters using responsibilities
$$\\mu_k = \\frac{\\sum_n \\gamma_{nk} x_n}{\\sum_n \\gamma_{nk}}$$
$$\\Sigma_k = \\frac{\\sum_n \\gamma_{nk} (x_n - \\mu_k)(x_n - \\mu_k)^T}{\\sum_n \\gamma_{nk}}$$
$$\\pi_k = \\frac{\\sum_n \\gamma_{nk}}{N}$$

### Why This Works: The ELBO

EM maximizes a **lower bound** on the log-likelihood (ELBO). Each iteration is guaranteed to increase this bound.

### K-Means as a Special Case

If you:
- Fix all covariances to $\\sigma^2 I$
- Take $\\sigma \\to 0$

The soft responsibilities become hard assignments, and EM reduces to K-Means!

---

## 7. Choosing K

### Elbow Method

Plot J (within-cluster variance) vs K. Look for an "elbow" where adding more clusters stops helping much.

### Silhouette Score

For each point, compute:
$$s = \\frac{b - a}{\\max(a, b)}$$
where $a$ = average distance to same-cluster points, $b$ = average distance to nearest other cluster.

- $s \\approx 1$: Point is well-clustered
- $s \\approx 0$: Point is on the boundary
- $s < 0$: Point might be in the wrong cluster

### BIC/AIC for GMM

These penalize model complexity, balancing fit vs. simplicity.

---

## 8. Key Takeaways

1. **K-Means minimizes within-cluster variance** via alternating assignment and update
2. **K-Means++ provides smart initialization** that's provably near-optimal
3. **GMM provides soft assignments** and can model elliptical clusters
4. **EM is the general algorithm** for maximum likelihood with latent variables
5. **K-Means is a special case of EM** with hard assignments and spherical clusters
`,

    "PCA & SVD (Eigenvectors)": `
# Principal Component Analysis & Singular Value Decomposition

## 1. The Dimensionality Problem

High-dimensional data is hard:
- **Visualization**: Can't plot 1000 dimensions
- **Computation**: Expensive to process
- **The Curse**: In high dimensions, all points become "far apart"

**PCA asks**: What are the **most important directions** in my data?

---

## 2. The Variance Maximization View

### The Intuition

If you had to project 2D data onto a 1D line, which line should you choose?

The line that **captures the most variance**—the direction where data is most "spread out."

### Mathematical Formulation

Given centered data $X$ (mean subtracted), find unit vector $u$ that maximizes variance of projections:

$$\\max_u Var(Xu) = \\max_u u^T X^T X u \\quad \\text{subject to } \\|u\\|^2 = 1$$

### Using Lagrange Multipliers

$$L = u^T X^T X u - \\lambda(u^T u - 1)$$

Taking derivative and setting to zero:
$$X^T X u = \\lambda u$$

**This is an eigenvalue problem!** The optimal $u$ is an eigenvector of $X^T X$.

### Which Eigenvector?

The variance along $u$ is:
$$u^T X^T X u = u^T (\\lambda u) = \\lambda$$

So **the eigenvector with the largest eigenvalue captures the most variance**.

---

## 3. The Covariance Matrix

### Definition

For centered data $X$ (n samples × p features):
$$\\Sigma = \\frac{1}{n-1} X^T X$$

Each entry $\\Sigma_{ij}$ is the covariance between features $i$ and $j$.

### Properties

1. **Symmetric**: $\\Sigma = \\Sigma^T$
2. **Positive semi-definite**: All eigenvalues ≥ 0
3. **Real eigenvalues**: Due to symmetry
4. **Orthogonal eigenvectors**: Also due to symmetry

---

## 4. The PCA Algorithm

\`\`\`python
import numpy as np

def pca(X, n_components):
    # 1. Center the data
    X_centered = X - X.mean(axis=0)
    
    # 2. Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # 3. Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort by eigenvalue (descending)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # 5. Select top k components
    components = eigenvectors[:, :n_components]
    
    # 6. Project data
    X_reduced = X_centered @ components
    
    return X_reduced, components, eigenvalues
\`\`\`

---

## 5. Singular Value Decomposition (SVD)

### The Decomposition

ANY matrix $X$ (m × n) can be written as:
$$X = U \\Sigma V^T$$

where:
- $U$ (m × m): Left singular vectors (orthonormal)
- $\\Sigma$ (m × n): Diagonal matrix of singular values
- $V$ (n × n): Right singular vectors (orthonormal)

### Connection to Eigenvalues

- $V$ contains eigenvectors of $X^T X$
- $U$ contains eigenvectors of $X X^T$
- Singular values $\\sigma_i = \\sqrt{\\lambda_i}$ where $\\lambda_i$ are eigenvalues of $X^T X$

### SVD for PCA

For centered data $X$:
- The right singular vectors $V$ are the principal components
- The singular values tell you the standard deviation along each component

**In practice, SVD is preferred over eigendecomposition** because:
1. Numerically more stable
2. Works directly on $X$, not $X^T X$
3. Doesn't require forming the covariance matrix

---

## 6. How Many Components?

### Explained Variance Ratio

$$\\text{Explained Variance Ratio}_k = \\frac{\\lambda_k}{\\sum_i \\lambda_i}$$

This tells you what fraction of total variance each component captures.

### The Scree Plot

Plot eigenvalues (or explained variance) vs component number. Look for an "elbow" where gains drop off.

### The 95% Rule

Keep enough components to explain 95% of variance:
$$\\sum_{k=1}^{d} \\frac{\\lambda_k}{\\sum_i \\lambda_i} \\geq 0.95$$

---

## 7. The Reconstruction View

### Low-Rank Approximation

Using only $k$ components, you can reconstruct the data:
$$\\hat{X} = X V_k V_k^T$$

where $V_k$ contains the first $k$ principal components.

### Eckart-Young Theorem

This reconstruction is the **best possible** rank-$k$ approximation in terms of Frobenius norm:
$$\\min_{\\text{rank}(\\hat{X}) \\leq k} \\|X - \\hat{X}\\|_F$$

---

## 8. Applications and Interpretations

### Dimensionality Reduction

Project from $p$ dimensions to $k$ dimensions for visualization or faster computation.

### Denoising

If noise is in low-variance directions, projecting to top components removes it.

### Feature Extraction

Principal components can reveal hidden structure. In face recognition, the top components are called "eigenfaces."

### Latent Semantics

In text analysis (LSA), SVD of term-document matrices reveals semantic topics.

---

## 9. Key Takeaways

1. **PCA finds directions of maximum variance** via eigendecomposition
2. **Principal components are eigenvectors** of the covariance matrix
3. **Eigenvalues tell you the variance** captured by each component
4. **SVD provides PCA** without computing the covariance matrix explicitly
5. **Reconstruction error is minimized** by the Eckart-Young theorem
`,

    "Ensembles (GBM/Boosting)": `
# Ensembles: Gradient Boosting & Boosting

## 1. The Ensemble Philosophy

Why settle for one model when you can have a committee?

- **Bagging (Bootstrap Aggregating)**: Reduce variance (Random Forest). Independent learners.
- **Boosting**: Reduce bias. Sequential learners correcting mistakes.
- **Stacking**: Combine different model types with a meta-learner.

---

## 2. Gradient Boosting Machines (GBM)

### The Intuition

We want to find $F(x)$ to minimize loss $L(y, F(x))$.
Instead of optimizing parameters $\\theta$, we optimize the **function** $F(x)$ directly in function space.

Gradient Descent in parameter space:
$$\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta L$$

Gradient Boosting in function space:
$$F_{t+1}(x) = F_t(x) - \\eta \\nabla_{F(x)} L$$

The "negative gradient" is the **residual** $(y - F(x))$ for squared error!

### The Algorithm

1. Initialize $F_0(x) = \\text{mean}(y)$
2. For $m = 1$ to $M$:
   - Compute pseudo-residuals: $r_{im} = -[\\frac{\\partial L(y_i, F(x_i))}{\\partial F(x_i)}]_{F(x)=F_{m-1}(x)}$
   - Train weak learner $h_m(x)$ (decision tree) to predict $r_{im}$
   - Find step size $\\gamma_m$
   - Update $F_m(x) = F_{m-1}(x) + \\gamma_m h_m(x)$

---

## 3. XGBoost vs LightGBM vs CatBoost

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Split Finding** | Histogram-based (approx) | Gradient-based One-Side Sampling (GOSS) | Ordered Boosting |
| **Tree Growth** | Level-wise (depth-first) | Leaf-wise (best-first) | Symmetric Trees |
| **Categorical** | One-hot required | Native support (Fisher) | Native support (Target Stats) |
| **Speed** | Fast | Very Fast | Fast / Slow training |
| **Accuracy** | High | High | Best on categorical |

### XGBoost Key Innovations

1. **Regularization** in objective: $L + \\gamma T + \\frac{1}{2}\\lambda ||w||^2$
2. **Second-order approximation**: Uses Hessian for faster convergence.
3. **Sparsity aware**: Learn default direction for missing values.

---

## 4. Key Takeaways

1. **Bagging reduces variance**, **Boosting reduces bias**.
2. **GBM is gradient descent in function space**.
3. **XGBoost/LightGBM** dominate tabular data competitions.
4. **CatBoost** handles categorical features without leakage.
`,

    "Word2Vec Embeddings": `
# Word2Vec: (See NLP Tab for Deep Dive)

(This topic is covered extensively in the NLP section. Please navigate there for the deep mathematical derivation!)
`,

    "Recommendation Systems": `
# Recommendation Systems: From First Principles

## 1. The Fundamental Problem

**Question**: You run a video streaming service with 1 million users and 10,000 movies. How do you recommend movies to users?

### Why This Is Hard

1. **Sparsity**: Each user watches ~100 movies. That's 100/10,000 = 1% of the catalog. The rating matrix is 99% empty!
2. **Scale**: 1M users × 10K movies = 10 billion user-item pairs to score
3. **Cold Start**: New users have no history. New items have no ratings.
4. **Diversity**: Generic recommendations ("everyone likes The Godfather") aren't useful

The core insight: **users who agreed in the past will agree in the future**.

---

## 2. Collaborative Filtering: From Neighbors to Matrices

### User-Based CF: The Intuition

**Idea**: Find users similar to you, recommend what they liked.

If Alice and Bob both loved movies {A, B, C}, and Alice also loved D, then Bob will probably like D.

**The Math**: Similarity between users $u$ and $v$:

$$\\text{sim}(u, v) = \\frac{\\sum_{i \\in I_{uv}} (r_{ui} - \\bar{r}_u)(r_{vi} - \\bar{r}_v)}{\\sqrt{\\sum_{i} (r_{ui} - \\bar{r}_u)^2} \\sqrt{\\sum_{i} (r_{vi} - \\bar{r}_v)^2}}$$

This is just the **Pearson correlation** of ratings on commonly-rated items.

**Prediction** for user $u$ on item $i$:
$$\\hat{r}_{ui} = \\bar{r}_u + \\frac{\\sum_{v \\in N(u)} \\text{sim}(u,v)(r_{vi} - \\bar{r}_v)}{\\sum_{v \\in N(u)} |\\text{sim}(u,v)|}$$

**Why this works**: Weighted average of how much similar users deviated from their mean.

### The Problem with Neighborhood Methods

- **Computational cost**: Finding k-nearest neighbors = $O(U^2 \\cdot I)$
- **Sparsity**: If users share few items, similarity is unreliable
- **No latent structure**: Doesn't capture that "action movies" are a meaningful category

---

## 3. Matrix Factorization: The Deep Insight

### The Core Assumption

Imagine there are $k$ latent factors (genres, artistic style, target audience).

Each user has preferences: $\\mathbf{p}_u \\in \\mathbb{R}^k$
Each item has attributes: $\\mathbf{q}_i \\in \\mathbb{R}^k$

**The hypothesis**: Rating = dot product of user and item factors
$$r_{ui} \\approx \\mathbf{p}_u^T \\mathbf{q}_i$$

Why dot product? If user likes "action" (high $p_u[\\text{action}]$) and item is "action" (high $q_i[\\text{action}]$), the product is large.

### The Optimization Problem

We want to find $P$ and $Q$ to minimize:

$$\\min_{P,Q} \\sum_{(u,i) \\in \\text{observed}} (r_{ui} - \\mathbf{p}_u^T \\mathbf{q}_i)^2 + \\lambda(\\|\\mathbf{p}_u\\|^2 + \\|\\mathbf{q}_i\\|^2)$$

**Why regularization?** Without it, the model can overfit by making embeddings arbitrarily large.

### Solving It: Alternating Least Squares (ALS)

The problem is **non-convex** in $(P, Q)$ jointly. But it's **convex** in $P$ if $Q$ is fixed!

**Algorithm**:
1. Initialize $Q$ randomly
2. Fix $Q$, solve for $P$ (this is just ridge regression!)
3. Fix $P$, solve for $Q$
4. Repeat until convergence

**Update for user $u$**:
$$\\mathbf{p}_u = (Q^T Q + \\lambda I)^{-1} Q^T \\mathbf{r}_u$$

### Implicit Feedback: Clicks Instead of Ratings

**Problem**: Most real data is clicks, views, purchases—not explicit ratings.

**Key difference**: Absence of interaction doesn't mean dislike—just unknown!

**Weighted objective**:
$$\\min_{P,Q} \\sum_{u,i} c_{ui}(y_{ui} - \\mathbf{p}_u^T \\mathbf{q}_i)^2$$

where $c_{ui}$ = confidence (e.g., number of views).

---

## 4. Two-Tower Neural Networks

### Why Neural Networks?

Matrix factorization is linear. Can't capture: "users who like sci-fi AND comedy".

### The Architecture

User tower: user features → dense embedding $\\mathbf{u}$
Item tower: item features → dense embedding $\\mathbf{i}$
Prediction: $\\hat{r}_{ui} = \\sigma(\\mathbf{u}^T \\mathbf{i})$

### Why Two Towers?

**Key insight**: Towers can be computed **independently**!

- Precompute all item embeddings offline
- At serving time, compute user embedding and do ANN search
- **Latency**: $O(d)$ + $O(\\log N)$ instead of $O(N \\cdot d)$

---

## 5. The Complete Pipeline

Real systems use a **funnel**:

### Stage 1: Candidate Generation (Recall)
- **Goal**: 1M items → 1000 candidates
- **Methods**: CF, two-tower retrieval
- **Latency**: 50ms

### Stage 2: Ranking (Precision)
- **Goal**: Score 1000 candidates precisely
- **Method**: Deep model with cross-features
- **Latency**: 100ms

### Stage 3: Re-ranking (Diversity)
- **Goal**: Ensure diverse results
- **Methods**: MMR, DPP

---

## 6. Cold Start Solutions

### New User
1. Ask for preferences (onboarding)
2. Use demographic data
3. Show popular items
4. Explore-exploit (Thompson sampling)

### New Item
1. Content-based filtering (metadata)
2. Item-to-item similarity
3. Give new items a boost

---

## 7. Key Takeaways

1. **Collaborative filtering**: Similar users have similar tastes
2. **Matrix factorization**: Discovers latent factors you can't see
3. **Two-tower architecture**: Enables scale via precomputation
4. **Real systems use pipelines**: Recall → Ranking → Re-ranking
5. **Online metrics matter most**: Optimize engagement, not RMSE
`,

    "Distributed Training": `
# Distributed Training: Scaling Beyond One GPU

## 1. The Fundamental Problem

**Single GPU memory**: ~24GB (A100) or ~80GB (H100)
**GPT-3**: 175B params × 4 bytes = 700GB
**LLaMA-70B**: 70B × 4 bytes = 280GB

**Conclusion**: We can't fit large models on one GPU!

Even if we could, training would take years on a single device.

---

## 2. Data Parallelism: The Simplest Approach

### The Idea

1. **Replicate** the model on N GPUs
2. Each GPU processes a **different batch**
3. **Aggregate gradients** across GPUs
4. Update all copies identically

### The Math

If effective batch size = B, and we have N GPUs:
- Each GPU sees B/N samples
- Each computes gradients on its subset
- Gradients are averaged across GPUs

$$\\nabla L_{\\text{total}} = \\frac{1}{N} \\sum_{i=1}^N \\nabla L_i$$

### AllReduce: The Key Operation

**Problem**: How do GPUs share gradients efficiently?

**Naïve approach**: Send all to GPU 0, average, broadcast back
- **Bottleneck**: GPU 0 receives $N \\times$ gradient size

**Ring-AllReduce**: Each GPU sends/receives equally

1. Arrange GPUs in a ring
2. Each GPU sends a chunk to the next
3. After N-1 steps, all GPUs have the sum
4. Then scatter the result

**Bandwidth**: $2(N-1)/N \\cdot D$ per GPU (nearly optimal!)

### When Data Parallelism Fails

When the **model itself** doesn't fit on one GPU!

---

## 3. Model Parallelism: Splitting the Model

### Pipeline Parallelism

**Idea**: Split model into stages, each on a different GPU.

Layer 1-10 → GPU 0
Layer 11-20 → GPU 1
Layer 21-30 → GPU 2

**Problem**: Pipeline bubble! While GPU 2 processes forward pass, GPU 0 is idle.

**Solution**: Micro-batching (GPipe)
- Split batch into micro-batches
- Interleave forward/backward passes
- **Bubble overhead**: $(P-1)/(M+P-1)$ where P=pipeline stages, M=micro-batches

### Tensor Parallelism

**Idea**: Split individual layers across GPUs.

For a linear layer $Y = XW$:
- Split $W$ column-wise: $W = [W_1, W_2]$
- GPU 0: $Y_1 = XW_1$
- GPU 1: $Y_2 = XW_2$
- Concatenate: $Y = [Y_1, Y_2]$

**Advantage**: No pipeline bubble
**Disadvantage**: Requires communication within each layer

---

## 4. ZeRO: Memory Optimization

### The Insight

In data parallelism, each GPU stores:
- Model parameters: $\\Psi$
- Gradients: $\\Psi$
- Optimizer states (Adam): $2\\Psi$

Total: $4\\Psi$ per GPU (all redundant!)

### ZeRO Stages

**ZeRO-1**: Partition optimizer states
- Memory: $4\\Psi/N + \\Psi$ per GPU

**ZeRO-2**: Also partition gradients
- Memory: $4\\Psi/N$ per GPU

**ZeRO-3**: Also partition parameters
- Memory: $4\\Psi/N$ per GPU
- Communicate parameters when needed

---

## 5. Memory Optimization Techniques

### Gradient Checkpointing

**Problem**: Activations for backprop consume memory

**Solution**: Don't store all activations. Recompute during backward.

**Trade-off**: 
- Memory: $O(\\sqrt{n})$ instead of $O(n)$
- Compute: ~33% overhead (one extra forward pass)

### Mixed Precision Training

**FP32**: 4 bytes per parameter
**FP16/BF16**: 2 bytes per parameter

**Algorithm**:
1. Store master weights in FP32
2. Forward/backward in FP16
3. Accumulate gradients in FP32
4. Update master weights

**Why BF16 over FP16?**
- BF16: Same exponent range as FP32 (less overflow)
- FP16: More precision but smaller range

---

## 6. Practical Considerations

### Synchronous vs Asynchronous SGD

**Synchronous**: All GPUs wait for slowest (stragglers)
- Consistent gradients
- Can be slow

**Asynchronous**: GPUs update independently
- Faster
- Stale gradients (may hurt convergence)

### Gradient Compression

**TopK Sparsification**: Only send top K% of gradients
- 99% compression possible with minimal accuracy loss
- Requires error feedback mechanism

---

## 7. Key Takeaways

1. **Data parallelism**: Replicate model, split data, aggregate gradients
2. **Ring-AllReduce**: Near-optimal communication for gradient aggregation
3. **Model parallelism**: When model doesn't fit on one GPU
4. **ZeRO**: Eliminate redundancy in optimizer states
5. **Mixed precision**: 2x memory savings, often faster
6. **Checkpointing**: Trade compute for memory
`,

    "Model Calibration": `
# Model Calibration: Are Your Probabilities Honest?

## 1. The Fundamental Problem

Your model predicts: $P(\\text{cancer}) = 0.8$

**Question**: Among all patients where the model predicted 80%, how many actually had cancer?

If the answer is 80%, your model is **calibrated**.
If the answer is 60%, your model is **overconfident**.

### Why This Matters

- **Medical diagnosis**: Overconfident predictions → unnecessary surgeries
- **Autonomous driving**: Underconfident → overly cautious behavior
- **Decision making**: Probabilities should mean what they say!

---

## 2. Measuring Calibration

### Reliability Diagrams

1. Bin predictions by confidence (0-10%, 10-20%, etc.)
2. For each bin, compute actual accuracy
3. Plot: predicted confidence vs actual accuracy
4. Perfect calibration = diagonal line

### Expected Calibration Error (ECE)

$$\\text{ECE} = \\sum_{m=1}^M \\frac{|B_m|}{n} |\\text{acc}(B_m) - \\text{conf}(B_m)|$$

where:
- $B_m$ = samples in bin $m$
- $\\text{acc}(B_m)$ = accuracy in bin
- $\\text{conf}(B_m)$ = average confidence in bin

**Intuition**: Weighted average of calibration error across bins.

---

## 3. Why Are Neural Networks Miscalibrated?

### Modern NNs Are Overconfident

**Observation** (Guo et al. 2017): Modern deep networks are significantly more overconfident than older, simpler models.

**Causes**:
1. **Increased capacity**: More parameters = more overfitting
2. **Lack of regularization**: BatchNorm, but not dropout
3. **NLL training**: Optimizes log-likelihood, not calibration
4. **No epistemic uncertainty**: Doesn't know what it doesn't know

---

## 4. Calibration Methods

### Temperature Scaling (Simplest)

**Idea**: Scale logits by a learned temperature $T$:

$$\\hat{p}_i = \\frac{e^{z_i/T}}{\\sum_j e^{z_j/T}}$$

- $T > 1$: Softens probabilities (reduces overconfidence)
- $T < 1$: Sharpens probabilities

**Training**: Learn $T$ on validation set to minimize NLL.

**Why it works**:
- Single parameter, can't overfit
- Preserves ranking (accuracy unchanged)
- Corrects systematic overconfidence

### Platt Scaling

Learn a logistic regression on top of logits:

$$\\hat{p} = \\sigma(a \\cdot z + b)$$

**When to use**: Binary classification, more flexibility than temperature.

### Isotonic Regression

**Non-parametric**: Learn a monotonic function mapping logits → calibrated probabilities.

**Advantage**: No assumptions about functional form
**Disadvantage**: Needs more validation data

---

## 5. Implementation

Temperature Scaling in Python:

\`\`\`python
import torch
import torch.nn as nn

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature
    
    def calibrate(self, logits, labels, lr=0.01, max_iter=50):
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr)
        
        def eval_fn():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss
        
        for _ in range(max_iter):
            optimizer.step(eval_fn)
        
        return self.temperature.item()
\`\`\`

---

## 6. Calibration vs Accuracy

**Key insight**: Calibration and accuracy are independent!

- A model can be accurate but miscalibrated
- A model can be calibrated but inaccurate

**Example**:
- Weather forecast: "30% chance of rain" every day
- If it rains 30% of days, perfectly calibrated!
- But not useful for deciding to bring an umbrella

---

## 7. Key Takeaways

1. **Calibration ≠ accuracy**: They're orthogonal properties
2. **Modern NNs are overconfident**: This is a known problem
3. **Temperature scaling**: Simple, effective, preserves accuracy
4. **Always evaluate calibration**: Especially for high-stakes decisions
5. **ECE measures miscalibration**: ~0.05 is typical for calibrated models
`
};




// DEDICATED CONTENT FOR ML SUB-TOPICS

mlContent["XGBoost vs LightGBM vs CatBoost"] = `
# XGBoost vs LightGBM vs CatBoost Comparison

## Core Differences

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Tree growth | Level-wise | Leaf-wise | Symmetric |
| Categorical handling | Manual encoding | Native | Native (best) |
| Speed | Slower | Fastest | Medium |
| Overfitting | Medium | Higher risk | Lower risk |

## XGBoost

- **Level-wise splitting**: Grows all leaves at same depth before next level
- **Regularized objective**: L1 + L2 regularization on leaf weights
- **Best for**: Structured data, competitions

## LightGBM

- **Leaf-wise splitting**: Grows leaf with highest gain
- **Histogram-based**: Bins continuous features for speed
- **GOSS**: Gradient-based One-Side Sampling - keeps high-gradient samples
- **Best for**: Large datasets, speed-critical applications

## CatBoost

- **Ordered boosting**: Uses permutation to avoid target leakage
- **Native categoricals**: No need for one-hot encoding
- **Symmetric trees**: All splits at same depth use same feature
- **Best for**: Categorical-heavy data, production without preprocessing

## When to Choose

- **XGBoost**: Default choice, well-documented
- **LightGBM**: When dataset is large (>100K rows)
- **CatBoost**: When you have many categorical features
`;

mlContent["Kernel Trick Mathematics"] = `
# The Kernel Trick: Feature Space Without Computing Features

## The Problem

Want to map data to high-dimensional space for linear separation, but computing $\\phi(x)$ explicitly is expensive or infinite-dimensional.

## The Insight

We only need **dot products** in the feature space:
$$K(x, y) = \\phi(x)^T \\phi(y)$$

If we can compute $K(x, y)$ directly, we never need $\\phi(x)$!

## Common Kernels

**Linear**: $K(x, y) = x^T y$

**Polynomial**: $K(x, y) = (x^T y + c)^d$
- Maps to all polynomial features up to degree d

**RBF (Gaussian)**: $K(x, y) = \\exp(-\\gamma \\|x - y\\|^2)$
- Infinite-dimensional feature space!

## Why RBF Works

The RBF kernel corresponds to an infinite Taylor expansion:
$$e^{-\\gamma \\|x-y\\|^2} = \\sum_{k=0}^\\infty \\frac{(-\\gamma)^k}{k!} \\|x-y\\|^{2k}$$

Each term is a dot product in a different polynomial feature space.

## The Math Behind SVMs

The SVM dual problem only uses $x_i^T x_j$:
$$\\max_\\alpha \\sum_i \\alpha_i - \\frac{1}{2}\\sum_{i,j} \\alpha_i \\alpha_j y_i y_j K(x_i, x_j)$$

Replace $x_i^T x_j$ with $K(x_i, x_j)$ → nonlinear SVM!
`;

mlContent["Activation Functions (ReLU, Sigmoid, Tanh)"] = `
# Activation Functions Deep Dive

## Why Non-linearity?

Without activations, a deep network collapses to a single linear transformation:
$$W_2(W_1 x) = (W_2 W_1) x = W' x$$

## Sigmoid: $\\sigma(x) = \\frac{1}{1+e^{-x}}$

**Output**: (0, 1)
**Gradient**: $\\sigma'(x) = \\sigma(x)(1 - \\sigma(x))$
**Problem**: Vanishing gradients when |x| is large (gradient ≈ 0)

## Tanh: $\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$

**Output**: (-1, 1) — zero-centered!
**Gradient**: $\\tanh'(x) = 1 - \\tanh^2(x)$
**Better than sigmoid**: Centered outputs, stronger gradients

## ReLU: $\\text{ReLU}(x) = \\max(0, x)$

**Output**: [0, ∞)
**Gradient**: 0 or 1 — no vanishing!
**Problem**: Dead ReLUs (neurons that never activate)

## Leaky ReLU: $\\max(\\alpha x, x)$ for small $\\alpha$

Fixes dead ReLU problem by allowing small negative gradient.

## GELU: Used in Transformers

$$\\text{GELU}(x) = x \\cdot \\Phi(x)$$

where $\\Phi$ is the CDF of standard normal. Smooth approximation to ReLU.

## When to Use What

- **Hidden layers**: ReLU (fast, works well)
- **Output (binary)**: Sigmoid
- **Output (multi-class)**: Softmax
- **Transformers**: GELU or SiLU (Swish)
`;

mlContent["Expectation Maximization Algorithm"] = `
# Expectation Maximization (EM) Algorithm

## The Problem

Maximize likelihood when there are **latent variables**:
$$\\max_\\theta \\log p(X | \\theta)$$

But $p(X | \\theta) = \\sum_Z p(X, Z | \\theta)$ is intractable to maximize directly.

## The EM Solution

Iterate between:

**E-step**: Compute expected latent variable values given current parameters
$$Q(\\theta | \\theta^{(t)}) = E_{Z|X,\\theta^{(t)}}[\\log p(X, Z | \\theta)]$$

**M-step**: Maximize expected complete-data log-likelihood
$$\\theta^{(t+1)} = \\arg\\max_\\theta Q(\\theta | \\theta^{(t)})$$

## Example: Gaussian Mixture Model

**E-step**: Compute responsibility of each cluster:
$$\\gamma_{ik} = \\frac{\\pi_k \\mathcal{N}(x_i | \\mu_k, \\Sigma_k)}{\\sum_j \\pi_j \\mathcal{N}(x_i | \\mu_j, \\Sigma_j)}$$

**M-step**: Update parameters:
$$\\mu_k = \\frac{\\sum_i \\gamma_{ik} x_i}{\\sum_i \\gamma_{ik}}$$

## Why EM Converges

Each iteration increases the likelihood (or keeps it same). Proof uses Jensen's inequality on the log function.

## Limitations

- Converges to **local** maximum (not global)
- Sensitive to initialization
- Can be slow (many iterations needed)
`;

mlContent["SVD vs PCA Relationship"] = `
# SVD vs PCA: The Deep Connection

## PCA Recap

Find directions of maximum variance:
$$\\max_w \\text{Var}(Xw) = \\max_w w^T X^T X w$$

Subject to $\\|w\\| = 1$. Solution: eigenvectors of covariance matrix $X^T X$.

## SVD Definition

Any matrix $X$ can be decomposed:
$$X = U \\Sigma V^T$$

- $U$: left singular vectors (orthonormal)
- $\\Sigma$: singular values (diagonal)
- $V$: right singular vectors (orthonormal)

## The Connection

If $X$ is centered, then:
- **PCA loadings** = columns of $V$ (right singular vectors)
- **PCA scores** = $X V = U \\Sigma$
- **Eigenvalues of covariance** = $\\frac{\\sigma_i^2}{n-1}$

## Why SVD is Better Numerically

- PCA requires computing $X^T X$ first (squares condition number)
- SVD works directly on $X$ (more stable)
- SVD can handle non-square matrices

## Dimensionality Reduction

Keep top $k$ components:
$$X_k = U_k \\Sigma_k V_k^T$$

This is the **best rank-k approximation** in Frobenius norm!
`;

mlContent["Mathematical Proof of Bias-Variance Decomposition"] = `
# Bias-Variance Decomposition: Full Proof

## Setup

For regression with squared loss, decompose expected error:
$$E[(y - \\hat{f}(x))^2]$$

where expectation is over training sets and noise.

## The Derivation

Let $f(x)$ = true function, $\\epsilon$ = noise with $E[\\epsilon] = 0$, $\\text{Var}(\\epsilon) = \\sigma^2$.

$$E[(y - \\hat{f})^2] = E[(f + \\epsilon - \\hat{f})^2]$$

Expand:
$$= E[(f - \\hat{f})^2] + E[\\epsilon^2] + 2E[(f - \\hat{f})\\epsilon]$$

The cross term is 0 because $\\epsilon$ is independent of $\\hat{f}$:
$$= E[(f - \\hat{f})^2] + \\sigma^2$$

Now decompose the first term using $E[\\hat{f}]$:
$$E[(f - \\hat{f})^2] = E[(f - E[\\hat{f}] + E[\\hat{f}] - \\hat{f})^2]$$

$$= (f - E[\\hat{f}])^2 + E[(\\hat{f} - E[\\hat{f}])^2]$$

## Final Result

$$\\text{Error} = \\underbrace{(f - E[\\hat{f}])^2}_{\\text{Bias}^2} + \\underbrace{E[(\\hat{f} - E[\\hat{f}])^2]}_{\\text{Variance}} + \\underbrace{\\sigma^2}_{\\text{Irreducible}}$$

## Key Insight

- **Bias**: How wrong is our model on average?
- **Variance**: How much does our model change with different training data?
- **Trade-off**: Complex models = low bias, high variance
`;

// New L5+ content with dedicated explanations

mlContent["Backpropagation Calculus (Chain Rule)"] = `
# Backpropagation: Chain Rule in Action

## The Core Idea

Backpropagation computes gradients using the **chain rule** of calculus, propagating error from output to input layer by layer.

## Mathematical Setup

For a simple network: $y = f_3(f_2(f_1(x)))$

Loss: $L = \\ell(y, \\hat{y})$

**Chain rule**:
$$\\frac{\\partial L}{\\partial W_1} = \\frac{\\partial L}{\\partial f_3} \\cdot \\frac{\\partial f_3}{\\partial f_2} \\cdot \\frac{\\partial f_2}{\\partial f_1} \\cdot \\frac{\\partial f_1}{\\partial W_1}$$

## Gradient Flow Example

For layer $i$ with activation $a_i = \\sigma(W_i a_{i-1} + b_i)$:

**Forward**:
$$z_i = W_i a_{i-1} + b_i$$
$$a_i = \\sigma(z_i)$$

**Backward**:
$$\\delta_i = \\frac{\\partial L}{\\partial z_i} = \\frac{\\partial L}{\\partial a_i} \\cdot \\sigma'(z_i)$$
$$\\frac{\\partial L}{\\partial W_i} = \\delta_i \\cdot a_{i-1}^T$$
$$\\frac{\\partial L}{\\partial a_{i-1}} = W_i^T \\delta_i$$

## Key Insight

Each layer passes gradients to the previous layer via $W^T$ (transposed weights). This is why weight initialization matters—bad init can cause vanishing/exploding gradients.
`;

mlContent["RBF, Polynomial, Linear Kernels"] = `
# Kernel Functions: Linear, Polynomial, RBF

## Linear Kernel

$$K(x, y) = x^T y$$

- Feature space: original input space
- Use when: data is linearly separable
- Fastest but least flexible

## Polynomial Kernel

$$K(x, y) = (\\gamma x^T y + c)^d$$

- Feature space: all polynomial terms up to degree d
- Parameters: degree d, coefficient c, scale $\\gamma$
- For $d=2$, $(x_1, x_2) \\to (x_1^2, x_2^2, \\sqrt{2}x_1x_2, \\sqrt{2c}x_1, \\sqrt{2c}x_2, c)$

## RBF (Gaussian) Kernel

$$K(x, y) = \\exp(-\\gamma \\|x - y\\|^2)$$

- Feature space: **infinite dimensional**
- $\\gamma$ controls "reach" of each training point
- High $\\gamma$: each point only affects local area (overfitting risk)
- Low $\\gamma$: each point affects wide area (underfitting risk)

## Choosing Kernels

1. **Start with RBF**: Usually works well
2. **Use linear for high-dimensional sparse data**: (text, genomics)
3. **Polynomial for interaction features**: when you know lower-degree interactions matter
`;

mlContent["The Gauss-Markov Theorem (BLUE)"] = `
# Gauss-Markov Theorem: Best Linear Unbiased Estimator

## The Setup

Linear model: $y = X\\beta + \\epsilon$

Assumptions:
1. $E[\\epsilon] = 0$ (errors have zero mean)
2. $\\text{Var}(\\epsilon) = \\sigma^2 I$ (homoscedasticity, no correlation)
3. $X$ is fixed and full rank

## The Theorem

The OLS estimator $\\hat{\\beta} = (X^T X)^{-1} X^T y$ is **BLUE**:
- **B**est: minimum variance among
- **L**inear: estimators that are linear in y
- **U**nbiased: $E[\\hat{\\beta}] = \\beta$
- **E**stimator

## Proof Sketch

Any linear unbiased estimator has form $\\tilde{\\beta} = Cy$ where $CX = I$.

Let $C = (X^T X)^{-1} X^T + D$ for some D where $DX = 0$.

$$\\text{Var}(\\tilde{\\beta}) = \\sigma^2 CC^T = \\text{Var}(\\hat{\\beta}) + \\sigma^2 DD^T$$

Since $DD^T$ is positive semi-definite, OLS has minimum variance.

## Key Insight

Gauss-Markov only guarantees best among LINEAR estimators. Biased estimators (Ridge, Lasso) can have lower MSE through bias-variance tradeoff.
`;

mlContent["Closed Form: Normal Equation (Derivation & Matrix Calculus)"] = `
# Normal Equation Derivation

## The Objective

Minimize squared error:
$$L(\\beta) = \\|y - X\\beta\\|^2 = (y - X\\beta)^T (y - X\\beta)$$

## Matrix Calculus

Expand:
$$L = y^T y - 2\\beta^T X^T y + \\beta^T X^T X \\beta$$

Take gradient with respect to $\\beta$:
$$\\frac{\\partial L}{\\partial \\beta} = -2 X^T y + 2 X^T X \\beta$$

Set to zero:
$$X^T X \\beta = X^T y$$

## The Normal Equation

$$\\hat{\\beta} = (X^T X)^{-1} X^T y$$

## Geometric Interpretation

$X\\hat{\\beta}$ is the **projection** of $y$ onto the column space of $X$.

The residual $y - X\\hat{\\beta}$ is orthogonal to all columns of $X$:
$$X^T (y - X\\hat{\\beta}) = 0$$

This is exactly the normal equation!

## When to Use

- Small datasets (n < 10K): Direct solution is efficient
- Large datasets: Use gradient descent (computing $(X^T X)^{-1}$ is O(p³))
`;

mlContent["MLE Derivation for Logistic Regression"] = `
# MLE for Logistic Regression

## The Model

$$P(y=1|x) = \\sigma(w^T x) = \\frac{1}{1 + e^{-w^T x}}$$

## Likelihood Function

For n samples with labels $y_i \\in \\{0, 1\\}$:

$$L(w) = \\prod_{i=1}^n P(y_i | x_i) = \\prod_{i=1}^n \\sigma(w^T x_i)^{y_i} (1 - \\sigma(w^T x_i))^{1-y_i}$$

## Log-Likelihood

$$\\ell(w) = \\sum_{i=1}^n [y_i \\log \\sigma(w^T x_i) + (1-y_i) \\log(1 - \\sigma(w^T x_i))]$$

## Negative Log-Likelihood = Cross-Entropy

$$\\text{NLL} = -\\ell(w) = -\\sum_{i=1}^n [y_i \\log \\hat{p}_i + (1-y_i) \\log(1 - \\hat{p}_i)]$$

This is exactly the **binary cross-entropy loss**!

## Gradient

Using $\\sigma'(z) = \\sigma(z)(1-\\sigma(z))$:

$$\\nabla_w \\ell = \\sum_{i=1}^n (y_i - \\sigma(w^T x_i)) x_i = X^T (y - \\hat{p})$$

## Key Insight

MLE maximizes likelihood. Minimizing cross-entropy = maximizing log-likelihood. Same optimization, different perspective!
`;

mlContent["Ridge (L2): Lagrangian Derivation & Handling Multicollinearity"] = `
# Ridge Regression: Derivation and Purpose

## The Problem: Multicollinearity

When features are highly correlated, $X^T X$ is nearly singular:
- $(X^T X)^{-1}$ has huge values
- Small changes in data → huge changes in coefficients
- High variance!

## The Ridge Objective

Add L2 penalty:
$$L(\\beta) = \\|y - X\\beta\\|^2 + \\lambda \\|\\beta\\|^2$$

## Derivation

$$\\nabla L = -2X^T(y - X\\beta) + 2\\lambda\\beta = 0$$

$$X^T X \\beta + \\lambda \\beta = X^T y$$

$$(X^T X + \\lambda I)\\beta = X^T y$$

## The Ridge Solution

$$\\hat{\\beta}_{ridge} = (X^T X + \\lambda I)^{-1} X^T y$$

## Why It Works

Adding $\\lambda I$ to $X^T X$:
1. Makes the matrix **always invertible** (smallest eigenvalue ≥ λ)
2. **Shrinks coefficients** toward zero
3. **Reduces variance** at cost of some bias

## Connection to Bayesian Inference

Ridge = MAP estimate with Gaussian prior:
$$\\beta \\sim N(0, \\frac{\\sigma^2}{\\lambda} I)$$

## Key Insight

λ controls bias-variance tradeoff:
- λ=0: OLS (high variance)
- λ→∞: β→0 (high bias)
`;

// Dedicated content for L5+ topics
mlContent["Collaborative Filtering"] = `
# Collaborative Filtering

## The Core Idea

Predict user preferences based on similar users' preferences.

**Assumption**: Users who agreed in the past will agree in the future.

## User-Based CF

1. Find users similar to target user
2. Aggregate their ratings for items target hasn't seen

$$\\hat{r}_{ui} = \\bar{r}_u + \\frac{\\sum_{v \\in N(u)} sim(u,v)(r_{vi} - \\bar{r}_v)}{\\sum_{v \\in N(u)} |sim(u,v)|}$$

## Item-Based CF

Instead of similar users, find similar items:

$$\\hat{r}_{ui} = \\frac{\\sum_{j \\in N(i)} sim(i,j) \\cdot r_{uj}}{\\sum_{j \\in N(i)} |sim(i,j)|}$$

## Similarity Metrics

- **Cosine**: $sim(u,v) = \\frac{r_u \\cdot r_v}{\\|r_u\\| \\|r_v\\|}$
- **Pearson**: Correlation of ratings
- **Jaccard**: For implicit feedback

## Limitations

- Cold start: New users/items have no history
- Sparsity: Most users rate few items
- Scalability: O(U²) or O(I²)
`;

mlContent["Matrix Factorization"] = `
# Matrix Factorization for Recommendations

## The Idea

Decompose the user-item rating matrix into two low-rank matrices:

$$R \\approx P Q^T$$

- $P \\in \\mathbb{R}^{U \\times k}$: User embeddings
- $Q \\in \\mathbb{R}^{I \\times k}$: Item embeddings
- k: Number of latent factors

## Prediction

$$\\hat{r}_{ui} = p_u^T q_i$$

## Objective Function

$$\\min_{P,Q} \\sum_{(u,i) \\in \\text{observed}} (r_{ui} - p_u^T q_i)^2 + \\lambda(\\|p_u\\|^2 + \\|q_i\\|^2)$$

## Optimization

**SGD update**:
$$p_u \\leftarrow p_u + \\eta(e_{ui} q_i - \\lambda p_u)$$
$$q_i \\leftarrow q_i + \\eta(e_{ui} p_u - \\lambda q_i)$$

where $e_{ui} = r_{ui} - \\hat{r}_{ui}$

**ALS (Alternating Least Squares)**:
- Fix Q, solve for P (closed form)
- Fix P, solve for Q
- Repeat

## Key Insight

Matrix factorization captures latent factors like "action-level" or "arthouse-level" that users and items share.
`;

mlContent["Data Parallelism"] = `
# Data Parallelism for Distributed Training

## The Idea

Split data across workers, each has full model copy.

1. Each worker computes gradients on its batch
2. Synchronize gradients across workers
3. All workers update with same gradient

## AllReduce

Efficient gradient aggregation:
- Sum all gradients across workers
- Send sum back to all workers

**Ring-AllReduce**: O(N) per worker, regardless of # workers

## Implementation

\`\`\`python
# Each worker
for batch in local_data:
    loss = model(batch)
    grads = loss.backward()
    
    # Synchronize
    global_grads = all_reduce(grads, op='mean')
    
    optimizer.step(global_grads)
\`\`\`

## Scaling Laws

- **Perfect scaling**: 8 GPUs = 8x faster
- **Reality**: Communication overhead limits speedup
- **Large batch size**: Need to adjust learning rate

## Key Insight

Data parallelism works because gradient is sum over examples:
$$\\nabla L = \\sum_i \\nabla L_i = \\text{sum of worker gradients}$$
`;

mlContent["Temperature Scaling"] = `
# Temperature Scaling for Calibration

## The Problem

Modern neural networks are **overconfident**. A model might predict 95% probability for a class it's wrong about.

## What is Calibration?

A model is calibrated if:
$$P(y = \\hat{y} | \\hat{p} = p) = p$$

"When I say 80% confident, I'm right 80% of the time."

## Temperature Scaling

Scale logits before softmax:

$$\\hat{p} = \\text{softmax}(z / T)$$

where T > 1 makes probabilities more uniform (less confident).

## Finding Optimal T

On validation set, minimize negative log-likelihood:

$$T^* = \\arg\\min_T -\\sum_i \\log \\text{softmax}(z_i / T)[y_i]$$

This is a 1D optimization (very fast).

## Key Insight

Temperature scaling preserves accuracy (same argmax) while fixing confidence. It's the simplest calibration method that works.
`;

mlContent["Expected Calibration Error"] = `
# Expected Calibration Error (ECE)

## Definition

ECE measures how well predicted probabilities match actual outcomes.

$$\\text{ECE} = \\sum_{b=1}^B \\frac{|B_b|}{n} |\\text{acc}(B_b) - \\text{conf}(B_b)|$$

where:
- $B_b$: Predictions in bin b (e.g., confidence 0.8-0.9)
- $\\text{acc}(B_b)$: Accuracy within bin
- $\\text{conf}(B_b)$: Average confidence in bin

## Reliability Diagram

Visual representation:
- X-axis: Predicted confidence
- Y-axis: Actual accuracy
- Perfect calibration: diagonal line

## Interpretation

- ECE = 0: Perfect calibration
- ECE = 0.1: On average, 10% gap between confidence and accuracy

## Limitations of ECE

- Sensitive to number of bins
- Can't distinguish overconfidence from underconfidence
- Alternative: Maximum Calibration Error (MCE)
`;

export default mlContent;


