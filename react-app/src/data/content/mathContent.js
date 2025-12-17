// Comprehensive Math Content for Study Modal
// Each topic has DEEP explanations with intuition, derivations, and the "why" behind everything

export const mathContent = {
    "Probability & Statistics": `
# Probability & Statistics for Machine Learning

## 1. Why Probability in ML?

Machine learning is fundamentally about **uncertainty**. We never know the true relationship between inputs and outputs—we can only estimate it from finite, noisy data.

Probability provides the language to:
- Quantify uncertainty in predictions
- Express what we know vs. don't know
- Make optimal decisions under uncertainty

---

## 2. Random Variables: The Foundation

A random variable maps outcomes to numbers. It's not random (it's a function!), but its INPUT is random.

**Example**: Roll a die. Let $X$ = "number showing."
- $P(X = 3) = 1/6$
- $E[X] = 3.5$

### Discrete vs. Continuous

| Type | Probability | Sum/Integral | Example |
|------|-------------|--------------|---------|
| Discrete | $P(X = x)$ | $\\sum_x P(X=x) = 1$ | Coin flips |
| Continuous | $P(X \\in [a,b]) = \\int_a^b f(x)dx$ | $\\int_{-\\infty}^{\\infty} f(x)dx = 1$ | Height |

For continuous: $P(X = \\text{exactly } 1.7) = 0$. We can only talk about ranges!

---

## 3. The Gaussian Distribution: Why It's Everywhere

$$f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)$$

### Why Is It So Common?

**Central Limit Theorem**: If $X_1, X_2, ..., X_n$ are i.i.d. with mean $\\mu$ and variance $\\sigma^2$:

$$\\bar{X}_n = \\frac{1}{n}\\sum_{i=1}^n X_i \\xrightarrow{d} \\mathcal{N}\\left(\\mu, \\frac{\\sigma^2}{n}\\right)$$

**In words**: The sum of many small, independent effects is approximately Gaussian, regardless of the original distribution.

This explains why:
- Measurement errors (sum of many factors) are Gaussian
- Stock returns (many market factors) are approximately Gaussian
- Physical quantities (many atomic interactions) are Gaussian

---

## 4. Bayes' Theorem: The Foundation of Inference

$$P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}$$

### The ML Interpretation

$$\\underbrace{P(\\theta | D)}_{\\text{Posterior}} = \\frac{\\overbrace{P(D | \\theta)}^{\\text{Likelihood}} \\cdot \\overbrace{P(\\theta)}^{\\text{Prior}}}{\\underbrace{P(D)}_{\\text{Evidence}}}$$

- **Prior**: What you believed before seeing data
- **Likelihood**: How probable the data is given a hypothesis
- **Posterior**: Updated belief after seeing data
- **Evidence**: Normalizing constant (often ignored in optimization)

### Example: Medical Testing

- Disease prevalence: $P(D) = 0.001$ (1 in 1000)
- Test sensitivity: $P(+|D) = 0.99$ (catches 99% of cases)
- False positive rate: $P(+|\\neg D) = 0.05$

**Question**: You test positive. What's $P(D|+)$?

$$P(D|+) = \\frac{P(+|D) \\cdot P(D)}{P(+|D) \\cdot P(D) + P(+|\\neg D) \\cdot P(\\neg D)}$$

$$= \\frac{0.99 \\times 0.001}{0.99 \\times 0.001 + 0.05 \\times 0.999} = \\frac{0.00099}{0.05094} \\approx 0.019$$

**Only 2%!** The prior (rare disease) dominates.

---

## 5. Expectation and Variance

### Expectation: The Average Value

$$E[X] = \\sum_x x \\cdot P(X=x) = \\int x \\cdot f(x) dx$$

**Key properties**:
- $E[aX + b] = aE[X] + b$ (linearity)
- $E[XY] = E[X]E[Y]$ **only if independent**

### Variance: Spread Around the Mean

$$\\text{Var}(X) = E[(X - \\mu)^2] = E[X^2] - (E[X])^2$$

**Key properties**:
- $\\text{Var}(aX + b) = a^2 \\text{Var}(X)$ (shift doesn't change variance)
- $\\text{Var}(X + Y) = \\text{Var}(X) + \\text{Var}(Y) + 2\\text{Cov}(X,Y)$

---

## 6. Covariance and Correlation

$$\\text{Cov}(X, Y) = E[(X - \\mu_X)(Y - \\mu_Y)] = E[XY] - E[X]E[Y]$$

$$\\text{Corr}(X, Y) = \\frac{\\text{Cov}(X, Y)}{\\sigma_X \\sigma_Y} \\in [-1, 1]$$

### Interpretation

- **Correlation = 1**: Perfect positive linear relationship
- **Correlation = 0**: No LINEAR relationship (could still be dependent!)
- **Correlation = -1**: Perfect negative linear relationship

**Warning**: $X = \\sin(t)$, $Y = \\cos(t)$ have correlation ≈ 0 but are completely dependent!

---

## 7. Key Takeaways

1. **CLT explains why Gaussians are everywhere**—sums of independent effects become Gaussian
2. **Bayes' theorem updates beliefs with evidence**—prior × likelihood ∝ posterior
3. **Correlation ≠ dependence**—zero correlation can still mean strong nonlinear relationships
4. **Variance adds for independent variables**—but covariance complicates things
`,

    "Matrix Calculus": `
# Matrix Calculus for Deep Learning

## 1. Why Matrix Calculus?

Deep learning is about computing gradients of loss functions with respect to weight matrices. You can't do this without matrix calculus.

The key question: If $L$ is a scalar loss and $W$ is a matrix of parameters, what is $\\frac{\\partial L}{\\partial W}$?

---

## 2. Notation: The Layout Convention

There are two conventions. We'll use the **numerator layout** (gradient is a row vector):

$$\\nabla_x f = \\frac{\\partial f}{\\partial x^T} = \\begin{bmatrix} \\frac{\\partial f}{\\partial x_1} & \\cdots & \\frac{\\partial f}{\\partial x_n} \\end{bmatrix}$$

Or the **denominator layout** (gradient is a column vector, matching $x$):

$$\\nabla_x f = \\begin{bmatrix} \\frac{\\partial f}{\\partial x_1} \\\\ \\vdots \\\\ \\frac{\\partial f}{\\partial x_n} \\end{bmatrix}$$

Deep learning typically uses denominator layout (gradient has same shape as parameter).

---

## 3. The Core Identities You Need

### Scalar-by-Vector

For $f: \\mathbb{R}^n \\to \\mathbb{R}$:

| Function | Gradient |
|----------|----------|
| $f = a^T x = x^T a$ | $\\nabla_x f = a$ |
| $f = x^T x = \\|x\\|^2$ | $\\nabla_x f = 2x$ |
| $f = x^T A x$ | $\\nabla_x f = (A + A^T)x = 2Ax$ (if $A$ symmetric) |

### The Least Squares Gradient

The most important identity for ML:

$$f = \\|Ax - b\\|^2 = (Ax - b)^T(Ax - b)$$

Let's derive the gradient step by step:

$$f = x^T A^T A x - 2b^T A x + b^T b$$

Taking the gradient:
$$\\nabla_x f = 2A^T A x - 2A^T b = 2A^T(Ax - b)$$

**This appears in linear regression, gradient descent, and countless other places.**

---

## 4. The Jacobian: Vector-to-Vector Derivatives

For $f: \\mathbb{R}^n \\to \\mathbb{R}^m$:

$$J = \\frac{\\partial f}{\\partial x} = \\begin{bmatrix} \\frac{\\partial f_1}{\\partial x_1} & \\cdots & \\frac{\\partial f_1}{\\partial x_n} \\\\ \\vdots & \\ddots & \\vdots \\\\ \\frac{\\partial f_m}{\\partial x_1} & \\cdots & \\frac{\\partial f_m}{\\partial x_n} \\end{bmatrix}$$

The Jacobian is the matrix of all partial derivatives. It's the multivariable version of the derivative.

---

## 5. The Chain Rule for Matrices

If $L = f(g(x))$ where $g: \\mathbb{R}^n \\to \\mathbb{R}^m$ and $f: \\mathbb{R}^m \\to \\mathbb{R}$:

$$\\nabla_x L = J_g^T \\nabla_g L$$

**In words**: Multiply the Jacobian transpose by the upstream gradient.

### Example: Neural Network Layer

Forward: $z = Wx + b$, then $a = \\sigma(z)$

Backward: Given $\\frac{\\partial L}{\\partial a}$,

$$\\frac{\\partial L}{\\partial z} = \\frac{\\partial L}{\\partial a} \\odot \\sigma'(z)$$
$$\\frac{\\partial L}{\\partial W} = \\frac{\\partial L}{\\partial z} x^T$$
$$\\frac{\\partial L}{\\partial x} = W^T \\frac{\\partial L}{\\partial z}$$

---

## 6. The Hessian: Second Derivatives

For $f: \\mathbb{R}^n \\to \\mathbb{R}$:

$$H = \\nabla^2 f = \\begin{bmatrix} \\frac{\\partial^2 f}{\\partial x_1^2} & \\cdots & \\frac{\\partial^2 f}{\\partial x_1 \\partial x_n} \\\\ \\vdots & \\ddots & \\vdots \\\\ \\frac{\\partial^2 f}{\\partial x_n \\partial x_1} & \\cdots & \\frac{\\partial^2 f}{\\partial x_n^2} \\end{bmatrix}$$

### Why It Matters

- **Convexity**: $f$ is convex iff $H \\succeq 0$ (positive semidefinite)
- **Curvature**: Eigenvalues of $H$ tell you how curved $f$ is in each direction
- **Newton's method**: Uses $H^{-1}$ for fast optimization

---

## 7. Key Takeaways

1. **Gradient has same shape as parameter** (denominator layout)
2. **$\\nabla_x (Ax-b)^T(Ax-b) = 2A^T(Ax-b)$** — memorize this!
3. **Chain rule: multiply by Jacobian transpose**
4. **Hessian determines convexity** — positive definite = strictly convex
`,

    "Linear Algebra": `
# Linear Algebra: The Language of Machine Learning

## 1. Why Linear Algebra?

Machine learning is fundamentally about:
- **Data as matrices**: Each row is a sample, each column is a feature
- **Operations as matrix multiplications**: Neural network layers are just $Wx + b$
- **Decompositions**: PCA, SVD, eigendecomposition reveal hidden structure

---

## 2. Eigenvalues and Eigenvectors

### The Definition

For a square matrix $A$:
$$Av = \\lambda v$$

$v$ is an **eigenvector**: $A$ only scales it, doesn't rotate it.
$\\lambda$ is the **eigenvalue**: the scaling factor.

### Why They Matter

1. **Matrix powers**: $A^n v = \\lambda^n v$ (easy to compute)
2. **Stability**: $|\\lambda| < 1$ means convergence, $|\\lambda| > 1$ means explosion
3. **PCA**: Principal components are eigenvectors of the covariance matrix

### Key Properties

- $\\text{trace}(A) = \\sum_i \\lambda_i$
- $\\det(A) = \\prod_i \\lambda_i$
- Symmetric matrices have real eigenvalues and orthogonal eigenvectors

---

## 3. Singular Value Decomposition (SVD)

### The Decomposition

ANY matrix $A$ (m × n) can be written as:
$$A = U \\Sigma V^T$$

where:
- $U$ (m × m): Left singular vectors (orthonormal columns)
- $\\Sigma$ (m × n): Diagonal matrix of singular values $\\sigma_1 \\geq \\sigma_2 \\geq ... \\geq 0$
- $V$ (n × n): Right singular vectors (orthonormal columns)

### The Geometric Interpretation

Any linear transformation can be decomposed into:
1. **Rotate** (by $V^T$)
2. **Scale** (by $\\Sigma$)
3. **Rotate** (by $U$)

### Connection to Eigendecomposition

- $V$ = eigenvectors of $A^T A$
- $U$ = eigenvectors of $A A^T$
- $\\sigma_i = \\sqrt{\\lambda_i}$ where $\\lambda_i$ are eigenvalues of $A^T A$

---

## 4. Positive Definite Matrices

### Definition

$A$ is **positive definite** if for all $x \\neq 0$:
$$x^T A x > 0$$

### Intuition

The quadratic form $x^T A x$ is always positive — like a bowl that curves upward in every direction. This guarantees a unique minimum.

### Equivalent Conditions

1. All eigenvalues are positive
2. All pivots (in Gaussian elimination) are positive
3. $A = B^T B$ for some full-rank $B$
4. The matrix is symmetric and all leading principal minors are positive

### Why It Matters in ML

- **Covariance matrices** are positive semidefinite
- **Hessians of convex functions** are positive semidefinite
- **Guarantees** unique solutions to optimization problems

---

## 5. Low-Rank Approximation

### The Eckart-Young Theorem

The best rank-$k$ approximation (minimizing Frobenius norm) is:
$$A_k = \\sum_{i=1}^k \\sigma_i u_i v_i^T$$

Just keep the top $k$ singular values!

### Why This Matters

- **Compression**: Store only $k(m + n + 1)$ numbers instead of $mn$
- **Denoising**: Small singular values often correspond to noise
- **Dimensionality reduction**: This is essentially PCA

---

## 6. The Four Fundamental Subspaces

For matrix $A$ (m × n):

| Subspace | Dimension | Contains |
|----------|-----------|----------|
| Column space $C(A)$ | rank($A$) | Image of $A$ |
| Row space $C(A^T)$ | rank($A$) | "Input directions that matter" |
| Null space $N(A)$ | $n$ - rank($A$) | Vectors mapped to zero |
| Left null space $N(A^T)$ | $m$ - rank($A$) | Orthogonal to column space |

**Key relationship**: Row space ⊥ Null space, Column space ⊥ Left null space

---

## 7. Key Takeaways

1. **Eigenvectors are directions that only get scaled** by the matrix
2. **SVD works for any matrix** — it's the workhorse of numerical linear algebra
3. **Positive definite matrices** guarantee unique minima
4. **Low-rank approximation** is optimal (Eckart-Young)
5. **The four subspaces** give complete geometric understanding
`,

    "Information Theory": `
# Information Theory: The Mathematics of Communication

## 1. What Is Information?

Claude Shannon's insight: **Information is surprise.**

If I tell you "the sun rose today," that's not informative—you expected it.
If I tell you "a meteor hit your house," that's VERY informative—you didn't expect it.

Rare events carry more information than common ones.

---

## 2. Entropy: Average Surprise

### The Definition

$$H(X) = -\\sum_x P(x) \\log P(x) = E[-\\log P(X)]$$

### Why This Formula?

We want information to satisfy:
1. **Rare events have more info**: $I(x) = -\\log P(x)$
2. **Independent events add**: $I(x, y) = I(x) + I(y)$

Entropy is the EXPECTED information—the average surprise.

### Examples

| Distribution | Entropy | Intuition |
|--------------|---------|-----------|
| Certain (P=1) | 0 bits | No surprise |
| Fair coin (0.5, 0.5) | 1 bit | Need 1 yes/no question |
| Unfair coin (0.9, 0.1) | 0.47 bits | Usually heads, less surprising |
| Uniform on 8 items | 3 bits | Need 3 yes/no questions |

---

## 3. KL Divergence: Distance Between Distributions

### Definition

$$D_{KL}(P \\| Q) = \\sum_x P(x) \\log \\frac{P(x)}{Q(x)}$$

### Intuition

How many **extra bits** do you need if you designed a code for distribution $Q$ but the true distribution is $P$?

### Key Properties

1. **Non-negative**: $D_{KL}(P \\| Q) \\geq 0$ (Gibbs' inequality)
2. **Zero iff equal**: $D_{KL}(P \\| Q) = 0 \\Leftrightarrow P = Q$
3. **NOT symmetric**: $D_{KL}(P \\| Q) \\neq D_{KL}(Q \\| P)$

### Why Asymmetry Matters

- $D_{KL}(P \\| Q)$: Penalizes $Q$ being low where $P$ is high (mode-seeking)
- $D_{KL}(Q \\| P)$: Penalizes $Q$ being high where $P$ is low (mode-covering)

In VAEs, we use $D_{KL}(q \\| p)$ to force the encoder to be mode-covering.

---

## 4. Cross-Entropy: The Loss Function

### Definition

$$H(P, Q) = -\\sum_x P(x) \\log Q(x)$$

### Connection to KL Divergence

$$H(P, Q) = H(P) + D_{KL}(P \\| Q)$$

Since $H(P)$ is constant (true distribution doesn't change), **minimizing cross-entropy = minimizing KL divergence**.

### Why Cross-Entropy for Classification?

For one-hot true distribution $P$:
$$H(P, Q) = -\\log Q(\\text{true class})$$

This is exactly the negative log-likelihood of the true class under our model!

---

## 5. Mutual Information: Shared Knowledge

### Definition

$$I(X; Y) = H(X) + H(Y) - H(X, Y) = D_{KL}(P(X,Y) \\| P(X)P(Y))$$

### Intuition

How much does knowing $Y$ tell you about $X$?

- $I(X; Y) = 0$: $X$ and $Y$ are independent (no reduction in uncertainty)
- $I(X; Y) = H(X)$: $Y$ completely determines $X$

### Applications in ML

- **Feature selection**: Pick features with high MI with the target
- **Representation learning**: InfoNCE loss maximizes MI between representations
- **IB principle**: Compress input while preserving MI with output

---

## 6. Key Takeaways

1. **Information = surprise = $-\\log P(x)$**
2. **Entropy = average surprise** — maximum for uniform distribution
3. **KL divergence = extra bits** using wrong code — asymmetric!
4. **Cross-entropy loss** minimizes KL divergence to true distribution
5. **Mutual information** quantifies shared information between variables
`,

    "MLE & MAP": `
# Maximum Likelihood & MAP Estimation

## 1. The Estimation Problem

You observe data $D = \\{x_1, ..., x_n\\}$ from some distribution with unknown parameters $\\theta$.

**The question**: How do you estimate $\\theta$?

---

## 2. Maximum Likelihood Estimation (MLE)

### The Idea

Choose $\\theta$ that makes the observed data **most probable**.

$$\\hat{\\theta}_{MLE} = \\arg\\max_\\theta P(D | \\theta)$$

### Why It Works

Imagine you flip a coin 10 times and get 7 heads. You're choosing between:
- $\\theta = 0.5$ (fair coin): $P(7H | \\theta=0.5) = 0.117$
- $\\theta = 0.7$ (biased): $P(7H | \\theta=0.7) = 0.267$

MLE chooses $\\theta = 0.7$ because it makes your observation more likely.

### The Log-Likelihood Trick

Products are hard to differentiate. Use logs (monotonic, so max is preserved):

$$\\hat{\\theta}_{MLE} = \\arg\\max_\\theta \\sum_{i=1}^n \\log P(x_i | \\theta)$$

### Example: Gaussian MLE

Given $x_1, ..., x_n \\sim \\mathcal{N}(\\mu, \\sigma^2)$:

$$\\log L = -\\frac{n}{2}\\log(2\\pi\\sigma^2) - \\frac{1}{2\\sigma^2}\\sum_{i=1}^n (x_i - \\mu)^2$$

Setting derivatives to zero:
$$\\hat{\\mu}_{MLE} = \\frac{1}{n}\\sum_{i=1}^n x_i = \\bar{x}$$
$$\\hat{\\sigma}^2_{MLE} = \\frac{1}{n}\\sum_{i=1}^n (x_i - \\bar{x})^2$$

**Note**: Variance estimator uses $n$, not $n-1$ — it's biased!

---

## 3. Maximum A Posteriori (MAP)

### The Idea

MLE ignores prior knowledge. What if you have beliefs about $\\theta$ before seeing data?

$$\\hat{\\theta}_{MAP} = \\arg\\max_\\theta P(\\theta | D) = \\arg\\max_\\theta P(D | \\theta) P(\\theta)$$

(We drop $P(D)$ since it doesn't depend on $\\theta$)

### The Log Form

$$\\hat{\\theta}_{MAP} = \\arg\\max_\\theta \\left[ \\underbrace{\\log P(D | \\theta)}_{\\text{log-likelihood}} + \\underbrace{\\log P(\\theta)}_{\\text{log-prior}} \\right]$$

**Key insight**: The log-prior acts as a **regularizer**!

---

## 4. Connection to Regularization

### Gaussian Prior → L2 Regularization

If $P(\\theta) = \\mathcal{N}(0, \\tau^2 I)$:
$$\\log P(\\theta) = -\\frac{1}{2\\tau^2} \\|\\theta\\|^2 + \\text{const}$$

The MAP objective becomes:
$$\\log P(D|\\theta) - \\frac{\\lambda}{2}\\|\\theta\\|^2$$

This is exactly **Ridge regression**!

### Laplace Prior → L1 Regularization

If $P(\\theta) = \\prod_i \\frac{1}{2b}e^{-|\\theta_i|/b}$:
$$\\log P(\\theta) = -\\frac{1}{b}\\|\\theta\\|_1 + \\text{const}$$

This is exactly **Lasso regression**!

---

## 5. MLE vs MAP vs Full Bayesian

| Approach | What it computes | Prior? | Output |
|----------|------------------|--------|--------|
| MLE | $\\arg\\max_\\theta P(D|\\theta)$ | No | Point estimate |
| MAP | $\\arg\\max_\\theta P(\\theta|D)$ | Yes | Point estimate |
| Full Bayesian | $P(\\theta|D)$ | Yes | Full distribution |

### When MAP = MLE

When the prior is **uniform (flat)**, the prior term is constant and drops out. MAP reduces to MLE.

### When to Use Which

- **MLE**: Lots of data, want simplicity
- **MAP**: Want regularization, have prior knowledge
- **Full Bayesian**: Need uncertainty quantification, small data

---

## 6. Key Takeaways

1. **MLE finds parameters that maximize data probability**
2. **MAP adds prior knowledge**, acting as regularization
3. **Gaussian prior = L2 regularization** (Ridge)
4. **Laplace prior = L1 regularization** (Lasso)
5. **Uniform prior makes MAP = MLE**
`,

    "Optimization": `
# Optimization: Finding the Best Parameters

## 1. The Big Picture

Machine learning is optimization. We have:
- A model with parameters $\\theta$
- A loss function $L(\\theta)$ measuring "badness"
- Goal: Find $\\theta^* = \\arg\\min_\\theta L(\\theta)$

---

## 2. Gradient Descent: The Core Algorithm

### The Intuition

You're blindfolded on a hilly landscape. Feel which way is downhill. Take a step. Repeat.

$$\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta L(\\theta_t)$$

The gradient $\\nabla L$ points **uphill**, so we go the opposite direction.

### Why It Works

For small enough $\\eta$, you're guaranteed to decrease the loss (Taylor's theorem):
$$L(\\theta - \\eta \\nabla L) \\approx L(\\theta) - \\eta \\|\\nabla L\\|^2 < L(\\theta)$$

### Variants

| Method | What it uses | Pros | Cons |
|--------|-------------|------|------|
| Batch GD | All data | Stable | Slow, memory-heavy |
| SGD | Single sample | Fast, online | Noisy |
| Mini-batch | Subset (32-256) | Best of both | Industry standard |

---

## 3. Momentum: Physics Intuition

### The Problem

SGD oscillates in narrow valleys. Imagine a ball rolling in a canyon—it bounces side-to-side instead of rolling down.

### The Solution

Add "velocity" that accumulates:
$$v_t = \\gamma v_{t-1} + \\eta \\nabla L$$
$$\\theta_t = \\theta_{t-1} - v_t$$

The ball now **builds up speed** in consistent directions and **dampens oscillations** in inconsistent directions.

### Nesterov Momentum

Look ahead before computing the gradient:
$$v_t = \\gamma v_{t-1} + \\eta \\nabla L(\\theta_{t-1} - \\gamma v_{t-1})$$

Compute gradient at where you're **about to be**, not where you are. Slightly better in practice.

---

## 4. Adam: Adaptive Learning Rates

### The Idea

Different parameters may need different learning rates. Features that appear rarely need larger updates.

Adam maintains:
- **First moment** (mean of gradients): $m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$
- **Second moment** (mean of squared gradients): $v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2$

### Bias Correction

Early estimates are biased toward zero (initialized to 0). Correct:
$$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$

### The Update

$$\\theta_t = \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$$

**Intuition**: Divide by sqrt of squared gradients → parameters with large gradients get smaller updates.

### Default Hyperparameters

- $\\beta_1 = 0.9$: Momentum decay
- $\\beta_2 = 0.999$: RMSprop-like decay
- $\\epsilon = 10^{-8}$: Numerical stability
- $\\eta = 0.001$: Learning rate

---

## 5. Convexity: When Life Is Easy

### Convex Functions

A function is **convex** if:
$$f(\\lambda x + (1-\\lambda) y) \\leq \\lambda f(x) + (1-\\lambda) f(y)$$

Geometrically: The line between any two points lies above the function.

### Why Convexity Matters

- **One minimum**: Every local minimum is global
- **Gradient descent converges**: Guaranteed!
- **Rate**: $O(1/t)$ for SGD, $O(1/t^2)$ with momentum

### The Reality

Neural networks are **non-convex**. Yet SGD still works. Why?

1. Loss landscapes are high-dimensional with many escape routes
2. Most local minima are nearly as good as global minimum
3. Saddle points are more common than bad local minima

---

## 6. Learning Rate Schedules

Learning rate is the most important hyperparameter.

| Schedule | Formula | When to use |
|----------|---------|-------------|
| Constant | $\\eta$ | Baseline, debugging |
| Step decay | $\\eta \\times 0.1^{\\lfloor t/N \\rfloor}$ | Image classification |
| Cosine annealing | $\\eta_{min} + \\frac{\\eta_{max} - \\eta_{min}}{2}(1 + \\cos(\\frac{t\\pi}{T}))$ | Modern LLMs |
| Warmup | Linear increase then decay | Transformers |

### Why Warmup?

Early in training, the model is far from good parameters. Large updates can cause instability. Warmup starts slow, then accelerates.

---

## 7. Key Takeaways

1. **Gradient descent follows the negative gradient** downhill
2. **Momentum helps escape local minima** and navigate ravines
3. **Adam adapts learning rates** per-parameter
4. **Convex functions have one minimum**—but neural nets aren't convex
5. **Learning rate scheduling** is crucial for best performance
`,

    "A/B Testing": `
# A/B Testing & Statistical Inference

## 1. Why A/B Testing?

Machine learning models make **predictions**. A/B tests establish **causation**.

"Users who saw recommendation A bought more" could mean:
- A caused more purchases (causal)
- A was shown to users who already buy more (selection bias)

Randomized experiments eliminate confounding variables.

---

## 2. The Hypothesis Testing Framework

### Setup

- **Null hypothesis ($H_0$)**: Treatment has no effect (A = B)
- **Alternative ($H_1$)**: Treatment has an effect (A ≠ B)

We assume $H_0$ is true and ask: "How surprising is my data?"

### Types of Errors

|  | $H_0$ True (No Effect) | $H_0$ False (Real Effect) |
|--|------------------------|--------------------------|
| Reject $H_0$ | Type I Error (α) | Correct! |
| Don't Reject | Correct | Type II Error (β) |

- **α (significance level)**: Probability of false positive. Typically 0.05.
- **β**: Probability of missing a real effect.
- **Power = 1 - β**: Probability of detecting a real effect. Aim for 0.8+.

---

## 3. The P-Value: Understanding It Correctly

### Definition

$$p = P(\\text{data as extreme or more} | H_0 \\text{ is true})$$

### What It IS

- Probability of seeing this result if there's truly no effect
- Measure of how "surprising" the data is under $H_0$

### What It IS NOT

- ❌ Probability that $H_0$ is true
- ❌ Probability that the result is due to chance
- ❌ Effect size or practical significance

### The Decision Rule

$$p < \\alpha \\Rightarrow \\text{Reject } H_0$$

If $p = 0.03$ and $\\alpha = 0.05$: Reject $H_0$. "Statistically significant."

---

## 4. Sample Size Calculation

### The Formula

$$n = \\frac{2(z_{\\alpha/2} + z_\\beta)^2 \\sigma^2}{\\delta^2}$$

where:
- $z_{\\alpha/2} \\approx 1.96$ for α = 0.05
- $z_\\beta \\approx 0.84$ for 80% power
- $\\sigma^2$ = variance
- $\\delta$ = minimum detectable effect

### The Tradeoffs

- **Larger δ (bigger effect to detect)**: Smaller n needed
- **Higher power**: Larger n needed
- **Lower α**: Larger n needed

### Rule of Thumb

For detecting a 1% relative lift in conversion rate of 5% with 80% power:
$$n \\approx 16 \\times \\frac{(1-p)}{\\delta^2}$$

Often requires **tens of thousands of samples per variant**.

---

## 5. The Multiple Testing Problem

### The Problem

You run 20 A/B tests. With α = 0.05, you expect 1 false positive even if none of the treatments work!

$$P(\\text{at least one false positive}) = 1 - (1 - 0.05)^{20} = 0.64$$

### Solutions

**Bonferroni Correction**: Use α/k for each of k tests.
- Conservative: Reduces true positives too

**False Discovery Rate (FDR)**: Control expected proportion of false positives among rejections.
- Benjamini-Hochberg procedure
- Often better for exploratory analysis

---

## 6. Common Pitfalls

### Peeking

Checking results mid-experiment and stopping when significant → inflated false positive rate.

**Solution**: Pre-register analysis plan. Use sequential testing methods if you need early stopping.

### Simpson's Paradox

Effect reverses when data is disaggregated.

**Example**: Treatment looks better overall, but worse in every subgroup.

**Solution**: Always segment analysis when relevant confounders exist.

### Novelty Effects

New feature looks great initially because it's novel, not because it's better.

**Solution**: Run longer. Wait for effect to stabilize.

---

## 7. Key Takeaways

1. **A/B tests establish causation** where correlation studies cannot
2. **P-value ≠ P(null is true)**—it's P(data | null)
3. **Sample size depends on** effect size, variance, and desired power
4. **Multiple testing inflates** false positives—use corrections
5. **Pre-register analysis** to avoid p-hacking
\`,

    "Advanced Optimization": \`
# Advanced Optimization: Beyond First-Order Methods

## 1. Why Go Beyond Gradient Descent?

**Gradient descent uses first derivatives only**. But functions have more structure!

**Second derivatives tell us**:
- How fast the gradient is changing
- Whether we're in a valley or on a plateau
- The optimal step size (no hyperparameter!)

---

## 2. Newton's Method: The Ideal

### The Idea

Instead of just following the gradient, **use curvature** to jump directly to the minimum.

**Taylor expansion to second order**:
$$L(\\theta + \\Delta) \\approx L(\\theta) + \\nabla L^T \\Delta + \\frac{1}{2} \\Delta^T H \\Delta$$

where $H = \\nabla^2 L$ is the **Hessian** (matrix of second derivatives).

**Setting derivative to zero**:
$$\\nabla L + H \\Delta = 0 \\implies \\Delta = -H^{-1} \\nabla L$$

**Update**:
$$\\theta_{t+1} = \\theta_t - H^{-1} \\nabla L$$

### Why It's Optimal (for Quadratics)

For a quadratic function, Newton's method finds the minimum in **one step**!

For general convex functions, it converges **quadratically**: error squares each iteration.

### Why We Can't Use It

- **Computing Hessian**: $O(n^2)$ storage for $n$ parameters
- **Inverting Hessian**: $O(n^3)$ computation
- For a 1M parameter model: 1 trillion entries, impossible!

---

## 3. Quasi-Newton Methods: Approximating the Hessian

### The Key Insight

We don't need the exact Hessian. We need a **good approximation** that's cheap to compute.

**BFGS** (Broyden-Fletcher-Goldfarb-Shanno) builds up a Hessian approximation using gradient information:

$$B_{k+1} = B_k + \\frac{y_k y_k^T}{y_k^T s_k} - \\frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}$$

where:
- $s_k = \\theta_{k+1} - \\theta_k$ (step taken)
- $y_k = \\nabla L_{k+1} - \\nabla L_k$ (gradient change)

**Why this works**: The update ensures $B_{k+1}$ satisfies the **secant condition**:
$$B_{k+1} s_k = y_k$$

### L-BFGS: Limited Memory

Full BFGS still stores an $n \\times n$ matrix. **L-BFGS** keeps only the last $m$ pairs $(s_k, y_k)$:

- Memory: $O(mn)$ instead of $O(n^2)$
- Typically $m = 10-20$

**Algorithm**: Compute $H^{-1} \\nabla L$ implicitly using two-loop recursion.

### When to Use L-BFGS

| Scenario | Method |
|----------|--------|
| Deep learning (stochastic) | Adam, SGD |
| Small models (full batch) | L-BFGS |
| Convex optimization | L-BFGS |
| sklearn default | L-BFGS |

---

## 4. Constrained Optimization: When There Are Rules

### The Problem

Minimize $f(x)$ subject to constraints:
- $g_i(x) \\leq 0$ (inequality constraints)
- $h_j(x) = 0$ (equality constraints)

**Example**: Minimize cost, subject to: production ≥ demand, capacity ≤ 100.

### The Lagrangian

Combine objective and constraints:
$$\\mathcal{L}(x, \\lambda, \\mu) = f(x) + \\sum_j \\lambda_j h_j(x) + \\sum_i \\mu_i g_i(x)$$

where $\\lambda, \\mu$ are **Lagrange multipliers** (shadow prices).

**Intuition**: Multipliers measure "how much would the objective improve if I relaxed this constraint?"

---

## 5. KKT Conditions: The Optimality Certificate

For a point $x^*$ to be optimal, the **Karush-Kuhn-Tucker conditions** must hold:

**1. Stationarity**:
$$\\nabla f(x^*) + \\sum_j \\lambda_j \\nabla h_j(x^*) + \\sum_i \\mu_i \\nabla g_i(x^*) = 0$$

**2. Primal Feasibility**:
$$h_j(x^*) = 0, \\quad g_i(x^*) \\leq 0$$

**3. Dual Feasibility**:
$$\\mu_i \\geq 0$$

**4. Complementary Slackness**:
$$\\mu_i g_i(x^*) = 0$$

### Why Complementary Slackness?

Either:
- The constraint is **tight** ($g_i(x^*) = 0$) and $\\mu_i > 0$ (constraint matters)
- The constraint is **slack** ($g_i(x^*) < 0$) and $\\mu_i = 0$ (constraint doesn't matter)

**Insight**: Only active constraints affect the solution!

---

## 6. Connection to SVM

The SVM dual problem:
$$\\max_\\alpha \\sum_i \\alpha_i - \\frac{1}{2} \\sum_{i,j} \\alpha_i \\alpha_j y_i y_j (x_i^T x_j)$$

subject to $\\alpha_i \\geq 0$ and $\\sum_i \\alpha_i y_i = 0$.

**KKT conditions reveal**:
- Points with $\\alpha_i > 0$ are **support vectors**
- Other points have $\\alpha_i = 0$ (complementary slackness!)
- This is why SVM solution depends only on support vectors

---

## 7. Duality: A Different Perspective

### The Dual Problem

Instead of minimizing over $x$, we can maximize over multipliers:

$$\\max_{\\lambda, \\mu} \\min_x \\mathcal{L}(x, \\lambda, \\mu)$$

**Weak duality**: Dual optimum ≤ Primal optimum (always)
**Strong duality**: They're equal (under certain conditions)

**Why useful?**
- Sometimes dual is easier to solve
- Provides bounds on optimal value
- Kernel trick exploits this (SVM)

---

## 8. Key Takeaways

1. **Newton's method uses second derivatives** for quadratic convergence
2. **L-BFGS approximates Hessian** cheaply—use for small convex problems
3. **Lagrangian combines objective and constraints** with multipliers
4. **KKT conditions are necessary** for optimality
5. **Complementary slackness**: Only active constraints matter
6. **SVM is a constrained optimization problem**—KKT gives support vectors
`
};


// Add aliases for problem names that should map to main content
// This enables STUDY buttons for individual sub-topics
mathContent["Probability Distributions"] = mathContent["Probability & Statistics"];
mathContent["Bayes Theorem"] = mathContent["Probability & Statistics"];
mathContent["Bayes Theorem (Posterior)"] = mathContent["Probability & Statistics"];
mathContent["Expectation & Variance"] = mathContent["Probability & Statistics"];
mathContent["Eigenvalues & Eigenvectors"] = mathContent["Linear Algebra"];
mathContent["SVD (Singular Value Decomposition)"] = mathContent["Linear Algebra"];
mathContent["Positive Definite Matrices"] = mathContent["Linear Algebra"];
mathContent["Entropy (Shannon)"] = mathContent["Information Theory"];
mathContent["KL Divergence (Relative Entropy)"] = mathContent["Information Theory"];
mathContent["Cross-Entropy Loss Derivation"] = mathContent["Information Theory"];
mathContent["MLE (Maximum Likelihood Estimation)"] = mathContent["MLE & MAP"];
mathContent["Gradient Descent (Taylor Series)"] = mathContent["Optimization"];
mathContent["Convex Optimization"] = mathContent["Optimization"];
mathContent["A/B Testing (Power Analysis)"] = mathContent["A/B Testing"];
mathContent["Lagrange Multipliers"] = mathContent["Matrix Calculus"];

// New L5+ content aliases
mathContent["L-BFGS"] = mathContent["Advanced Optimization"];
mathContent["Newton's Method"] = mathContent["Advanced Optimization"];
mathContent["Quasi-Newton Methods"] = mathContent["Advanced Optimization"];
mathContent["KKT Conditions"] = mathContent["Advanced Optimization"];
mathContent["Constrained Optimization"] = mathContent["Advanced Optimization"];
mathContent["Lagrangian Duality"] = mathContent["Advanced Optimization"];

export default mathContent;

