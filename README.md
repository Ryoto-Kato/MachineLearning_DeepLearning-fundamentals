# Deep Learning and Machine Learning Summary
## Examples
- Deep Learning (Python, Pytorch, Tensorflow)
    - Basics
    - Autoencoder
    - GAN
- Machine Learning (Python, Pytorch, Scikit-learn)
    - Dimensionality Reduction Techniques
    - Clustering
    - Support Vector Machine

# Theoretical Summary
## Tasks
- Check the definition of Bias and Variance 
## Machine Learning
- Bias (How far from the ground truth)
    - __High-Bias__: All predictions of samples are off from ground truth
    - __Low-Bias__: All predictions are close to ground truth
- Variance (How variate the prediction from ground truth)
    - __High-Variance__: All prediction variate
    - __Low-Variance__: All prediction do not variate

### Techniques
-  K-fold cross validation
    - Use all your training data and the best hyperparameters for final training of your model.
    - Process
        - Devide training dataset into k-subgroups
        - Assign one group as validation data and rest of them will be training data

- Leave one out cross validation
    - This is techniques where we do not habe enough data set
    - One data sample is assigned to validation data and rest of them  will be used for the training 

### K-nearest neighbor (Low bias and High variance)
- KNN (more robust compared to 1NN)
    - Profile
        - Expensive memory and native inference
        - Can not apply to huge dataset (Data structure)
            - Use tree-based search structure (kd-tree)
        - We should have small feature attirbutes but enough samples

where k is the number of samples we consider as its neighbor
$$P(y=c | \mathbf{x}, k) = \frac{1}{K}\sum_{j\in N_k(\mathbf{x})} I(y_j = c)$$

$$\hat{y_i} = argmax_c P(y_i=c| \mathbf{x_i}, k)$$

- Weighted KNN (More robust compared to KNN)
    - Sample which is far away from x, it will be accouonted less.
    - Sample which is close to x, it will be accounted more.

$$P(y=c|\mathbf{x}, k) = \frac{1}{z}\sum_{j\in N_k(x)} \frac{1}{d(\mathbf{x}, \mathbf{x_j})} y_j$$

$$z = \frac{1}{\sum_{j\in N_k(x)} d(\mathbf{x}, \mathbf{x_j})}$$

- KNN regression
$$\hat{y} = \frac{1}{z} \sum_{j \in N_k(\mathbf{x})} \frac{1}{d(\mathbf{x}, \mathbf{x_j})} y_j$$

- Measure the performance
    - Confusion table
        |               | y_true = 1 | y_true = 0 |
        | ------------- |:----------:|:---------:|
        | y_pred = 1    | True positive | False positive |
        | y_pred = 0    | True negative | False negative |
    - Accuracy 
        $$\frac{TP+TN}{TP + FP + TN + FN}$$
    - __Recall__
        $$\frac{TP}{TP + TN}$$
    - __Precision__
        $$\frac{TP}{TP + FP}$$
    - F1 score
        $$\frac{2(Recall * Precision)}{Recall + Precision}$$
    - Ex: Where numerous candidates are there, we want to give a candidate to the suitable position
        - Since we have numerous candidates, we want to assign hire a candidate efficiently. To do so, we need to avoid to have many samples from False Positive.
        - In this case, it is better to prioritise Precision rather than recall 
- Distance measurement
    $$L_{\inf \text{Norm}} = max_i|u_i - v_i|$$
    $$\text{Mahalarobis} = \sqrt{(u-v)^T\Sigma^{-1}(u-v)}$$

- Standarization: scale each feature to zero mean and unit variance (This should be done before applying KNN)
$$x_{i, std} = \frac{x_i - \mu_i}{\sigma_i}$$

- As features dimensionality gets higher, we need exponentially increased amount of sample to get sufficient sample density for explanation of the data and neighbors will be far away

### Decision tree
- DT is much smaller complexity with respect to memory and storage for inference
- More flexible decision function
- Overfitting with DT
    - Where __imparity__ $i(t)$ is eual to zero
- Regularization
    - Post prunning
        
        0. We have DT $T$
        1. Use validation data set to get erro estimate err(T)
        2. Obtain error $err(T-T_t)$
            - $T-T_t$ is prunned DT which prunes $T_t$ from T
        3. Improvement of imparity
            - $\Delta i = err_(T) - err(T-T_t)$
        4. Repeat until err($T$) < err($T-T_t$)
    
    - Bagging (Boostrap aggregating)
        - Random forest
        - Multiple DT tranined on different dataset (mini-batch) using only randomly selected set of features
        - The number of features
            - $log_{2}d$ for regression
            - $\sqrt{d}$ for classification

Label prediction is based on the majority in the partition
$$P(y = c| \mathbf{x}, \mathbf{R}) = \frac{\eta_{c, R}}{\sum_{c' \in c} \eta_{c', R}}$$

Prediction will be
$$\hat{y} = argmax_c P(y=c|\mathbf{x}, \mathbf{R})$$

- Naive decision tree construction consts expensive
- Building the optimal decision tree is interactable using __greedy heuristic__
    - How much spilit $t$ can improve the imparity at children imparity.

- Imparity measurement ($\pi_c = P(y = c, \mathbf{x}, t)$)
    1. Misclassification rate
    $$i_{E(t)} = 1 - \text{max}\pi_c$$
    2. Entropy
    $$i_t(t) = - \sum_{c_i \in c} \pi_{ci} log_2 \pi_{ci}$$
    3. Gini Index
    $$i_g(t) = 1 - \sum_{c_i \in c} \pi_{ci}^2$$

- Imparity improvement

where $P_L$ and $P_R$ are probability how much samples are assigned into the partition from the parent nodes

$$\Delta i(s, t) = i(t) - (P_L*i(t_L)+P_R*i(t_R))$$

### Probabilistic inference (Toy example)
- Probabilistic distribution
    - Based on the Baysian rule
$$P(\theta | D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

- Bernorili distribution
    - where $y_i = {0, 1}$
$$P(y_i = T| \theta) = Ber(y_i | \theta) = \theta^{y_i}(1-\theta)^{1-yi}$$

- Beta distribution (Prior-distribution of Bernorilli distribution setting
    - This is conjugate setting with Bernorilli distribution setting
$$Beta(\theta| \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha-1}(1-\theta)^{\beta-1}$$

- __Beta__ and __Benoulli__ are conjugate setting
    - The posterior distribution $P(\theta | D)$ is modeled by beta distribution
$$P(\theta |D) =Beta(\theta|\alpha + |T|, \beta + |H|) \quad \\
= \prod_{i = 1}^N Ber(x_i | \theta) Beta (\theta| \alpha, \beta)$$

- Mode of Beta distribution
$$\frac{\alpha -1 }{\alpha + \beta - 2}$$

- Toy example (coin flip)
$$\theta_{MLE} =\frac{|T|}{|T|+|H|}$$
$$\theta_{MAP} =\frac{|T|+\alpha-1}{|T|+|H|+\alpha + \beta -2}$$
where $\alpha=\beta=1$, the prior distribution is not informative at all and $\theta_{MLE} = \theta_{MAP}$

- Hoeffding's inequality

How many samples (experiments) do we need to preserve <1% error
$$P(|\theta_{MLE} - \theta| \leqq \epsilon) \leqq 2\exp(-2N\epsilon^2)\leqq\delta$$
    
where N = |T|+|H|
For example, if $\delta = 0.01$ (error between MLE and $\theta$ should be close enough
$$N \geqq \frac{ln(2/\delta)}{2\epsilon^2} => N\geqq256$$

- Prediction of next flip
MLE (Point estimation)
$$P(F = T|\theta_{MLE}) = Ber(F=T | \frac{|T|}{|T|+|H|})$$

MAP (point estimation with prior knowledge)
$$P(F =T |\theta_{MAP}) = Ber(F=T | \frac{|T| + \alpha -1 }{|T| + |H| + \alpha + \beta -2})$$

Fully Baysian (estimation of uncertainty)
$$P(F=T|D) = Ber(F=T|\frac{|T|+\alpha}{|T|+|H|+\alpha+\beta})$$

using joint and marginalization, __posterior predictive distribution__
$$P(f|D,\alpha,\beta)=\int P(f|\theta)P(\theta|D, \alpha, \beta) d\theta$$

This posterior predictive distribution can be interpreted as weighted mean of probability of next flip, in which weighting coefficinet posterior distribution

- MLE: Maximum likelihood, Small dataset leads overfitting since it is only based on data, equivalend with mode of likelihood
- MAP: Maximum a prior: If likelihood is dominant (the number of samples is large enough), this will be close to MLE, otherwise close to prior distribution
- Fully Baysian: Uncertainty of estimation
    - as we observe a lot of data, the fully bayesian will close to MLE, otherwise close to prior

### Linear regression
- Find mapping $f(\cdot)$ from inputs to targets.
- Target y is generated by a deterministic unction $f(x)$ plus noise
$$y_i = f(x_i) + \epsilon_i = \mathcal{N}(f_w(x_i), \beta^{-1}) = \sqrt{\frac{\beta}{2\pi}}\exp(-\frac{\beta}{2}(x_i - f_w(x_i))^2)$$
$$\epsilon_i \sim \mathcal{N}(f_w(x_i), \beta^{-1})$$

- Normal equation for least square problem (we can solve with closed form solution)

where $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is invertible, $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ will be equal to $\mathbf{X}^{-1}$
$$w^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}$$

### Nonlinear regression
- Find funcion f_w($\cdot$) which weighted average of multiple polynomials $\mathbf{\Phi}$
- The degree of polynomial is declared as $M$

$$y_i = f_w(x_i) + \epsilon_i = \mathbf{W}^T\mathbf{\Phi(x_i)} + \epsilon_i = \sum_{j = 1}^M w_j\phi_j(x_i) + \epsilon_i$$

- We can model linear regression model by non-linear regression model
    - Where $\phi_j(x_i) = x_i$ (identity function) and $\phi_0(x_i) = 1$, $\mathbf{w}^T\Phi(\mathbf{x_i}) = \mathbf{w}^T\mathbf{x_i}$

- Final form of the non-linear regression
    - The closed-form solution (same with LS normal equation)
        - This could be describe as the Kernel tric
$$w^* = (\Phi^T\Phi)^{-1}\Phi^{T}y$$

Where __design matrix__

$$\Phi = [\mathbf{\Phi(\mathbf{x_0}), \Phi(\mathbf{x_1}), \Phi(\mathbf{x_2}}), ..., \Phi(\mathbf{x_N})]^T \in \real^{N \times M}$$

### Deterministic (linaear, non-linear) model
- Non-linear regression, Linear regression are deterministic model which can infer an outcome given a input by using trained (modeled) mapping funcion from input to output
- Drawbacks
    - It can not describe __How certain the prediction is__
    -> Probabilistic model
- Least sequare error
$$E_{LS}(\mathbf{w}) = \frac{1}{2}\sum_{i=i}^N[\mathbf{w}^T\mathbf{x_i} - y_i]^2$$
- __Ridge error__
    - Introducing regurization term to prevent model from overfitting
    - Overfitting model -> large weights
$$E_{Ridge}(\mathbf{w}) = \frac{1}{2}\sum_{i=i}^N[\mathbf{w}^T\mathbf{x_i} - y_i]^2 + \frac{1}{2}\mathbf{w}^T\mathbf{w}$$

- Reguralization of linear regression
    - Penalize the large weights
        - Approach 
            - __Cross validation (LOOCV)__ to find out the appropriate degree of polynomial
            - Provide __achievable finite optimum__ for log minimization task with __differentiable function__ if the error function is based on BCE(Binary cross entropy) or Hinge Loss

### Probabilistic linear regression
Where we assume that we can model data distribution by __Gauss__ distribution including __Noise__
$$y_i = \mathcal{N}(f_w(x_i), \beta^{-1}) = f_w(x_i) + \beta^{-1} = \mathbf{w}^T\phi(x_i) + \epsilon_i = \sum_{j=1}^Mw_j\phi_j(x_i) + \epsilon_i$$
#### Maximum Likelihood Estimation (MLE) = LS
- point estimation of parameters such that they can most likely yield our observation

$$P(D|\mathbf{w}) = \prod_{i=1}^N P(x_i | \mathbf{w}, \beta^{-1}) = \prod_{i=1}^{N}\mathcal{N}(f_w(x_i), \beta^{-1}) \text{(i,i,d)}$$
- __i, i, d__
    - independent distribution
        - The likelihood of given pbservation can be obtained by multiplication of each likelihood
    - identical
        - The circumstance will not be changed by any chance

- Negative log likelihood
    - To estimate maximum likelihood, it could be really large number because of the number of experiments
    - To make it finite, we will map them to logarithmic function and find parameters such that they can minimize log-likelihood instead of maximization tasks

    - The closed form solution of $\mathbf{w}_{MLE}$ and $\mathbf{\beta}_{MLE}$
        - $\mathbf{w}_{MLE}$ can be derived by the normal equations 
        - $\mathbf{\beta}_{MLE}$ can be derived by avarage of squared error
$$\mathbf{w}_{MLE} = (\Phi^T\Phi)^{-1}\Phi^T\mathbf{y}$$
$$\beta_{MLE}^{-1} = \frac{1}{N}\sum_{i=1}^N(w_{MLE}^T\mathbf{\phi(x_i)} -y_i)^2$$

### Maximum a priori estimation (MAP) = Ridge
- Point estimation of parameters such that they can most likely yield correct label given input.
- prior distibution
    - __isotropic multivariate normal distribution__
    - We strongly assume that weights $\mathbf{w}$ can be sampled from the zero-centred Gauss distribution
    - This could be also interpreted as the "__Reguralizer__" since we assume that weights are distributed around zero most likely
        - It can prevent weights to be really large
$$P(w|\alpha) = \mathcal{N}(\mathbf{w}|0, \alpha^{-1}I) = (\frac{\alpha}{2\pi})^{\frac{M}{2}}\exp(-\frac{\alpha}{2}\mathbf{w}^T\mathbf{w})$$

### Fully Baysian
- Model entire posterior distribution 
- The baysian rule is equivalent with following
$$P(\mathbf{w}|D) \propto P(D|\mathbf{w})P(\mathbf{w}|0, \beta^-1)$$

Where we assume that likelihood and prior distribution are normal distribution, from conjugate setting, we can model this "posteori" distribution with Normal distribution too.

$$P(\mathbf{w}|D) \propto \mathcal{N}(\mathbf{y}|f_w(\mathbf{x}), \beta^{-1}) \cdot \mathcal{N}(w|0, \alpha^{-1}I)\propto \mathcal{N}(\mathbf{w}|\mu,\Sigma)$$

$$\mu = \beta\Sigma\Phi^T\mathbf{y}, \Sigma = \alpha I+ \beta\Phi^T\Phi$$
\phi(x_i)^T\Sigma(\phi(x_i)))
where $\alpha = 0$, prior distribution is not informative at all -> $\mathbf{w}_{MAP} = \mathbf{w}_{MLE}$

where N=0, $\mathcal{N}(\mathbf{w}|\mu, \Sigma) = \mathcal{N}(\mathbf{w}|0, \alpha^{-1}I)$

### Posterior predictive distribution
- We can estimate next observation by using fully baysian
- We inject helper $\mathbf{w}$ to perform marginalization to get __data-dependent uncertainty of estimation__
    - Around our obervation, we have prediction with high-confidence since $\beta^{-1} + \phi(x_i)^T\Sigma(\phi(x_i)))$ yields small variance, otherwise large variance. 

$$P(y_i|x_i, D, \alpha) = \int p(y_i|x_i, \mathbf{w})p(\mathbf{w}|D) d\mathbf{w} = \mathcal{N}(y_i | \mathbf{\mu}^Tx_i, \beta^{-1}+\phi(x_i)^T\Sigma(\phi(x_i)))$$



