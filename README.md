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

