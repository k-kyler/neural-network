# Neural Network from scratch

## Mathematical Foundation of Neural Networks (MLP)

### Forward Propagation

1. **Input Layer**: Input features $X = [x_1, x_2, ..., x_n]$

2. **Hidden Layer Computation**:

   - For each neuron $j$ in a hidden layer:
     - Weighted sum: $z_j^{[l]} = \sum_{i=1}^{n} w_{ji}^{[l]} \cdot a_i^{[l-1]} + b_j^{[l]}$
     - Activation: $a^{[l]}_j = g^{[l]}(z^{[l]}_j)$
   - Where:
     - $w^{[l]}_{ji}$ is the weight from the $i$-th neuron in layer $l-1$ to the $j$-th neuron in layer $l$
     - $b^{[l]}_j$ is the bias for the $j$-th neuron in layer $l$
     - $g^{[l]}$ is the activation function (e.g., ReLU, sigmoid, tanh)

3. **Output Layer**:

   - Final output: $\hat{y} = g^{[L]}(z^{[L]})$
   - For classification, often softmax:

     $$\hat{y}_i = \frac{\exp(z_i^{[L]})}{\sum_{j=1}^{k} \exp(z_j^{[L]})}$$

### Backward Propagation (Learning)

1. **Loss Function**: $J(W, b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)})$

   - For regression: Mean Squared Error (MSE)
   - For classification: Cross-Entropy Loss

2. **Gradient Computation**:

   - Output layer error: $\delta^{[L]} = \nabla_a J \odot g^{[L]'}(z^{[L]})$
   - Hidden layer error: $\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot g^{[l]'}(z^{[l]})$
   - Weight gradients: $\nabla_{W^{[l]}} J = \delta^{[l]} (a^{[l-1]})^T$
   - Bias gradients: $\nabla_{b^{[l]}} J = \delta^{[l]}$

3. **Parameter Update** (Gradient Descent):
   - $W^{[l]} := W^{[l]} - \alpha \nabla_{W^{[l]}} J$
   - $b^{[l]} := b^{[l]} - \alpha \nabla_{b^{[l]}} J$
   - Where $\alpha$ is the learning rate

## Obesity Dataset Information

The dataset used in this project is the [Obesity Dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) which contains various attributes related to eating habits and physical condition of individuals. It was created to estimate obesity levels based on eating habits and physical condition.

**Features**:

- Demographics: Gender, Age, Height, Weight
- Family history: family_history_with_overweight
- Eating habits:
  - FAVC: Frequent consumption of high caloric food
  - FCVC: Frequency of consumption of vegetables
  - NCP: Number of main meals
  - CAEC: Consumption of food between meals
  - CALC: Consumption of alcohol
- Physical condition and activities:
  - SCC: Calories consumption monitoring
  - FAF: Physical activity frequency
  - TUE: Time using technology devices
  - SMOKE: Smoking habits
- Transportation: MTRANS (Mode of transportation)

**Target Variable**: NObeyesdad - Obesity level (Normal_Weight, Overweight_Level_I, Overweight_Level_II, etc.)
