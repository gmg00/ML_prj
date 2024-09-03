# Neural Network from Scratch

## Overview

This repository contains a Python implementation of a neural network from scratch. The project was developed as part of a machine learning course and focuses on applying and evaluating custom-built models on classification and regression tasks using the MONK and ML-CUP datasets.

## Key Contributions

### Implementation Details

- **Custom Neural Network Architecture**: 
  - The network is built using a custom `Layer` class for handling forward and backward propagation, and a `NeuralNetwork` class to manage the overall network architecture. This structure supports various configurations of hyperparameters and activation functions.
  
- **Hyperparameter Tuning**: 
  - Extensive hyperparameter tuning was performed using grid search combined with 5-fold cross-validation. The tuning process was guided by Mean Squared Error (MSE) for MONK and Mean Euclidean Error (MEE) for ML-CUP.
  
- **Optimization Techniques**:
  - **Nesterov Momentum**: Implemented to enhance convergence speed and stability, offering improvements over standard momentum.
  - **Adam Optimizer**: Added to further accelerate convergence and stabilize training. Though not used in the final model, it provided valuable insights during experimentation.
  - **Regularization**: Both L1 (Lasso) and L2 (Ridge) regularization techniques were implemented. L1 regularization was tested, although it did not yield significant improvements over L2.

- **Preprocessing**:
  - **Normalization and Standardization**: These techniques were tested to assess their impact on model performance. While normalization helped reduce overfitting, standardization did not show a significant advantage in our experiments.

## Results

### MONK Dataset

- **MONK-1**: 
  - **Configuration**: 4 sigmoidal units, online training, \(\eta = 0.05\), \(\alpha = 0.6\), \(\lambda = 0\).
  - **Performance**: 
    - **Training MSE**: 4.24e-04
    - **Test MSE**: 6.47e-04
    - **Accuracy**: 100% on both training and test sets.

- **MONK-2**: 
  - **Configuration**: 4 sigmoidal units, online training, \(\eta = 0.05\), \(\alpha = 0.6\), \(\lambda = 0\).
  - **Performance**: 
    - **Training MSE**: 4.72e-05
    - **Test MSE**: 5.32e-05
    - **Accuracy**: 100% on both training and test sets.

- **MONK-3**: 
  - **Configuration**: 4 sigmoidal units, batch training, \(\eta = 0.3313\), \(\alpha = 0.5\), \(\lambda = 0\).
  - **Performance**: 
    - **Training MSE**: 3.99e-02
    - **Test MSE**: 3.01e-02
    - **Accuracy**: 94.26% (training), 96.53% (test).

### ML-CUP Dataset

- **Best Model Configuration**:
  - **Architecture**: 2 hidden layers, each with 70 tanh units.
  - **Hyperparameters**: \(\eta = 0.001\), \(\lambda = 3e-05\), \(\alpha = 0.9\), batch size = 150, Nesterov momentum = True.
  
- **Performance**:
  - **Training MEE**: 0.683
  - **Test MEE**: 0.838

## Conclusion

This project demonstrates the construction and tuning of a neural network from scratch, with particular attention to optimization techniques and regularization strategies. The models built show strong performance on both MONK and ML-CUP datasets, validating the effectiveness of the implemented methodologies. Despite some challenges with overfitting, the final models offer a solid foundation for further exploration and refinement.

## Authors

- Gian Marco Gori
- Francesco Luigi Moretti
- Silvia Sonnoli

For any inquiries, please contact us at:
- g.gori21@studenti.unipi.it
- f.moretti14@studenti.unipi.it
- s.sonnoli@studenti.unipi.it
