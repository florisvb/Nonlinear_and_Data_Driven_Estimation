# Nonlinear and Data Driven Estimation

A comprehensive course on nonlinear and data-driven estimation techniques taught by Floris van Breugel. This repository contains Jupyter notebooks, Python implementations, and practical examples covering fundamental concepts from batch estimation to advanced data-driven techniques using neural networks and dimensionality reduction.

![course overview](images/nonlinear_estimation_course_overview.png)

## 📚 Course Progression

The course is designed to follow a single example, a planar drone, throughout the sequence of lessons to help build intuition and see how the tools stack together. This example is intended to serve as a template for how other projects can be applied to the sequence of lessons. The course material is designed to build progressively:

**Foundation (Lessons 1-6):** Classical estimation theory, from batch least squares to the Kalman filter

**Observability Analysis (Lessons 7-8):** Understanding what can be estimated from available measurements

**Nonlinear Filtering (Lessons 9-10):** Extended and Unscented Kalman Filters for nonlinear systems

**Data-Driven Methods (Lessons 11-16):** Machine learning approaches including ANNs and SINDy

**Dimensionality Reduction (Lessons 17-18):** Dimensionality reduction and reduced-order estimation for ANNs

## 📚 Getting Started

New to the course? Check out our [**Tips on Getting Started with GitHub, Google Colab, and Docker**](github_colab_tips.md) to set up your development environment.

---

## 📖 Course Contents

### Lesson 1: Dynamics Demo
**Topics:**
- Planar drone dynamics modeling and simulation
- Control affine system representation
- Animation and visualization of dynamical systems
- Introduction to PyBounds for constraint handling

**Key Notebooks:** `A_planar_drone_dynamics.ipynb`, `B_planar_drone_animation.ipynb`

---

### Lesson 2: Batch Least Squares
**Topics:**
- Parameter estimation fundamentals
- Linear least squares (batch processing)
- Statistical foundations of least squares estimation
- Error analysis and uncertainty quantification

**Key Notebooks:** `A_Example_Parameter_Estimation.ipynb`, `B_Linear_Least_Squares.ipynb`, `C_Linear_Least_Squares_Statistics.ipynb`

---

### Lesson 3: Sequential Least Squares
**Topics:**
- Linear sequential estimation algorithms
- Random variables and covariance analysis
- Recursive parameter estimation
- Transition from batch to sequential processing

**Key Notebooks:** `A_Linear_Sequential.ipynb`, `B_Random_Variables_Covariance.ipynb`

---

### Lesson 4: Minimum Variance Estimation and Cramér-Rao Bound
**Topics:**
- Minimum variance estimation theory
- Cramér-Rao Lower Bound (CRLB)
- Performance bounds for estimators
- Optimal estimator design

**Key Notebooks:** `A_Minimum_Variance_Estimation.ipynb`, `B_Cramer_Rao_Bound.ipynb`

---

### Lesson 5: Discrete Linear Kalman Filter
**Topics:**
- Linearizing continuous-time nonlinear models
- Discretization techniques for dynamic systems
- Discrete-time Kalman filter derivation and implementation
- Understanding Kalman filter limitations

**Key Notebooks:** `A_linearizing_and_discretizing_dynamics.ipynb`, `B_discrete_linear_kalman_filter.ipynb`, `C_breaking_the_linear_kalman_filter.ipynb`

---

### Lesson 6: Linear Observability
**Topics:**
- Observability fundamentals for linear systems
- Observability matrix analysis
- Observability Gramian
- Determining state observability from measurements

**Key Notebooks:** `A_Linear_Observability.ipynb`

---

### Lesson 7: Analytical Nonlinear Observability
**Topics:**
- Symbolic computation for nonlinear observability
- Nonlinear observability analysis
- Monocular camera example case study
- Analytical observability conditions

**Key Notebooks:** `A_symbolic_nonlinear_observability.ipynb`

---

### Lesson 8: Empirical Nonlinear Observability
**Topics:**
- Empirical observability matrix computation
- Data-driven observability analysis
- PyBounds framework for empirical analysis
- Custom simulator integration with PyBounds

**Key Notebooks:** `A_empirical_nonlinear_observability.ipynb`, `B_empirical_nonlinear_observability_pybounds.ipynb`, `C_pybounds_with_custom_simulator_tutorial.ipynb`

---

### Lesson 9: Extended Kalman Filter
**Topics:**
- Extended Kalman Filter (EKF) theory
- Linearization of nonlinear dynamics
- EKF implementation for planar drone
- Comparison with linear Kalman filter

**Key Notebooks:** `A_planar_drone_EKF.ipynb`

---

### Lesson 10: Unscented Kalman Filter
**Topics:**
- Unscented Kalman Filter (UKF) algorithm
- Sigma point generation and propagation
- Comparison of EKF vs. UKF performance
- Monocular camera state estimation

**Key Notebooks:** `A_planar_drone_EKF_UKF.ipynb`, `A_planar_drone_EKF_UKF_monocamera.ipynb`

---

### Lesson 11: Generating Training Data
**Topics:**
- Synthetic data generation for machine learning
- Training dataset creation from simulations
- Data visualization and validation
- Preparing data for neural network training

**Key Notebooks:** `A_generate_training_data_demo.ipynb`, `B_visualize_training_data.ipynb`

---

### Lesson 12: Artificial Neural Network Estimators
**Topics:**
- Neural network-based state estimation
- ANN architecture design for estimation problems
- Training and evaluating ANN estimators
- Application to planar drone altitude estimation

**Key Notebooks:** `A_planar_drone_altitude_ANN.ipynb`, `B_Evaluate_ANN_Estimator.ipynb`

---

### Lesson 13: AI Kalman Filter
**Topics:**
- Hybrid AI-Kalman filtering approaches
- Integrating neural networks with UKF
- Data-driven process and measurement models
- Enhanced state estimation with learned models

**Key Notebooks:** `A_planar_drone_AI_UKF.ipynb`

---

### Lesson 14: Naive Numerical Differentiation
**Topics:**
- Numerical differentiation from noisy data
- PyNumDiff library and methods
- Smoothing and filtering techniques
- Application to velocity estimation from position data

**Key Notebooks:** `A_planar_drone_pynumdiff.ipynb`

---

### Lesson 15: SINDy (Sparse Identification of Nonlinear Dynamics)
**Topics:**
- Discovering governing equations from data
- PySINDy library for sparse regression
- Learning dynamics models from observations
- Learning measurement models
- Integration with UKF for state estimation

**Key Notebooks:** `A_planar_drone_pysindy_dynamics_model.ipynb`, `B_planar_drone_pysindy_measurement_model.ipynb`, `C_planar_drone_pysindy_UKF.ipynb`

---

### Lesson 16: Artificial Neural Network Models
**Topics:**
- Neural network-based dynamics models
- Neural network-based measurement models
- Data-driven UKF with learned models
- End-to-end learning for state estimation

**Key Notebooks:** `A_keras_dynamics_model.ipynb`, `B_keras_measurement_model.ipynb`, `C_data_driven_UKF.ipynb`

---

### Lesson 17: Dimensionality Reduction with SVD
**Topics:**
- Singular Value Decomposition (SVD) theory
- Terrain and optic flow data analysis
- SVD-based compression techniques
- Reduced-order representations of high-dimensional data

**Key Notebooks:** `A_terrain_and_optic_flow_for_planar_drone.ipynb`, `B_SVD_compression_of_optic_flow.ipynb`

---

### Lesson 18: Reduced-Order ANN Estimator
**Topics:**
- Combining dimensionality reduction with neural networks
- Reduced-order modeling (ROM) for state estimation
- ANN training on compressed representations
- Performance evaluation of ROM-ANN estimators

**Key Notebooks:** `A_planar_drone_ray_distance_ANN_with_ROM.ipynb`, `B_evaluate_ROM_ANN.ipynb`

---

## 🎯 Learning Objectives

By completing this course, you will:
- Master classical estimation theory from batch to sequential methods
- Understand and implement Kalman filtering for linear and nonlinear systems
- Analyze observability for both linear and nonlinear systems
- Apply modern data-driven techniques including neural networks and SINDy
- Combine classical filtering with machine learning for hybrid estimators
- Perform dimensionality reduction for high-dimensional estimation problems
- Develop practical skills in state estimation for autonomous systems

## 🛠️ Tools & Technologies

- **Python 3.x**
- **Jupyter Notebooks**
- **NumPy / SciPy** - Numerical computing
- **Matplotlib** - Visualization
- **SymPy** - Symbolic mathematics
- **TensorFlow / Keras** - Neural networks
- **PySINDy** - Sparse identification of nonlinear dynamics
- **PyNumDiff** - Numerical differentiation
- **PyBounds** - Empirical observability analysis

## 📂 Repository Structure

Each lesson directory contains:
- Jupyter notebooks with theory, examples, and exercises
- Solution notebooks (marked with `_SOLUTION` or `_SOLUTIONS`)
- PDF lecture notes and derivations
- README files with additional context

## 🚀 Running the Code

### Option 1: Local Installation
```bash
git clone https://github.com/florisvb/Nonlinear_and_Data_Driven_Estimation.git
cd Nonlinear_and_Data_Driven_Estimation
# Install dependencies -- see `requirements_minimal.txt` and `Utility/Requirements`
jupyter notebook
```

### Option 2: Google Colab
All notebooks can be run directly in Google Colab. See [github_colab_tips.md](github_colab_tips.md) for instructions.

## 👨‍🏫 About

This course material is developed and maintained by [Floris van Breugel](https://github.com/florisvb), Associate Professor in Mechanical Engineering at the University of Nevada, Reno.

## 📄 License

This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**In brief:** You are free to use and adapt these materials for non-commercial educational purposes with attribution. Commercial use requires permission. See [LICENSE](LICENSE) for full details.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit pull requests.

---

*This course bridges classical control and estimation theory with modern data-driven machine learning approaches, providing students with both theoretical foundations and practical implementation skills.*
