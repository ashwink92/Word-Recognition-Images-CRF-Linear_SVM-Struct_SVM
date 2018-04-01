# Word-Recognition-Images-CRF-Linear_SVM-Struct_SVM
Created a module for implementing
Conditional random fields(CRF) using L-BFGS solver to recognize words and alphabets from images using NumPy in Python
  1. Formulating the Objective Function for the solver to minimize 
     min -C/n \sum log(p(y | x)) + 1/2 \sum ||w<sup>2</sup><sub>y</sub>|| + 1/2 \sum T<sub>ij</sub><sup>2</sup>
Improving benchmarks set with similar implementations of
1. linear SVM
2. Struct SVM
