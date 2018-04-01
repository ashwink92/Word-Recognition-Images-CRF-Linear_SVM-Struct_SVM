# Word-Recognition-Images-CRF-Linear_SVM-Struct_SVM
Created a module for implementing
Conditional random fields(CRF) using L-BFGS solver to recognize words and alphabets from images using NumPy in Python
  1. Formulating the Objective Function for the solver to minimize
     \n min -C/n \sum log(p(y | x)) + 1/2 \sum ||w<sup>2</sup><sub>y</sub>|| + 1/2 \sum T<sub>ij</sub><sup>2</sup>
        Where w is node weights(pixel weight) for each letter in the alphabet
        and T is the edge weight between each letter pair in the word
  2. Formulating the Gradient functions for both the edge weights and node weights(T and w respectively)
  3. Providing the objective function and gradient function to the BFGS solver to find the suitable t and w vectors that minimize the      objective function
Improving benchmarks set with similar implementations of
1. linear SVM
2. Struct SVM
