# Diabetes Feature Selection in R

This project applies three linear modeling approaches to a diabetes dataset with interaction terms: standard linear regression, principal component regression (PCR), and LASSO regression. The goal is to predict a quantitative diabetes progression response while exploring which features are most important.

## Problem Statement

We are given measurements for 442 diabetes patients, including 10 main features and multiple interaction terms, along with a quantitative response that measures disease progression after one year. The objective is to build and compare three models:

- A linear regression model using selected significant features  
- A principal component regression model using an optimal number of components  
- A LASSO regression model with an optimally tuned λ  

For each model, we train on a subset of patients and evaluate mean squared error (MSE) on a held-out test set. :contentReference[oaicite:0]{index=0}

## Steps Performed

1. Load the `diabetes_interaction.txt` dataset into `myData`.   
2. Inspect structure, head, column names, and the response summary.  
3. Split the data into training and testing sets by randomly selecting 300 patients for training and using the remaining observations for testing (with `set.seed(0)`).  
4. Fit a full linear regression model using all features on the training set.  
5. Identify significant predictors (p-value < 0.05) from the full model and refit a reduced linear model using only those features.  
6. Predict on the test set with the reduced model and compute test MSE. :contentReference[oaicite:2]{index=2}  
7. Fit a principal component regression (PCR) model with cross-validation and inspect the validation plot of R² versus number of components.  
8. Choose an appropriate number of components (where R² levels off), refit PCR with that number, and compute test MSE.   
9. Create a sequence of 100 λ values between 0.001 and 10 and run 10-fold cross-validated LASSO using `cv.glmnet`.  
10. Extract the λ with minimum cross-validation error, refit the LASSO model using this λ, predict on the test set, and compute test MSE.   

## Models

- Linear regression with selected significant features  
- Principal component regression (PCR)  
- LASSO regression with cross-validated λ  

## Tools

- R / RStudio  
- `glmnet` (LASSO)  
- `pls` (PCR)
