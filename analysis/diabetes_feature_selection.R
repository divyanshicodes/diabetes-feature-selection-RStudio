# Diabetes Feature Selection: Linear Regression, PCR, and LASSO
# Author: Divyanshi Mishra

# install.packages("glmnet")  # run once if needed
# install.packages("pls")     # run once if needed

library(glmnet)
library(pls)

# 1. Load data
myData <- read.table("diabetes_interaction.txt",
                     header = TRUE)

# 2. Inspect structure and response
str(myData)
head(myData)
colnames(myData)
summary(myData$response)

# 3. Train / test split (300 train, rest test)
set.seed(0)
myRatio  <- 300
trainIdx <- sample(nrow(myData), myRatio)

x_train <- myData[trainIdx, ]
x_test  <- myData[-trainIdx, ]

# -------------------------------------------------
# Linear Regression with Significant Features
# -------------------------------------------------

# 4. Full linear model with all features
lm_full <- lm(response ~ ., data = x_train)
summary(lm_full)  # inspect p-values

# According to the summary, example significant features include:
# sex, bmi, map, age.sex (plus intercept) – adjust as needed.

# 5. Reduced linear model using significant features only
lm_sig <- lm(response ~ sex + bmi + map + age.sex,
             data = x_train)

# 6. Predict on test set and compute MSE
lm_pred <- predict(lm_sig, x_test)
lm_mse  <- mean((lm_pred - x_test$response)^2)
lm_mse

# -------------------------------------------------
# Principal Component Regression (PCR)
# -------------------------------------------------

pcr_model <- pcr(response ~ .,
                 data = x_train,
                 validation = "CV")

# 7. Validation plot: R² vs number of components
validationplot(pcr_model, val.type = "R2")

# Suppose R² levels off around 15 components (adjust if you see different)
pcr_opt <- pcr(response ~ .,
               data = x_train,
               validation = "CV",
               ncomp = 15)

pcr_pred <- predict(pcr_opt, x_test, ncomp = 15)
pcr_mse  <- mean((pcr_pred - x_test$response)^2)
pcr_mse

# -------------------------------------------------
# LASSO Regression
# -------------------------------------------------

# Convert to matrix form: last column = response
mx_train <- as.matrix(x_train)
mx_test  <- as.matrix(x_test)

# 8. Lambda grid
lambdaVals <- seq(0.001, 10, length.out = 100)

# 9. Cross-validated LASSO (10-fold)
lasso_cv <- cv.glmnet(mx_train[, -65],
                      mx_train[, 65],
                      alpha = 1,
                      lambda = lambdaVals,
                      nfolds = 10)

# 10. Inspect lambda sequence and best lambda
lasso_cv$lambda
best_lambda <- lasso_cv$lambda.min
best_lambda

# 11. Final LASSO model with best lambda
lasso_model <- glmnet(mx_train[, -65],
                      mx_train[, 65],
                      alpha = 1,
                      lambda = best_lambda,
                      family = "gaussian")

lasso_pred <- predict(lasso_model, mx_test[, -65])
lasso_mse  <- mean((lasso_pred - mx_test[, 65])^2)
lasso_mse
