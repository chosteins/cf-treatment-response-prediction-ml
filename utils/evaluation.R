#evaluation.R

# MODEL EVALUATION UTILS

library(pROC)
library(PRROC)

# ------------------------------------------------------------------------------
# Evaluate binary classification predictions
# ------------------------------------------------------------------------------
#' Evaluate binary classification model performance
#' 
#' @param y_true True binary labels (0/1 or factor)
#' @param y_prob Predicted probabilities for positive class
#' @return List containing roc_auc, pr_auc, and input vectors
evaluate_predictions <- function(y_true, y_prob) {
  if (length(unique(y_true)) < 2) {
    warning("Only one class present in y_true")
    return(list(roc_auc = NA, pr_auc = NA, y_true = y_true, y_prob = y_prob))
  }
  
  roc_auc <- as.numeric(auc(roc(y_true, y_prob, quiet = TRUE)))
  pr <- pr.curve(
    scores.class0 = y_prob[y_true == 1],
    scores.class1 = y_prob[y_true == 0], 
    curve = FALSE
  )
  
  return(list(
    roc_auc = roc_auc, 
    pr_auc = as.numeric(pr$auc.integral),
    y_true = y_true, 
    y_prob = y_prob
  ))
}

# ------------------------------------------------------------------------------
# Evaluate regression predictions
# ------------------------------------------------------------------------------
#' Evaluate regression model performance
#' 
#' @param y_true True continuous outcome values
#' @param y_pred Predicted continuous values
#' @return List containing RMSE, MAE, R-squared, correlation, and input vectors
evaluate_regression <- function(y_true, y_pred) {
  rmse <- sqrt(mean((y_true - y_pred)^2))

  mae <- mean(abs(y_true - y_pred))
  
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  
  correlation <- cor(y_true, y_pred, method = "pearson")
  
  return(list(
    rmse = rmse,
    mae = mae,
    r_squared = r_squared,
    correlation = correlation,
    y_true = y_true,
    y_pred = y_pred
  ))
}