#01_run_logistic.R

# LOGISTIC REGRESSION (BASELINE)

suppressPackageStartupMessages({
  library(dplyr)
  library(pROC)
})

source("../utils/data_loading.R")
source("../utils/evaluation.R")

SPLITS_PATH <- "splits/cv_folds_k20.rds"
OUTPUT_DIR  <- "results/logistic"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

OUTCOMES <- c("zscore_binary", "zscore_continuous")
SEED     <- 222

run_logistic_baseline <- function(outcome) {
  message(sprintf("Logistic Baseline | outcome=%s", outcome))
  
  dat <- prepare_model_matrix(
    data_list = load_base_data(),
    outcome = outcome, use_compo = "raw"
  )
  
  X_full <- dat$X
  y <- dat$y
  splits <- readRDS(SPLITS_PATH)
  
  family <- if (grepl("binary", outcome)) "binomial" else "gaussian"
  cat("Model family:", family, "\n")
  
  # keep ONLY clinical covariates (exclude microbiome b_* features)
  keep_cols <- which(!grepl("^b_", colnames(X_full)))
  X <- X_full[, keep_cols, drop = FALSE]
  
  cat("Columns kept (no microbiome):", ncol(X), "\n")
  

  results <- vector("list", length(splits))
  set.seed(SEED)
  

  for (i in seq_along(splits)) {
    train_idx <- splits[[i]]$train_idx
    test_idx  <- splits[[i]]$test_idx
    

    weights <- NULL
    if (family == "binomial") {
      if (length(unique(y[train_idx])) < 2) next
      
      class_counts <- table(y[train_idx])
      weights <- ifelse(
        y[train_idx] == 1,
        1 / class_counts["1"],
        1 / class_counts["0"]
      )
    }
    

    train_df <- as.data.frame(X[train_idx, ])
    train_df$y <- y[train_idx]
    test_df <- as.data.frame(X[test_idx, ])
    

    if (family == "binomial") {
      model <- glm(y ~ ., data = train_df, family = binomial, weights = weights)
      
      pred_train <- predict(model, newdata = train_df, type = "response")
      pred_test  <- predict(model, newdata = test_df, type = "response")
      
      eval_train <- evaluate_predictions(y[train_idx], pred_train)
      eval_test  <- evaluate_predictions(y[test_idx], pred_test)
    } else {
      model <- glm(y ~ ., data = train_df, family = gaussian)
      
      pred_train <- predict(model, newdata = train_df, type = "response")
      pred_test  <- predict(model, newdata = test_df, type = "response")
      
      eval_train <- evaluate_regression(y[train_idx], pred_train)
      eval_test  <- evaluate_regression(y[test_idx], pred_test)
    }
    
    results[[i]] <- list(
      train = eval_train,
      test = eval_test
    )
    
    cat(sprintf("Fold %d complete\n", i))
  }
  

  out_rds <- file.path(OUTPUT_DIR, sprintf("logistic_%s.rds", outcome))
  saveRDS(results, out_rds)
  cat("Saved:", basename(out_rds), "\n")
  
  return(results)
}

run_logistic_all <- function(outcomes = OUTCOMES) {
  for (oc in outcomes) {
    try({
      run_logistic_baseline(oc)
    }, silent = FALSE)
  }
  message("Logistic baseline training complete.")
}

# Summarize results
summarize_logistic_results <- function() {
  library(tools)
  
  cat("\n--- Summarizing baseline results ---\n")
  
  rds_files <- list.files(OUTPUT_DIR, pattern = "^logistic_.*\\.rds$", full.names = TRUE)
  summary_list <- list()
  
  convert_auc <- function(x) if (inherits(x, "auc")) as.numeric(x) else x
  
  for (file in rds_files) {
    res <- readRDS(file)
    filename <- basename(file_path_sans_ext(file))
    parts <- unlist(strsplit(filename, "_"))
    
    method <- "logistic"
    outcome <- paste(parts[3:length(parts)], collapse = "_")
    family <- if (grepl("binary", outcome)) "binomial" else "gaussian"
    
    for (i in seq_along(res)) {
      fold_res <- res[[i]]
      if (is.null(fold_res$test) || is.null(fold_res$train)) next
      
      fold_res$test  <- lapply(fold_res$test,  convert_auc)
      fold_res$train <- lapply(fold_res$train, convert_auc)
      
      if (family == "binomial") {
        row_perf <- data.frame(
          method = method,
          transformation = "baseline",
          outcome = outcome,
          fold = i,
          type = "binary",
          roc_auc_test  = as.numeric(fold_res$test$roc_auc %||% NA),
          pr_auc_test   = as.numeric(fold_res$test$pr_auc %||% NA),
          roc_auc_train = as.numeric(fold_res$train$roc_auc %||% NA),
          pr_auc_train  = as.numeric(fold_res$train$pr_auc %||% NA)
        )
      } else {
        row_perf <- data.frame(
          method = method,
          transformation = "baseline",
          outcome = outcome,
          fold = i,
          type = "continuous",
          rmse_test = as.numeric(fold_res$test$rmse %||% NA),
          mae_test  = as.numeric(fold_res$test$mae %||% NA),
          r2_test   = as.numeric(fold_res$test$r_squared %||% NA),
          rmse_train = as.numeric(fold_res$train$rmse %||% NA),
          mae_train  = as.numeric(fold_res$train$mae %||% NA),
          r2_train   = as.numeric(fold_res$train$r_squared %||% NA)
        )
      }
      summary_list[[length(summary_list) + 1]] <- row_perf
    }
  }
  
  results_summary <- bind_rows(summary_list)
  
  write.csv(results_summary, file.path(OUTPUT_DIR, "logistic_results_summary.csv"), row.names = FALSE)
  
  cat("\nSaved:", file.path(OUTPUT_DIR, "logistic_results_summary.csv"), "\n")
  

  if (nrow(results_summary) > 0) {
    summary_binary <- results_summary %>% filter(type == "binary")
    summary_continuous <- results_summary %>% filter(type == "continuous")
    
    if (nrow(summary_binary) > 0) {
      auc_summary <- summary_binary %>%
        group_by(method, outcome) %>%
        summarise(
          mean_auc = mean(roc_auc_test, na.rm = TRUE),
          sd_auc   = sd(roc_auc_test, na.rm = TRUE),
          n        = n(),
          .groups = "drop"
        ) %>%
        mutate(mean_sd = sprintf("%.3f +/- %.3f", mean_auc, sd_auc))
      
      cat("\n--- Binary outcomes (ROC-AUC) ---\n")
      print(auc_summary, n = Inf)
    }
    
    if (nrow(summary_continuous) > 0) {
      rmse_summary <- summary_continuous %>%
        group_by(method, outcome) %>%
        summarise(
          mean_rmse = mean(rmse_test, na.rm = TRUE),
          sd_rmse   = sd(rmse_test, na.rm = TRUE),
          mean_r2   = mean(r2_test, na.rm = TRUE),
          sd_r2     = sd(r2_test, na.rm = TRUE),
          n         = n(),
          .groups = "drop"
        ) %>%
        mutate(
          rmse_sd = sprintf("%.3f +/- %.3f", mean_rmse, sd_rmse),
          r2_sd   = sprintf("%.3f +/- %.3f", mean_r2, sd_r2)
        )
      
      cat("\n--- Continuous outcome (RMSE) ---\n")
      print(rmse_summary, n = Inf)
    }
  }
  
  #return(results_summary)
}

if (!interactive()) {
  run_logistic_all()
  summarize_logistic_results()
}