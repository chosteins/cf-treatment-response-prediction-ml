#03_run_rf.R

# RANDOM FOREST


suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(ranger)
  library(pROC)
})

source("../utils/data_loading.R")
source("../utils/evaluation.R")


SPLITS_PATH <- "splits/cv_folds_k20.rds"
OUTPUT_DIR  <- "results/rf"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

TRANSFORMATIONS <- c("RA", "PA", "ALR", "CLR", "ILR", "rCLR", "arcsine")
OUTCOMES        <- c("zscore_binary", "zscore_continuous")

NTREES <- 500
SEED   <- 222

# Covariates to force in all models
COVAR_TARGETS <- c("Age", "Age2", "Sex", "SexM", "zscore_label")

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

#' Detect forced covariates in design matrix
#' @param X Design matrix
#' @param covar_targets List of covariate prefixes to force
#' @param outcome_name Outcome name to avoid data leakage
#' @return Character vector of column names to always split on
detect_forced_covars <- function(X, covar_targets, outcome_name = NULL) {
  if (length(covar_targets) == 0) return(character(0))
  covar_regex <- paste0("^(", paste(covar_targets, collapse = "|"), ")(\\b|_|$)")
  vars <- grep(covar_regex, colnames(X), value = TRUE)
  
  # Prevent data leakage: never force outcome-related variables
  leak_like <- c(outcome_name, "delta_label")
  vars <- setdiff(vars, leak_like)
  
  unique(vars)
}

#' Create mtry grid for hyperparameter tuning
#' @param p Number of features
#' @return Vector of mtry values to try
make_mtry_grid <- function(p) {
  grid <- unique(pmax(1, c(
    round(sqrt(p) / 2),
    round(sqrt(p)),
    round(sqrt(p) * 1.5)
  )))
  sort(unique(grid))
}

# ------------------------------------------------------------------------------
# Main Random Forest function
# ------------------------------------------------------------------------------
run_rf_model <- function(transformation, outcome) {
  message(sprintf("Random Forest | transformation=%s | outcome=%s", transformation, outcome))

  # Load data and splits
  dat <- prepare_model_matrix(
    data_list = load_base_data(),
    outcome = outcome,
    use_compo = transformation
  )
  X <- dat$X
  y <- dat$y
  splits <- readRDS(SPLITS_PATH)
  
  # Determine model family
  family <- if (grepl("binary", outcome)) "binomial" else "gaussian"
  cat("Model family:", family, "\n")
  
  # Compute class weights (for binary outcomes)
  class_weights <- NULL
  if (family == "binomial") {
    class_counts <- table(y)
    class_weights <- setNames(1 / class_counts, names(class_counts))
  }
  
  # Detect forced covariates
  always_keep <- detect_forced_covars(X, COVAR_TARGETS, outcome_name = outcome)
  if (length(always_keep) == 0) {
    warning("No forced covariates detected in X.")
  } else {
    cat("Forced covariates (", length(always_keep), "):", 
        paste(always_keep, collapse = ", "), "\n", sep = "")
  }
  
  # Tune mtry via cross-validation
  p <- ncol(X)
  mtry_grid <- make_mtry_grid(p)
  cv_mtry_results <- data.frame()
  set.seed(SEED)
  
  cat("\nTuning mtry...\n")
  for (mtry_val in mtry_grid) {
    fold_scores <- numeric(length(splits))
    
    for (i in seq_along(splits)) {
      train_idx <- splits[[i]]$train_idx
      test_idx  <- splits[[i]]$test_idx
      test_y <- y[test_idx]
      
      if (family == "binomial") {
        model <- ranger(
          formula = as.factor(y[train_idx]) ~ .,
          data = data.frame(X[train_idx, , drop = FALSE]),
          num.trees = NTREES,
          mtry = mtry_val,
          importance = "none",
          class.weights = class_weights,
          always.split.variables = always_keep,
          probability = TRUE,
          seed = SEED
        )
        pred <- predict(model, data.frame(X[test_idx, , drop = FALSE]))$predictions[, "1"]
        
        fold_scores[i] <- if (length(unique(test_y)) == 2) {
          as.numeric(auc(roc(test_y, pred, quiet = TRUE)))
        } else NA_real_
      } else {
        model <- ranger(
          formula = y[train_idx] ~ .,
          data = data.frame(X[train_idx, , drop = FALSE]),
          num.trees = NTREES,
          mtry = mtry_val,
          importance = "none",
          always.split.variables = always_keep,
          seed = SEED
        )
        pred <- predict(model, data.frame(X[test_idx, , drop = FALSE]))$predictions
        fold_scores[i] <- -sqrt(mean((test_y - pred)^2))  # Negative RMSE (maximize)
      }
    }
    
    cv_mtry_results <- rbind(
      cv_mtry_results,
      data.frame(mtry = mtry_val, mean_score = mean(fold_scores, na.rm = TRUE))
    )
  }
  
  best_mtry <- cv_mtry_results$mtry[which.max(cv_mtry_results$mean_score)]
  cat("Best mtry for", transformation, "(", outcome, "):", best_mtry, "\n")
  
  # Train final models on each split with permutation importance
  results <- vector("list", length(splits))
  
  for (i in seq_along(splits)) {
    train_idx <- splits[[i]]$train_idx
    test_idx  <- splits[[i]]$test_idx
    
    if (family == "binomial") {
      train_data <- data.frame(X[train_idx, , drop = FALSE], y = factor(y[train_idx]))
      model <- ranger(
        formula = y ~ .,
        data = train_data,
        num.trees = NTREES,
        mtry = best_mtry,
        importance = "permutation",
        class.weights = class_weights,
        always.split.variables = always_keep,
        probability = TRUE,
        seed = SEED
      )
      
      pred_train <- predict(model, data.frame(X[train_idx, , drop = FALSE]))$predictions[, "1"]
      pred_test  <- predict(model, data.frame(X[test_idx, , drop = FALSE]))$predictions[, "1"]
      
      eval_train <- evaluate_predictions(y[train_idx], pred_train)
      eval_test  <- evaluate_predictions(y[test_idx], pred_test)
    } else {
      train_data <- data.frame(X[train_idx, , drop = FALSE], y = y[train_idx])
      model <- ranger(
        formula = y ~ .,
        data = train_data,
        num.trees = NTREES,
        mtry = best_mtry,
        importance = "permutation",
        always.split.variables = always_keep,
        seed = SEED
      )
      
      pred_train <- predict(model, data.frame(X[train_idx, , drop = FALSE]))$predictions
      pred_test  <- predict(model, data.frame(X[test_idx, , drop = FALSE]))$predictions
      
      eval_train <- evaluate_regression(y[train_idx], pred_train)
      eval_test  <- evaluate_regression(y[test_idx], pred_test)
    }
    
    # Stability selection via bootstrap
    N_STAB <- 30
    TOP_K  <- 20
    
    stab_importances <- replicate(N_STAB, {
      boot_idx <- sample(train_idx, length(train_idx), replace = TRUE)
      
      if (family == "binomial") {
        boot_data <- data.frame(X[boot_idx, , drop = FALSE], y = factor(y[boot_idx]))
        model_boot <- ranger(
          formula = y ~ .,
          data = boot_data,
          num.trees = NTREES,
          mtry = best_mtry,
          importance = "permutation",
          class.weights = class_weights,
          always.split.variables = always_keep,
          probability = TRUE,
          seed = sample.int(1e6, 1)
        )
      } else {
        boot_data <- data.frame(X[boot_idx, , drop = FALSE], y = y[boot_idx])
        model_boot <- ranger(
          formula = y ~ .,
          data = boot_data,
          num.trees = NTREES,
          mtry = best_mtry,
          importance = "permutation",
          always.split.variables = always_keep,
          seed = sample.int(1e6, 1)
        )
      }
      
      imp_boot <- sort(model_boot$variable.importance, decreasing = TRUE)
      names(imp_boot)[seq_len(min(TOP_K, length(imp_boot)))]
    })
    
    stab_freq <- sort(table(stab_importances) / N_STAB, decreasing = TRUE)
    stab_df <- data.frame(
      variable = names(stab_freq),
      stability = as.numeric(stab_freq),
      stringsAsFactors = FALSE
    )
    
    stab_df <- stab_df %>%
      mutate(stability_scaled = stability / max(stability))
    
    # Normalize importances (0-1)
    imp <- sort(model$variable.importance, decreasing = TRUE)
    imp_norm <- imp / max(imp, na.rm = TRUE)
    selected_df <- data.frame(
      variable = names(imp),
      score_type = "importance",
      value = as.numeric(imp_norm),
      stringsAsFactors = FALSE
    )
    
    results[[i]] <- list(
      train = eval_train,
      test = eval_test,
      selected = selected_df,
      stability = stab_df
    )
  }
  
  # Save results
  out_rds <- file.path(OUTPUT_DIR, sprintf("rf_%s_%s.rds", transformation, outcome))
  saveRDS(results, out_rds)
  cat("Saved:", basename(out_rds), "\n")
}

# ------------------------------------------------------------------------------
# Run all RF configurations
# ------------------------------------------------------------------------------
run_rf_all <- function(transformations = TRANSFORMATIONS, outcomes = OUTCOMES) {
  for (tr in transformations) {
    for (oc in outcomes) {
      try({
        run_rf_model(tr, oc)
      }, silent = FALSE)
    }
  }
  message("Random Forest training complete.")
}

# ------------------------------------------------------------------------------
# Summarize RF results
# ------------------------------------------------------------------------------
summarize_rf_results <- function() {
  library(tools)
  
  cat("\n--- Summarizing Random Forest results ---\n")
  
  rds_files <- list.files(OUTPUT_DIR, pattern = "^rf_.*\\.rds$", full.names = TRUE)
  summary_list <- list()
  selected_list <- list()
  
  convert_auc <- function(x) if (inherits(x, "auc")) as.numeric(x) else x
  
  for (file in rds_files) {
    res <- readRDS(file)
    filename <- basename(file_path_sans_ext(file))
    parts <- unlist(strsplit(filename, "_"))
    
    method <- parts[1]
    transformation <- parts[2]
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
          transformation = transformation,
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
          transformation = transformation,
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
      
      # Selected variables
      if (!is.null(fold_res$selected)) {
        sel <- fold_res$selected %>%
          mutate(
            method = method,
            transformation = transformation,
            outcome = outcome,
            fold = i
          )
        selected_list[[length(selected_list) + 1]] <- sel
      }
      
      # Stability scores
      if (!is.null(fold_res$stability)) {
        stab <- fold_res$stability %>%
          mutate(
            method = method,
            transformation = transformation,
            outcome = outcome,
            fold = i,
            score_type = "stability",
            value = stability
          ) %>%
          select(method, transformation, outcome, fold, variable, score_type, value)
        selected_list[[length(selected_list) + 1]] <- stab
      }
    }
  }
  
  results_summary  <- bind_rows(summary_list)
  selected_summary <- bind_rows(selected_list)
  
  write.csv(results_summary,  file.path(OUTPUT_DIR, "rf_results_summary.csv"), row.names = FALSE)
  write.csv(selected_summary, file.path(OUTPUT_DIR, "rf_selected_variables.csv"), row.names = FALSE)
  
  cat("\nSaved summary files:\n")
  cat(" -", file.path(OUTPUT_DIR, "rf_results_summary.csv"), "\n")
  cat(" -", file.path(OUTPUT_DIR, "rf_selected_variables.csv"), "\n")
  
  # Display mean +/- SD by transformation and outcome
  if (nrow(results_summary) > 0) {
    if ("roc_auc_test" %in% names(results_summary)) {
      auc_summary <- results_summary %>%
        group_by(transformation, outcome) %>%
        summarise(
          mean_auc = mean(roc_auc_test, na.rm = TRUE),
          sd_auc   = sd(roc_auc_test, na.rm = TRUE),
          n        = n(),
          .groups = "drop"
        ) %>%
        arrange(outcome, desc(mean_auc)) %>%
        mutate(auc_mean_sd = sprintf("%.3f +/- %.3f", mean_auc, sd_auc))
      
      cat("\n--- Mean AUC +/- SD by transformation ---\n")
      print(auc_summary, n = Inf)
    }
  }
}

# ------------------------------------------------------------------------------
# Execute
# ------------------------------------------------------------------------------
if (!interactive()) {
  run_rf_all()
  summarize_rf_results()
}
