#02_run_lasso.R
# LASSO REGRESSION MODELS

suppressPackageStartupMessages({
  library(dplyr)
  library(glmnet)
  library(stringr)
  library(pROC)
})

source("../utils/data_loading.R")
source("../utils/evaluation.R")


SPLITS_PATH   <- "splits/cv_folds_k20.rds"
OUTPUT_DIR    <- "results/lasso"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

TRANSFORMATIONS <- c("RA", "PA", "ALR", "CLR", "ILR", "rCLR", "arcsine")
TRANSFORMATIONS <- c("RA")
OUTCOMES        <- c("zscore_binary", "zscore_continuous")
SEED            <- 222
N_STABILITY     <- 30  


run_lasso_stability <- function(transformation, outcome) {
  message(sprintf("LASSO | transformation=%s | outcome=%s", transformation, outcome))
  
  dat <- prepare_model_matrix(
    data_list = load_base_data(),
    outcome = outcome,
    use_compo = transformation
  )
  X <- dat$X
  y <- dat$y
  splits <- readRDS(SPLITS_PATH)
  

  family <- if (grepl("binary", outcome)) "binomial" else "gaussian"
  cat("Model family:", family, "\n")
  
  if (outcome == "zscore_continuous") {
    FORCED_VARS <- c("Age", "Age2", "Sex")
  } else {
    FORCED_VARS <- c("Age", "Age2", "Sex", "zscore_label")
  }
  cat("Forced covariates:", paste(FORCED_VARS, collapse = ", "), "\n")
  
  pf <- rep(1, ncol(X))
  forced_cols <- which(sapply(colnames(X), function(name) {
    any(startsWith(name, FORCED_VARS))
  }))
  pf[forced_cols] <- 0
  
  results <- vector("list", length(splits))
  set.seed(SEED)
  
  for (i in seq_along(splits)) {
    train_idx <- splits[[i]]$train_idx
    test_idx  <- splits[[i]]$test_idx
    
    if (family == "binomial" && length(unique(y[train_idx])) < 2) next
    
    weights <- NULL
    if (family == "binomial") {
      class_counts <- table(y[train_idx])
      weights <- ifelse(
        y[train_idx] == 1,
        1 / class_counts["1"],
        1 / class_counts["0"]
      )
    }
    
    fit <- cv.glmnet(
      x = X[train_idx, ],
      y = y[train_idx],
      alpha = 1,
      family = family,
      weights = weights,
      penalty.factor = pf,
      standardize = TRUE
    )
    
    if (family == "binomial") {
      pred_train <- predict(fit, X[train_idx, ], s = "lambda.min", type = "response")
      pred_test  <- predict(fit, X[test_idx, ],  s = "lambda.min", type = "response")
      
      eval_train <- evaluate_predictions(y[train_idx], pred_train)
      eval_test  <- evaluate_predictions(y[test_idx],  pred_test)
    } else {
      pred_train <- predict(fit, X[train_idx, ], s = "lambda.min")
      pred_test  <- predict(fit, X[test_idx, ],  s = "lambda.min")
      
      eval_train <- evaluate_regression(y[train_idx], as.numeric(pred_train))
      eval_test  <- evaluate_regression(y[test_idx],  as.numeric(pred_test))
    }
    
    beta <- as.matrix(coef(fit, s = "lambda.min"))
    selected <- rownames(beta)[which(beta[, 1] != 0)]
    selected_df <- data.frame(
      variable = selected,
      score_type = "coefficient",
      value = beta[which(beta[, 1] != 0), 1],
      stringsAsFactors = FALSE
    )
    
    stab_list <- replicate(N_STABILITY, {
      boot_idx <- sample(train_idx, length(train_idx), replace = TRUE)
      
      weights_b <- NULL
      if (family == "binomial") {
        class_counts_b <- table(y[boot_idx])
        weights_b <- ifelse(
          y[boot_idx] == 1,
          1 / class_counts_b["1"],
          1 / class_counts_b["0"]
        )
      }
      
      fit_b <- cv.glmnet(
        x = X[boot_idx, ],
        y = y[boot_idx],
        alpha = 1,
        family = family,
        weights = weights_b,
        penalty.factor = pf,
        standardize = TRUE
      )
      beta_b <- as.matrix(coef(fit_b, s = "lambda.min"))
      rownames(beta_b)[which(beta_b[, 1] != 0)]
    })
    
    stab_freq <- sort(table(unlist(stab_list)) / N_STABILITY, decreasing = TRUE)
    stab_df <- data.frame(
      variable = names(stab_freq),
      stability = as.numeric(stab_freq),
      stringsAsFactors = FALSE
    )
    
    results[[i]] <- list(
      train = eval_train,
      test  = eval_test,
      selected = selected_df,
      stability = stab_df
    )
    
    cat(sprintf("Fold %d | \n",
                i))
  }
  
  out_rds <- file.path(OUTPUT_DIR, sprintf("lasso_%s_%s.rds", transformation, outcome))
  saveRDS(results, out_rds)
  cat("Saved:", basename(out_rds), "\n")
  
  return(results)
}

run_lasso_all <- function(transformations = TRANSFORMATIONS, outcomes = OUTCOMES) {
  for (tr in transformations) {
    for (oc in outcomes) {
      try({
        run_lasso_stability(tr, oc)
      }, silent = FALSE)
    }
  }
  message("Training completed")
}

summarize_lasso_results <- function() {
  library(tools)
  
  cat("\n--- Summarizing LASSO results ---\n")
  
  rds_files <- list.files(OUTPUT_DIR, pattern = "^lasso_.*\\.rds$", full.names = TRUE)
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
  
  write.csv(results_summary,  file.path(OUTPUT_DIR, "lasso_results_summary.csv"), row.names = FALSE)
  write.csv(selected_summary, file.path(OUTPUT_DIR, "lasso_selected_variables.csv"), row.names = FALSE)
  
  cat("\nSaved summary files:\n")
  cat(" -", file.path(OUTPUT_DIR, "lasso_results_summary.csv"), "\n")
  cat(" -", file.path(OUTPUT_DIR, "lasso_selected_variables.csv"), "\n")
  
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

if (!interactive()) {
  run_lasso_all()
  summarize_lasso_results()
}