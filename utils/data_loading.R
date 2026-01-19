#data_loading.R

# DATA LOADING AND PREPROCESSING UTILITIES

library(dplyr)
library(caret)

# ------------------------------------------------------------------------------
# Load base microbiome data
# ------------------------------------------------------------------------------
#' Load simulated patient data with M0 (baseline) and M12 (follow-up)
#' 
#' @param data_path Path to the CSV file containing patient data
#' @return List with two data frames: m0 (baseline) and m12 (12-month follow-up)
load_base_data <- function(data_path = "../data/simulated_microbiome.csv") {
  data <- read.csv(data_path, sep = ",")
  data_m0 <- data[data$Time == "M0", ]
  data_m12 <- data[data$Time == "M12", ]
  return(list(m0 = data_m0, m12 = data_m12))
}

# ------------------------------------------------------------------------------
# Generate piecewise transformations (for spline modeling)
# ------------------------------------------------------------------------------
#' Transform compositional variables into bins (degree-0 splines)
#' 
#' @param df Data frame containing compositional variables
#' @param var_patterns Regex pattern to identify compositional columns (default: "^b_")
#' @param probs Quantile probabilities for binning (default: 6 bins)
#' @return Matrix of binary indicators for each bin
generate_piecewise <- function(df, var_patterns = "^b_", probs = seq(0, 1, length.out = 6)) {
  compo_cols <- grep(var_patterns, names(df), value = TRUE)
  piecewise_dfs <- list()

  for (var in compo_cols) {
    vals <- df[[var]]
    qs <- quantile(vals, probs = probs, na.rm = TRUE)
    qs_unique <- unique(qs)

    if (length(qs_unique) >= 2) {
      cut_var <- cut(vals, breaks = qs_unique, include.lowest = TRUE)
      
      if (nlevels(cut_var) >= 2) {
        mat <- model.matrix(~ cut_var - 1)
        colnames(mat) <- paste0(var, "_bin", seq_len(ncol(mat)))
        piecewise_dfs[[var]] <- as.data.frame(mat)
      } else {
        message("Skipped (only 1 level after binning): ", var)
      }
    } else {
      message("Skipped (not enough distinct quantiles): ", var)
    }
  }

  if (length(piecewise_dfs) == 0) {
    stop("No b_ variables could be transformed into bins (degree-0 splines).")
  }

  out_mat <- do.call(cbind, piecewise_dfs)
  return(out_mat)
}

# ------------------------------------------------------------------------------
# Prepare model design matrix
# ------------------------------------------------------------------------------
#' Prepare design matrix (X) and outcome vector (y) for modeling
#' 
#' @param data_list List containing m0 and m12 data frames
#' @param outcome Outcome type: "zscore_binary", "delta_binary", "zscore_continuous", "delta_continuous"
#' @param use_compo Compositional transformation: "raw", "RA", "PA", "ALR", "CLR", "ILR", "rCLR", "arcsine", "spline0"
#' @param transformed_dir Directory containing pre-computed transformations
#' @return List with X (design matrix), y (outcome vector)
prepare_model_matrix <- function(data_list, outcome = NULL, use_compo = NULL,
                                 transformed_dir = "../data/transformed") {
  data_m0 <- data_list$m0
  data_m12 <- data_list$m12

  # Impute missing Pielou values with 0
  data_m0$Pielou_16S[is.na(data_m0$Pielou_16S)] <- 0

  # Convert categorical variables to factors
  data_m0$Sex <- as.factor(data_m0$Sex)
  data_m0$zscore_label <- as.factor(data_m0$zscore_label)

  # Set patient ID as row names
  rownames(data_m0) <- data_m0$id_patient
  rownames(data_m12) <- data_m12$id_patient

  # Create Age squared term (for quadratic effects)
  data_m0$Age2 <- data_m0$Age^2


  features <- c("Age", "Age2", "Sex", "zscore_label", 
                "Chao1_16S", "Simpson_16S", "Shannon_16S", "Pielou_16S")

  if (use_compo == "raw") {
    compo <- dplyr::select(data_m0, starts_with("b_"))
  } else if (use_compo == "spline0") {
    compo <- generate_piecewise(data_m0)
  } else {
    # Load pre-computed transformation
    compo_path <- file.path(transformed_dir, paste0(use_compo, "_transformed.csv"))
    if (!file.exists(compo_path)) {
      stop("Transformation file not found: ", compo_path)
    }
    compo <- read.csv(compo_path, sep = ",")
    compo <- filter(compo, Time == "M0")
    rownames(compo) <- compo$id_patient
    compo <- compo[rownames(data_m0), ]
    compo <- dplyr::select(compo, starts_with("b_"))
  }


  X <- cbind(data_m0[, features], compo)


  dummy_model <- dummyVars(" ~ .", data = X, fullRank = TRUE, drop.levels = FALSE)
  X <- predict(dummy_model, newdata = X)


  X <- as.matrix(X)
  storage.mode(X) <- "numeric"


  if (outcome == "zscore_binary") {
    y <- data_m12[rownames(data_m0), "zscore_label"]
  } else if (outcome == "delta_binary") {
    y <- data_m12[rownames(data_m0), "group"]
  } else if (outcome == "zscore_continuous") {
    y <- data_m12[rownames(data_m0), "bmi_zscore"]
  } else if (outcome == "delta_continuous") {
    y <- data_m12[rownames(data_m0), "delta"]
  } else {
    stop("Unrecognized outcome: ", outcome)
  }

  return(list(X = X, y = y))
}