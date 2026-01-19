#00_transformation_preprocessing.R

# Generate all compositional transformations from raw microbiome data
# outputs are saved in ../data/transformed/ directory

library(dplyr)
library(tidyr)
library(compositions)
library(zCompositions)
library(matrixStats)


INPUT_FILE <- "../data/simulated_microbiome.csv"
OUTPUT_DIR <- "../data/transformed"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

df <- read.csv(INPUT_FILE, header = TRUE, sep = ",")
abund <- df %>% dplyr::select(starts_with("b_")) %>% as.data.frame()
meta  <- df %>% dplyr::select(-starts_with("b_"))

save_df <- function(new_df, name) {
  output_path <- file.path(OUTPUT_DIR, paste0(name, "_transformed.csv"))
  write.csv(new_df, output_path, row.names = FALSE)
  cat("Saved:", output_path, "\n")
}


# 1) Relative Abundances (RA)
row_sum <- rowSums(abund)
row_sum[row_sum == 0] <- 1
rel_abund <- abund / row_sum
save_df(cbind(meta, rel_abund), "RA")

# 2) Presence-Absence (PA)
presence_absence <- as.data.frame((abund > 0) * 1L)
save_df(cbind(meta, presence_absence), "PA")

# 3) Arcsine Square Root (on relative abundances)
arcsin_sqrt <- as.data.frame(asin(sqrt(rel_abund)))
save_df(cbind(meta, arcsin_sqrt), "arcsine")


abund_mat <- as.matrix(abund)
abund_nz <- zCompositions::cmultRepl(
  abund_mat,
  label = 0,
  method = "CZM",
  z.warning = 0,
  z.delete = FALSE
)

# 4) Additive Log-Ratio (ALR) with reference taxon
alr_ref <- "b_Granulicatella"
if (!(alr_ref %in% colnames(abund_nz))) {
  stop("ALR reference taxon '", alr_ref, "' not found")
}
denom_idx <- which(colnames(abund_nz) == alr_ref)
alr_trans <- compositions::alr(abund_nz, denom = denom_idx) %>% as.data.frame()
save_df(cbind(meta, alr_trans), "ALR")
writeLines(paste0("ALR reference taxon: ", alr_ref), 
           file.path(OUTPUT_DIR, "ALR_reference.txt"))

# 5) Isometric Log-Ratio (ILR)
stopifnot(identical(colnames(abund_nz), colnames(abund)))
ilr_mat <- compositions::ilr(abund_nz)
ilr_trans <- as.data.frame(ilr_mat)

# Name ILR coordinates by original taxa names
p <- ncol(abund_nz)
stopifnot(ncol(ilr_trans) == p - 1)
colnames(ilr_trans) <- colnames(abund_nz)[1:(p-1)]

save_df(cbind(meta, ilr_trans), "ILR")
cat("[ILR] ncol(abund) =", p, "-> ncol(ILR) =", ncol(ilr_trans), "\n")

# 6) Centered Log-Ratio (CLR)
clr_trans <- compositions::clr(abund_nz) %>% as.data.frame()
save_df(cbind(meta, clr_trans), "CLR")


# 7) Robust CLR (rCLR) - median-based
rclr_transform <- function(X, pseudocount = 0.5) {
  X <- as.matrix(X)
  X[X <= 0] <- pseudocount
  L <- log(X)
  med <- matrixStats::rowMedians(L, na.rm = TRUE)
  sweep(L, 1, med, FUN = "-")
}

rclr_trans <- rclr_transform(abund, pseudocount = 0.5) %>% as.data.frame()
save_df(cbind(meta, rclr_trans), "rCLR")
