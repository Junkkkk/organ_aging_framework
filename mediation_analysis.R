# Mediation atlas across organ aging models and diseases
# Verifies that the gap-disease regression coefficient equals the mediation b-path
# (Fig. 4C in Park et al.)

library(data.table)

# ---- Inputs ----
# Set DATA_DIR to the directory containing organ age estimates from Oh et al. (2025)
# and UK Biobank phenotype files. Files are not included in this repository.
DATA_DIR <- "data"

age_file <- file.path(DATA_DIR, "UKB_aging_model_age_prediction.csv")

pheno_files <- c(
  "phenotypes/cancer/ukb_cancer.csv",
  "phenotypes/cardiometabolic_vascular/ukb_cardio_vascular.csv",
  "phenotypes/musculoskeletal/ukb_musculoskeletal.csv",
  "phenotypes/dementia/ukb_all_cause_dementia.csv",
  "phenotypes/dementia/ukb_frontotemporal_dementia.csv",
  "phenotypes/dementia/ukb_vascular_dementia.csv",
  "phenotypes/dementia/ukb_alzheimer_disease.csv"
)
pheno_files <- file.path(DATA_DIR, pheno_files)

out_file <- file.path(DATA_DIR, "mediation_atlas_results.csv")

# ---- Load organ age estimates ----
data_age <- fread(age_file)

# ---- Loop over disease files and organ aging models ----
results <- list()

for (pheno_file in pheno_files) {

  data_pheno <- fread(pheno_file)
  disease_list <- colnames(data_pheno)[-1]
  tissue_list  <- unique(data_age$tissue)

  for (disease in disease_list) {
    cat("Running:", disease, "\n")

    for (tissue_name in tissue_list) {

      dat_t <- data_age[data_age$tissue == tissue_name, ]
      ID    <- intersect(dat_t$IID, data_pheno$eid)
      dat_t <- dat_t[match(ID, dat_t$IID), ]
      pheno <- data_pheno[match(ID, data_pheno$eid), ]

      # Restrict to held-out test split
      test_idx <- which(dat_t$split == "test")
      Age <- dat_t$Age[test_idx]
      M   <- dat_t$pred_Age[test_idx]
      Y   <- pheno[[disease]][test_idx]

      predicted_r <- cor(Age, M)

      # a-path (age -> biomarker), total effect, direct + b-path
      fit_a     <- lm(M ~ Age)
      fit_total <- lm(Y ~ Age)
      fit_b     <- lm(Y ~ Age + M)

      a         <- coef(fit_a)["Age"]
      b         <- coef(fit_b)["M"]
      tau       <- coef(fit_total)["Age"]
      tau_prime <- coef(fit_b)["Age"]

      mediation_effect <- a * b
      diff             <- tau - tau_prime
      prop_med         <- diff / tau

      se_a <- summary(fit_a)$coeff["Age", "Std. Error"]
      se_b <- summary(fit_b)$coeff["M",   "Std. Error"]

      p_a       <- summary(fit_a)$coeff["Age", "Pr(>|t|)"]
      p_b       <- summary(fit_b)$coeff["M",   "Pr(>|t|)"]
      p_total   <- summary(fit_total)$coeff["Age", "Pr(>|t|)"]
      p_direct  <- summary(fit_b)$coeff["Age", "Pr(>|t|)"]

      # Residual gap and calibrated gap
      gap_res <- residuals(fit_a)
      gap_cal <- gap_res / a

      fit_res <- lm(Y ~ gap_res)
      beta_res <- coef(fit_res)["gap_res"]
      se_res   <- summary(fit_res)$coeff["gap_res", "Std. Error"]
      p_res    <- summary(fit_res)$coeff["gap_res", "Pr(>|t|)"]

      fit_cal <- lm(Y ~ gap_cal)
      beta_cal <- coef(fit_cal)["gap_cal"]
      se_cal   <- summary(fit_cal)$coeff["gap_cal", "Std. Error"]
      p_cal    <- summary(fit_cal)$coeff["gap_cal", "Pr(>|t|)"]

      # Algebraic identity checks (should all be ~0)
      check_residual_equals_b      <- beta_res - b
      check_calibrated_equals_ab   <- beta_cal - mediation_effect
      check_scaling_a              <- beta_cal / beta_res - a

      # Age x biomarker interaction
      fit_int <- lm(Y ~ Age * M)
      h       <- coef(fit_int)["Age:M"]
      se_h    <- summary(fit_int)$coeff["Age:M", "Std. Error"]
      p_h     <- summary(fit_int)$coeff["Age:M", "Pr(>|t|)"]

      results[[length(results) + 1]] <- data.frame(
        Tissue = tissue_name,
        Disease = disease,
        Predicted_age_vs_age_pearson_r = round(predicted_r, 3),

        age_to_bioage_effect = a, age_to_bioage_std = se_a, p_age_to_bioage = p_a,
        bioage_to_disease_effect = b, bioage_to_disease_std = se_b, p_bioage_to_disease = p_b,

        total_effect = tau, p_total = p_total,
        direct_effect = tau_prime, p_direct = p_direct,
        mediation_effect = mediation_effect,
        prop_mediated = round(prop_med, 3),

        interaction_h = h, interaction_se = se_h, p_interaction = p_h,

        beta_residual = beta_res, se_residual = se_res, p_residual = p_res,
        beta_calibrated = beta_cal, se_calibrated = se_cal, p_calibrated = p_cal,

        check_residual_equals_b = check_residual_equals_b,
        check_calibrated_equals_ab = check_calibrated_equals_ab,
        check_scaling_a = check_scaling_a
      )
    }
  }
}

results_df <- rbindlist(results)
write.csv(results_df, out_file, row.names = FALSE)
