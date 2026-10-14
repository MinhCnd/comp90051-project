tf_5 = read.csv('/Users/minhcnd/Dropbox/University/Courses/COMP90051/Group Project/results/Transformer/5_prcnt_imbalance/test_results.csv')
tf_20 = read.csv('/Users/minhcnd/Dropbox/University/Courses/COMP90051/Group Project/results/Transformer/20_prcnt_imbalance/test_results.csv')
tf_40 = read.csv('/Users/minhcnd/Dropbox/University/Courses/COMP90051/Group Project/results/Transformer/40_prcnt_imbalance/output/test_results.csv')

log_res_5 = read.csv('/Users/minhcnd/Dropbox/University/Courses/COMP90051/Group Project/results/Logistic Regression/logreg_5_percent_test_results.csv')
log_res_20 = read.csv('/Users/minhcnd/Dropbox/University/Courses/COMP90051/Group Project/results/Logistic Regression/logreg_20_percent_test_results.csv')
log_res_40 = read.csv('/Users/minhcnd/Dropbox/University/Courses/COMP90051/Group Project/results/Logistic Regression/logreg_40_percent_test_results.csv')

xg_5 = read.csv('/Users/minhcnd/Dropbox/University/Courses/COMP90051/Group Project/results/XGBoost/xgboost_esnum_5_percent_test_results.csv')
xg_20 = read.csv('/Users/minhcnd/Dropbox/University/Courses/COMP90051/Group Project/results/XGBoost/xgboost_esnum_20_percent_test_results.csv')
xg_40 = read.csv('/Users/minhcnd/Dropbox/University/Courses/COMP90051/Group Project/results/XGBoost/xgboost_esnum_40_percent_test_results.csv')

run_t_test_comparison <- function(scores_baseline, scores_shifted, k=10,
                                    n_train_prop = 9000, n_test_prop = 1000) {
  #' Evaluates Nadeau & Bengio (2003) corrected resampled t-test
  #' 
  #' @param scores_baseline Numeric vector of fold scores at natural ratio (~8%)
  #' @param scores_shifted Numeric vector of fold scores at shifted ratio (e.g., 30%)
  #' @param k Number of train-test folds
  #' @param n_train_prop Proportion of dataset in training fold (default 0.9 for 10-fold CV)
  #' @param n_test_prop Proportion of dataset in test fold (default 0.1 for 10-fold CV)
  
  # Input validation
  if (length(scores_baseline) != k || length(scores_shifted) != k) {
    stop(paste("Both score vectors must have length equal to k =",k))
  }
  
  # Compute paired differences
  diffs = scores_baseline - scores_shifted
  mean_diff = mean(diffs)
  var_diff = var(diffs)
  
  # Nadeau & Bengio correction factor for var
  corr_factor = (1/k) + (n_test_prop/n_train_prop)
  var_corr = var_diff*corr_factor
  std_dev_corr = sqrt(var_corr)
  
  t_stat = mean_diff / std_dev_corr
  df = k - 1
  
  # 2-tailed p-val calc
  p_val = 2* pt(-abs(t_stat),df,0,lower.tail = TRUE)
  
  return(list(
    mean_diff = mean_diff,
    t_stat = t_stat,
    p_val = p_val
  ))
}

run_ratio_shift_t_test <- function(baseline_scores, shifted_scores_list, 
                                           metric_name = "ECE", 
                                           model_name = "XGBoost", 
                                           n_train_prop = 9000, n_test_prop = 1000, 
                                           alpha = 0.05) {
  
  num_comparisons <- length(shifted_scores_list)
  alpha_bonferroni <- alpha / num_comparisons
  
  results_list <- list()
  
  for (ratio_label in names(shifted_scores_list)) {
    shifted_scores <- shifted_scores_list[[ratio_label]]
    
    # Call resampled t_test func
    test_res <- run_t_test_comparison(
      scores_baseline = baseline_scores,
      scores_shifted = shifted_scores,
      n_train_prop = n_train_prop,
      n_test_prop = n_test_prop
    )
    
    # Extract metrics from output object
    t_stat <- as.numeric(test_res$t_stat)
    p_val <- as.numeric(test_res$p_val)
    mean_diff <- mean(baseline_scores - shifted_scores)
    
    is_sig <- p_val < alpha_bonferroni
    
    results_list[[ratio_label]] <- data.frame(
      Model = model_name,
      Metric = metric_name,
      Comparison = paste0("Baseline(5%) vs ", ratio_label),
      Mean_Diff = round(mean_diff, 4),
      t_statistic = round(t_stat, 3),
      p_value = ifelse(p_val < 0.0001, "< 0.0001", sprintf("%.4f", p_val)),
      Alpha_Bonferroni = round(alpha_bonferroni, 4),
      Statistically_Significant = is_sig,
      stringsAsFactors = FALSE
    )
  }
  
  summary_df <- do.call(rbind, results_list)
  rownames(summary_df) <- NULL
  return(summary_df)
}

#--------------------------------------------------------------------------------
# Results for Log Res
ece_baseline = log_res_5$ece
ece_shifted_ratios = list(
  "20% Ratio" = log_res_20$ece,
  "40% Ratio" = log_res_40$ece
)

# Run test suite for ECE (Expected Calibration Error)
lr_ece_results_table <- run_ratio_shift_t_test(
  baseline_scores = ece_baseline,
  shifted_scores_list = ece_shifted_ratios,
  metric_name = "ECE",
  model_name = "Logistic Regression"
)


# Outer Fold ROC-AUC Scores (demonstrating rank-invariance stability)
auc_baseline = log_res_5$auc
auc_shifted_ratios <- list(
  "20% Ratio" = log_res_20$auc,
  "40% Ratio" = log_res_40$auc
)

# Run test suite for ROC-AUC
lr_auc_results_table <- run_ratio_shift_t_test(
  baseline_scores = auc_baseline,
  shifted_scores_list = auc_shifted_ratios,
  metric_name = "ROC-AUC",
  model_name = "Logistic Regression"
)

#--------------------------------------------------------------------------------
# Results for XGBoost
ece_baseline = xg_5$ece
ece_shifted_ratios = list(
  "20% Ratio" = xg_20$ece,
  "40% Ratio" = xg_40$ece
)

# Run test suite for ECE (Expected Calibration Error)
xg_ece_results_table <- run_ratio_shift_t_test(
  baseline_scores = ece_baseline,
  shifted_scores_list = ece_shifted_ratios,
  metric_name = "ECE",
  model_name = "XGBoost"
)

# Outer Fold ROC-AUC Scores (demonstrating rank-invariance stability)
auc_baseline = xg_5$auc
auc_shifted_ratios <- list(
  "20% Ratio" = xg_20$auc,
  "40% Ratio" = xg_40$auc
)

# Run test suite for ROC-AUC
xg_auc_results_table <- run_ratio_shift_t_test(
  baseline_scores = auc_baseline,
  shifted_scores_list = auc_shifted_ratios,
  metric_name = "ROC-AUC",
  model_name = "XGBoost"
)

#--------------------------------------------------------------------------------
# Results for Transformer
ece_baseline = tf_5$ece
ece_shifted_ratios = list(
  "20% Ratio" = tf_20$ece,
  "40% Ratio" = tf_40$ece
)

# Run test suite for ECE (Expected Calibration Error)
tf_ece_results_table <- run_ratio_shift_t_test(
  baseline_scores = ece_baseline,
  shifted_scores_list = ece_shifted_ratios,
  metric_name = "ECE",
  model_name = "Transformer"
)

# Outer Fold ROC-AUC Scores (demonstrating rank-invariance stability)
auc_baseline = tf_5$auc
auc_shifted_ratios <- list(
  "20% Ratio" = tf_20$auc,
  "40% Ratio" = tf_40$auc
)

# Run test suite for ROC-AUC
tf_auc_results_table <- run_ratio_shift_t_test(
  baseline_scores = auc_baseline,
  shifted_scores_list = auc_shifted_ratios,
  metric_name = "ROC-AUC",
  model_name = "Transformer"
)


#---------------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat(" NADEAU & BENGIO (2003) CORRECTED T-TEST RESULTS (Logistic Regression) - ECE\n")
cat("==============================================================================\n\n")
print(lr_ece_results_table)
print(lr_auc_results_table)

cat("\n==============================================================================\n")
cat(" NADEAU & BENGIO (2003) CORRECTED T-TEST RESULTS (XGBoost) - ECE\n")
cat("==============================================================================\n\n")
print(xg_ece_results_table)
print(xg_auc_results_table)

cat("\n==============================================================================\n")
cat(" NADEAU & BENGIO (2003) CORRECTED T-TEST RESULTS (Transformer) - ECE\n")
cat("==============================================================================\n\n")
print(tf_ece_results_table)
print(tf_auc_results_table)