# K-Fold LA Ablation Conclusion

Based on the 5-fold cross-validation results across 3 seeds for the Logit Adjustment (LA) configurations:

*   **M5_LA_FF**: Test File F1 0.5736 ± 0.1102
*   **M5_LA_FT**: Test File F1 0.5698 ± 0.1021
*   **M5_LA_TF**: Test File F1 0.5961 ± 0.0942
*   **M5_LA_TT**: Test File F1 0.5980 ± 0.1339

### Key Findings & Recommendations

1.  **Performance:** Among the four settings, `TF` (Train True, Eval False) and `TT` (Train True, Eval True) perform the best on average (F1 ≈ 0.596 ~ 0.598), with very little difference between them.
2.  **Fluctuation:** The standard deviation is quite large (0.09 ~ 0.13), indicating that the evaluation fluctuation for this task is very obvious. This is strongly correlated with the small evaluation set sizes (e.g., the test set only has 13 files, and the validation set per fold only has 9~12 files).
3.  **Default Configuration:** We strongly recommend **`M5_LA_TF`** as the default configuration. While its mean F1 score is almost identical to `TT`, it exhibits a notably smaller standard deviation (0.0942 vs. 0.1339), making the model's performance more stable and reliable.

*Redundant configurations (`M5_LA_FF`, `M5_LA_FT`, `M5_LA_TT`) have been removed from the k-fold generation pool and default experiments list.*
