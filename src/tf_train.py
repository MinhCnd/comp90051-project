import logging

import numpy as np
import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import FTTransformerConfig

from src.metrics import get_auc
from src.pipeline import (get_continuous_categorical_columns,
                          stratified_k_fold_split, subsample_to_ratio)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_eval_ft_transformer(train_df: pd.DataFrame, 
                              val_df: pd.DataFrame,
                              continuous_cols: list[str],
                              categorical_cols: list[str],
                              target_col: str,
                              hparams: dict) -> np.ndarray:
    """
    Train and evaluate the FTTransformer model.

    Args:
        train_df (pd.DataFrame): Training dataframe as a Pandas DataFrame.
        val_df (pd.DataFrame): Validation dataframe as a Pandas DataFrame.
        continuous_cols (list[str]): List of continuous column names.
        categorical_cols (list[str]): List of categorical column names.
        target_col (str): Name of the target column.
        hparams (dict): Hyperparameters for the FTTransformer model.

    Returns:
        np.ndarray: Predicted probabilities and labels for the validation set.
    """

    data_config = DataConfig(target=[target_col],
                             continuous_cols=continuous_cols,
                             categorical_cols=categorical_cols,
                             num_workers=7,
                             dataloader_kwargs={"persistent_workers": True})

    trainer_config = TrainerConfig(
        batch_size = 256,
        max_epochs = 1,
        early_stopping="valid_loss",
        early_stopping_patience=3,
        progress_bar="simple"
    )

    optimizer_config = OptimizerConfig(
        optimizer="Adam"
    )

    model_config = FTTransformerConfig(
        task="classification",
        learning_rate=hparams.get("learning_rate", 1e-3),
        input_embed_dim=hparams.get("input_embed_dim", 32),
        num_heads=hparams.get("num_heads", 4),
        num_attn_blocks=hparams.get("num_attn_blocks", 2)
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config
    )

    tabular_model.fit(train_df, validation=val_df)

    preds = tabular_model.predict(val_df)

    if 'TARGET_1_probability' not in preds.columns or 'TARGET_prediction' not in preds.columns or 'TARGET_0_probability' not in preds.columns:
        raise ValueError("Expected columns 'TARGET_0_probability', 'TARGET_1_probability' and 'TARGET_prediction' not found in predictions.")

    max_probs = preds[['TARGET_0_probability', 'TARGET_1_probability']].max(axis=1)
    res = pd.concat([max_probs,preds['TARGET_prediction']],axis=1)
    res.columns = ['max_prob', 'predicted_label']
    return res.to_numpy()

def run_nested_cv_ft_transformer(df: pd.DataFrame, 
                                 target_col: str, 
                                 continuous_cols: list[str], 
                                 categorical_cols: list[str], 
                                 target_ratio=0.08, 
                                 outer_k=10, 
                                 inner_k=10, 
                                 param_grid: list[dict] | None = None):
    """
    Run nested cross-validation for the FTTransformer model.

    Args:
        df (pd.DataFrame): Input dataframe as a Pandas DataFrame.
        target_col (str): Name of the target column.
        continuous_cols (list[str]): List of continuous column names.
        categorical_cols (list[str]): List of categorical column names.
        target_ratio (float, optional): Desired ratio of the positive class. Defaults to 0.08.
        outer_k (int, optional): Number of outer folds. Defaults to 10.
        inner_k (int, optional): Number of inner folds. Defaults to 10.
        param_grid (list[dict], optional): Hyperparameter grid for inner cross-validation. Defaults to None.

    Returns:
        dict: Dictionary containing the results of the nested cross-validation.
    """

    if param_grid is None:
        param_grid = [
            {"learning_rate": 1e-3, "input_embed_dim": 32, "num_heads": 4},
            {"learning_rate": 5e-4, "input_embed_dim": 32, "num_heads": 4},
            {"learning_rate": 1e-4, "input_embed_dim": 64, "num_heads": 16},
        ]

    outer_auc_scores = []

    y_all = df[target_col].values
    outer_folds = stratified_k_fold_split(y_all, k=outer_k, seed=23)

    logger.info(f"Starting Nested Stratified K-Fold Cross-Validation ({outer_k} outer folds, {inner_k} inner folds)")

    for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
        logger.info(f"Starting outer fold {outer_fold_idx + 1}/{outer_k}")
        X_outer_train, X_outer_test = df.iloc[train_idx], df.iloc[test_idx]
        y_outer_train, y_outer_test = y_all[train_idx], y_all[test_idx]
        
        inner_folds = stratified_k_fold_split(y_outer_train, k=inner_k, seed=100+outer_fold_idx)

        best_params = None
        best_inner_score = -np.inf

        for p_idx, params in enumerate(param_grid):
            inner_val_aucs = []

            for inner_train_idx, inner_val_idx in inner_folds:
                X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[inner_val_idx]
                y_inner_val = y_outer_train[inner_val_idx]

                X_inner_train_shifted = subsample_to_ratio(X_inner_train, target_col, target_ratio)

                val_probs = train_eval_ft_transformer(
                    train_df=X_inner_train_shifted,
                    val_df=X_inner_val,
                    continuous_cols=continuous_cols,
                    categorical_cols=categorical_cols,
                    target_col=target_col,
                    hparams=params
                )

                val_auc = get_auc(val_probs, y_inner_val)
                inner_val_aucs.append(val_auc)

            mean_inner_auc = np.mean(inner_val_aucs)
            logger.debug(f" Param Candidate {p_idx + 1} ({params}): Mean Inner Validation AUC = {mean_inner_auc:.4f}")
            if mean_inner_auc > best_inner_score:
                best_inner_score = mean_inner_auc
                best_params = params

        logger.info(f"  Best params for inner fold {outer_fold_idx + 1}/{outer_k}: {best_params} with score {best_inner_score}")

        # Outer Evaluation: Final Training & testing
        # Shift the outer training set to match the target ratio
        X_outer_train_shifted = subsample_to_ratio(X_outer_train, target_col, target_ratio)

        outer_val_probs = train_eval_ft_transformer(
            train_df=X_outer_train_shifted,
            val_df=X_outer_test,
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            target_col=target_col,
            hparams=best_params
        )

        outer_val_auc = get_auc(outer_val_probs, y_outer_test)
        logger.info(f"Outer fold {outer_fold_idx + 1}/{outer_k}: AUC = {outer_val_auc}")
        outer_auc_scores.append(outer_val_auc)

    # Final Mean & Std.dev across outer folds
    mean_outer_auc = np.mean(outer_auc_scores)
    std_outer_auc = np.std(outer_auc_scores)
    logger.info(f"Final Mean Outer AUC = {mean_outer_auc}, Std.dev = {std_outer_auc}")

    logger.info("FINAL NESTED EXPERIMENTAL RESULTS")
    logger.info(f"Mean Outer AUC = {mean_outer_auc}, Std.dev = {std_outer_auc}")

    return {
        "auc_mean": mean_outer_auc,
        "auc_std": std_outer_auc,
        "outer_aucs": outer_auc_scores
    }

if __name__ == "__main__":
    df = pd.read_csv("./temp/processed_data.csv")

    df = df.sample(frac=0.12, random_state=42)

    cont_cols, cat_cols = get_continuous_categorical_columns()

    res = run_nested_cv_ft_transformer(
        df=df,
        target_col="TARGET",
        continuous_cols=cont_cols,
        categorical_cols=cat_cols,
        target_ratio=0.12,
        outer_k=2,
        inner_k=2,
        param_grid=None
    )
    print(res)