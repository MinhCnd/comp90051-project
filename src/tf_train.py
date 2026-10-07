import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, ExperimentConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import FTTransformerConfig

from src.metrics import get_roc_auc, get_ece, get_f1_precision_recall
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
        np.ndarray: Prediction probabilities for both classes and labels for the validation set.
    """

    data_config = DataConfig(target=[target_col],
                             continuous_cols=continuous_cols,
                             categorical_cols=categorical_cols,
                             num_workers=7,
                             dataloader_kwargs={"persistent_workers": True})

    trainer_config = TrainerConfig(
        batch_size = 256,
        max_epochs = 12,
        early_stopping="valid_loss",
        early_stopping_patience=5,
        progress_bar="none"
    )

    experiment_config = ExperimentConfig(
        project_name="CreditDefault",
        run_name="FT-Transformer",
        log_target="tensorboard",
    )

    optimizer_config = OptimizerConfig(
        optimizer="AdamW"
    )

    # Weight target class (1) higher than non-target class (0)
    # custom_loss = None
    weights_tensor = torch.tensor([1.0, 2.0])
    custom_loss = torch.nn.CrossEntropyLoss(weight=weights_tensor)

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
        optimizer_config=optimizer_config,
        experiment_config=experiment_config,
    )

    tabular_model.fit(train_df, validation=val_df, loss=custom_loss)

    preds = tabular_model.predict(val_df)
    if 'TARGET_1_probability' not in preds.columns or 'TARGET_prediction' not in preds.columns or 'TARGET_0_probability' not in preds.columns:
        raise ValueError("Expected columns 'TARGET_0_probability', 'TARGET_1_probability' and 'TARGET_prediction' not found in predictions.")

    res = pd.concat([preds['TARGET_0_probability'], preds['TARGET_1_probability'], preds['TARGET_prediction']], axis=1)
    res.columns = ['TARGET_0_probability', 'TARGET_1_probability', 'predicted_label']
    return res.to_numpy()

def run_nested_cv_ft_transformer(df: pd.DataFrame, 
                                 target_col: str, 
                                 continuous_cols: list[str], 
                                 categorical_cols: list[str], 
                                 target_ratio=0.08, 
                                 outer_k=10, 
                                 inner_k=10, 
                                 param_grid: list[dict] | None = None,
                                 debug: bool = False):
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
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        dict: Dictionary containing the results of the nested cross-validation.
    """

    if param_grid is None:
        param_grid = {
            "learning_rate": [1e-2, 1e-3, 1e-4],
            "input_embed_dim": [32, 64, 128],
            "num_heads": [4, 8, 16]
        }

    param_grid = list(ParameterGrid(param_grid))

    if debug:
        param_grid = param_grid[:1]  # Use only the first set of parameters in debug mode
        outer_k = 2  # Reduce the number of outer folds in debug mode
        inner_k = 2  # Reduce the number of inner folds in debug mode

    outer_f1_scores = []
    outer_precision_scores = []
    outer_recall_scores = []
    outer_auc_scores = []
    outer_ece_scores = []

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
            inner_val_f1s = []
            inner_val_precisions = []
            inner_val_recalls = []
            inner_val_eces = []
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

                #max pred and label
                p0 = val_probs[:, 0]
                p1 = val_probs[:, 1]
                pred_labels = val_probs[:, 2]
                max_probs = np.maximum(p0, p1)
                max_probs = np.concatenate([max_probs[:, None], pred_labels[:, None]], axis=1)
                val_f1, val_precision, val_recall = get_f1_precision_recall(pred_labels, y_inner_val)
                val_ece = get_ece(max_probs, y_inner_val)
                p1_labels = np.concatenate([p1[:, None], pred_labels[:, None]], axis=1)
                val_auc = get_roc_auc(p1_labels, y_inner_val)

                inner_val_f1s.append(val_f1)
                inner_val_precisions.append(val_precision)
                inner_val_recalls.append(val_recall)
                inner_val_eces.append(val_ece)
                inner_val_aucs.append(val_auc)

            mean_inner_auc = np.mean(inner_val_aucs)
            logger.debug(f" Param Candidate {p_idx + 1} ({params}): Mean Inner AUC = {mean_inner_auc:.4f}")
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

        outer_val_auc = get_roc_auc(outer_val_probs, y_outer_test)
        logger.info(f"Outer fold {outer_fold_idx + 1}/{outer_k}: AUC = {outer_val_auc}")
        outer_f1, outer_precision, outer_recall = get_f1_precision_recall(outer_val_probs[:, 1], y_outer_test)
        outer_ece = get_ece(outer_val_probs, y_outer_test)
        outer_f1_scores.append(outer_f1)
        outer_precision_scores.append(outer_precision)
        outer_recall_scores.append(outer_recall)
        outer_ece_scores.append(outer_ece)
        outer_auc_scores.append(outer_val_auc)

        # Append these results to a csv
        file_name = "./temp/ft_transformer_test_results.csv"
        pd.DataFrame({
            "param_idx": [p_idx + 1],
            "fold": [outer_fold_idx + 1],
            "f1": [outer_f1],
            "precision": [outer_precision],
            "recall": [outer_recall],
            "ece": [outer_ece],
            "auc": [outer_val_auc],
            "hparams": [best_params]
        }).to_csv(file_name, mode="a", header=not os.path.exists(file_name), index=False)

    # Final Mean & Std.dev across outer folds
    mean_outer_f1 = np.mean(outer_f1_scores)
    std_outer_f1 = np.std(outer_f1_scores)
    mean_outer_precision = np.mean(outer_precision_scores)
    std_outer_precision = np.std(outer_precision_scores)
    mean_outer_recall = np.mean(outer_recall_scores)
    std_outer_recall = np.std(outer_recall_scores)
    mean_outer_ece = np.mean(outer_ece_scores)
    std_outer_ece = np.std(outer_ece_scores)
    mean_outer_auc = np.mean(outer_auc_scores)
    std_outer_auc = np.std(outer_auc_scores)

    logger.info("FINAL NESTED EXPERIMENTAL RESULTS")
    logger.info(f"Mean Outer F1 = {mean_outer_f1}, Std.dev = {std_outer_f1}")
    logger.info(f"Mean Outer Precision = {mean_outer_precision}, Std.dev = {std_outer_precision}")
    logger.info(f"Mean Outer Recall = {mean_outer_recall}, Std.dev = {std_outer_recall}")
    logger.info(f"Mean Outer ECE = {mean_outer_ece}, Std.dev = {std_outer_ece}")
    logger.info(f"Mean Outer AUC = {mean_outer_auc}, Std.dev = {std_outer_auc}")

    pd.DataFrame({
        "fold": list(range(1, len(outer_auc_scores) + 1)),
        "f1": outer_f1_scores,
        "precision": outer_precision_scores,
        "recall": outer_recall_scores,
        "ece": outer_ece_scores,
        "auc": outer_auc_scores,
    }).to_csv("./temp/ft_transformer_fold_results_final.csv", index=False)

    return {
        "f1_mean": mean_outer_f1,
        "f1_std": std_outer_f1,
        "precision_mean": mean_outer_precision,
        "precision_std": std_outer_precision,
        "recall_mean": mean_outer_recall,
        "recall_std": std_outer_recall,
        "ece_mean": mean_outer_ece,
        "ece_std": std_outer_ece,
        "auc_mean": mean_outer_auc,
        "auc_std": std_outer_auc,
        "outer_aucs": outer_auc_scores
    }

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    args = arg_parser.parse_args()
    df = pd.read_csv("./temp/subsampled_data_1_1.csv")

    debug = args.debug

    if debug:
        df = subsample_to_ratio(df, target_col="TARGET", target_ratio=0.30, max_samples=10000, seed=50)

    # df = df.sample(frac=0.12, random_state=50)

    cont_cols, cat_cols = get_continuous_categorical_columns()

    #For TF-transformer, since cat_cols has been one-hot encoded, we don't need to pass them as categorical features.
    cont_cols = cont_cols + cat_cols

    param_grid = {
                "learning_rate": [9e-4, 8e-5, 7e-6],
                "input_embed_dim": [32, 64, 128],
                "num_heads": [8, 16, 32]
            }

    res = run_nested_cv_ft_transformer(
        df=df,
        target_col="TARGET",
        continuous_cols=cont_cols,
        categorical_cols=[],
        target_ratio=0.30,
        outer_k=10,
        inner_k=5,
        param_grid=param_grid,
        debug=debug
    )