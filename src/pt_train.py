import argparse
import functools
import time
import os
import logging
import pandas as pd
import numpy as np
import torch
from typing import Optional
import numpy as np
import pandas as pd
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, ExperimentConfig, OptimizerConfig, TrainerConfig, ModelConfig
from pytorch_tabular.models import FTTransformerConfig

from src.metrics import get_metrics
from src.pipeline import (get_continuous_categorical_columns,
                          stratified_k_fold_split, subsample_to_ratio, generate_parameters)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def timer(func):
    """Decorator that logs the execution time of the wrapped function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.info("%s took %.4f seconds", func.__name__, elapsed)
    return wrapper

def update_model_config(model: TabularModel, params: dict) -> TabularModel:
    model_config = model.config
    for key, value in params.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
        else:
            logger.warning(f"Unknown model config hyperparameter '{key}' ignored.")
    return model

def check_class_ratio(labels: np.ndarray, name: str, target_ratio: float, tolerance: float = 0.05) -> None:
    """Check if the class ratio of the given labels matches the target ratio.

    Args:
        labels (np.ndarray): Array of labels.
        name (str): Name of the dataset or split being checked.
        target_ratio (float): Target class ratio for the positive class.
        tolerance (float, optional): Allowed deviation from the target ratio. Defaults to 0.05.

    Raises:
        ValueError: If the class ratio differs from the target ratio by more than the specified tolerance.
    """
    class_ratio = np.mean(labels == 1)
    if abs(class_ratio - target_ratio) > tolerance:
        raise ValueError(f"Class ratio for {name} {class_ratio:.4f} differs from target ratio {target_ratio:.4f}")

@timer
def get_model(target_col: str, continuous_cols: list[str], categorical_cols: list[str], model_config: ModelConfig) -> TabularModel:
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
        run_name=model_config._model_name,
        log_target="tensorboard",
    )

    optimizer_config = OptimizerConfig(
        optimizer="AdamW"
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
        experiment_config=experiment_config,
    )
    return tabular_model

def train(train_df: pd.DataFrame, 
                val_df: pd.DataFrame,
                model: TabularModel,
                params: Optional[dict] = None) -> None:
    """
    Train the FTTransformer model.
    
    Args:
        train_df (pd.DataFrame): Training dataframe as a Pandas DataFrame.
        val_df (pd.DataFrame): Validation dataframe as a Pandas DataFrame.
        model (TabularModel): The PyTorch Tabular model to be trained and evaluated.
        params (Optional[dict], optional): Additional parameters for training. Defaults to None.

    """

    weights_tensor = torch.tensor([1.0, 1.0])
    custom_loss = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    if params is not None and "custom_loss" in params:
        weights_tensor = torch.tensor(params["custom_loss"])
        custom_loss = torch.nn.CrossEntropyLoss(weight=weights_tensor)
        logger.info("Using custom loss with weights: %s", params["custom_loss"])

    model.fit(train_df, validation=val_df, loss=custom_loss)

def eval(model: TabularModel, val_df: pd.DataFrame) -> np.ndarray:
    """
    Evaluate the FTTransformer model on the validation set.

    Args:
        model (TabularModel): The PyTorch Tabular model to be evaluated.
        val_df (pd.DataFrame): Validation dataframe as a Pandas DataFrame.

    Returns:
        np.ndarray: Prediction probabilities for both classes and labels for the validation set.
    """
    preds = model.predict(val_df)
    if 'TARGET_1_probability' not in preds.columns or 'TARGET_prediction' not in preds.columns or 'TARGET_0_probability' not in preds.columns:
        raise ValueError("Expected columns 'TARGET_0_probability', 'TARGET_1_probability' and 'TARGET_prediction' not found in predictions.")

    res = pd.concat([preds['TARGET_0_probability'], preds['TARGET_1_probability'], preds['TARGET_prediction']], axis=1)
    res.columns = ['TARGET_0_probability', 'TARGET_1_probability', 'predicted_label']
    return res.to_numpy()

def run_nested_cv(df: pd.DataFrame, 
    target_col: str, 
    continuous_cols: list[str], 
    categorical_cols: list[str], 
    model_config: ModelConfig,
    target_ratio=0.08, 
    outer_k=10, 
    inner_k=10, 
    param_grid: list[dict] | None = None,
    debug: bool = False,
    save_path: str | None = None):
    """
    Run nested cross-validation for the FTTransformer model.

    Args:
        df (pd.DataFrame): Input dataframe as a Pandas DataFrame.
        target_col (str): Name of the target column.
        continuous_cols (list[str]): List of continuous column names.
        categorical_cols (list[str]): List of categorical column names.
        model_config (ModelConfig): Configuration for the PyTorch Tabular model.
        target_ratio (float, optional): Desired ratio of the positive class. Defaults to 0.08.
        outer_k (int, optional): Number of outer folds. Defaults to 10.
        inner_k (int, optional): Number of inner folds. Defaults to 10.
        param_grid (list[dict], optional): Hyperparameter grid for inner cross-validation. Defaults to None.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        save_path (str | None, optional): Path to save the trained model for each outer fold. Defaults to None.

    Returns:
        dict: Dictionary containing the results of the nested cross-validation.
    """

    param_grid = generate_parameters(param_grid)

    if debug:
        param_grid = param_grid[:1]  # Use only the first set of parameters in debug mode
        outer_k = 2  # Reduce the number of outer folds in debug mode
        inner_k = 2  # Reduce the number of inner folds in debug mode

    outer_f1_scores = []
    outer_precision_scores = []
    outer_recall_scores = []
    outer_auc_scores = []
    outer_accuracy_scores = []
    outer_ece_scores = []

    y_all = df[target_col].values
    outer_folds = stratified_k_fold_split(y_all, k=outer_k, seed=23)

    logger.info(f"Starting Nested Stratified K-Fold Cross-Validation ({outer_k} outer folds, {inner_k} inner folds)")
    model = get_model(target_col, continuous_cols, categorical_cols, model_config)

    for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
        logger.info(f"Starting outer fold {outer_fold_idx + 1}/{outer_k}")
        X_outer_train, X_outer_test = df.iloc[train_idx], df.iloc[test_idx]
        y_outer_train, y_outer_test = y_all[train_idx], y_all[test_idx]

        target_test_ratio = np.mean(y_all == 1)
        
        inner_folds = stratified_k_fold_split(y_outer_train, k=inner_k, seed=100+outer_fold_idx)

        best_params = None
        best_inner_score = -np.inf

        for p_idx, params in enumerate(param_grid):
            inner_val_f1s = []
            inner_val_precisions = []
            inner_val_recalls = []
            inner_val_accuracies = []
            inner_val_eces = []
            inner_val_aucs = []
            inner_val_accuracies = []

            fold_idx = 0
            for inner_train_idx, inner_val_idx in inner_folds:
                fold_idx += 1
                X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[inner_val_idx]
                y_inner_val = y_outer_train[inner_val_idx]

                check_class_ratio(y_inner_val, "Inner Validation", target_test_ratio, tolerance=0.1)

                X_inner_train_shifted = subsample_to_ratio(X_inner_train, target_col, target_ratio)

                model = get_model(target_col, continuous_cols, categorical_cols, model_config)
                model = update_model_config(model, params)

                train(
                    train_df=X_inner_train_shifted,
                    val_df=X_inner_val,
                    model=model,
                    params=params
                )

                val_probs = eval(model, X_inner_val)
                metrics = get_metrics(val_probs, y_inner_val)

                inner_val_f1s.append(metrics["f1"])
                inner_val_precisions.append(metrics["precision"])
                inner_val_recalls.append(metrics["recall"])
                inner_val_accuracies.append(metrics["accuracy"])
                inner_val_eces.append(metrics["ece"])
                inner_val_aucs.append(metrics["auc"])

            file_name = f"{save_path}/train_results.csv"
            for idx in range(len(inner_folds)):
                pd.DataFrame({
                    "outer_fold_idx": [outer_fold_idx + 1],
                    "inner_fold_idx": [idx + 1],
                    "param_idx": [p_idx + 1],
                    "fold": [idx + 1],
                    "accuracy": [inner_val_accuracies[idx]],
                    "f1": [inner_val_f1s[idx]],
                    "precision": [inner_val_precisions[idx]],
                    "recall": [inner_val_recalls[idx]],
                    "ece": [inner_val_eces[idx]],
                    "auc": [inner_val_aucs[idx]]
                }).to_csv(file_name, mode="a", header=not os.path.exists(file_name), index=False)

            # Log the mean inner fold metrics for the current parameter candidate
            mean_inner_auc = np.mean(inner_val_aucs)
            logger.debug(f" Param Candidate {p_idx + 1} ({params}): Mean Inner AUC = {mean_inner_auc:.4f}")
            if mean_inner_auc > best_inner_score:
                best_inner_score = mean_inner_auc
                best_params = params

        
        logger.info(f"  Best params for inner fold {outer_fold_idx + 1}/{outer_k}: {best_params} with score {best_inner_score}")

        # Outer Evaluation: Final Training & testing
        # Split the outer train set into train and validation sets
        X_outer_fit, X_outer_val, y_outer_train, y_outer_val = train_test_split(
            X_outer_train,
            y_outer_train,
            test_size=0.15,
            random_state=42,
            stratify=y_outer_train
        )

        check_class_ratio(y_outer_test, "Outer Test", target_test_ratio, tolerance=0.1)
        
        # Shift the outer training set to match the target ratio
        X_outer_train_shifted = subsample_to_ratio(X_outer_fit, target_col, target_ratio, seed=42)

        # Update model config with the best hyperparameters found in the inner folds
        if best_params is None:
            raise ValueError("Something went wrong: No best hyperparameters found from inner folds.")
        model = get_model(target_col, continuous_cols, categorical_cols, model_config)
        model = update_model_config(model, best_params)

        train(
            train_df=X_outer_train_shifted,
            val_df=X_outer_val,
            model=model,
            params=best_params
        )

        outer_val_probs = eval(model, X_outer_test)

        # Save the best model for current outer fold
        if save_path is not None:
            model.save_model(f"{save_path}/best_model_fold_{outer_fold_idx + 1}")
            logger.info(f"Saved best model of outer fold {outer_fold_idx + 1} to {save_path}")

        metrics = get_metrics(outer_val_probs, y_outer_test)
        outer_auc = metrics["auc"]
        outer_f1 = metrics["f1"]
        outer_precision = metrics["precision"]
        outer_recall = metrics["recall"]
        outer_ece = metrics["ece"]
        outer_accuracy = metrics["accuracy"]

        outer_f1_scores.append(outer_f1)
        outer_precision_scores.append(outer_precision)
        outer_recall_scores.append(outer_recall)
        outer_ece_scores.append(outer_ece)
        outer_accuracy_scores.append(outer_accuracy)
        outer_auc_scores.append(outer_auc)

        logger.info(f"Outer fold {outer_fold_idx + 1}/{outer_k}: AUC = {outer_auc}, Accuracy = {outer_accuracy}, F1 = {outer_f1}, Precision = {outer_precision}, Recall = {outer_recall}, ECE = {outer_ece}")


        # Append these results to a csv
        file_name = f"{save_path}/test_results.csv"
        pd.DataFrame({
            "param_idx": [p_idx + 1],
            "fold": [outer_fold_idx + 1],
            "f1": [outer_f1],
            "accuracy": [outer_accuracy],
            "precision": [outer_precision],
            "recall": [outer_recall],
            "ece": [outer_ece],
            "auc": [outer_auc]
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
    arg_parser.add_argument("-f", "--file", type=str, help="Path to the input CSV file", default="./temp/processed_data_1_3.csv")
    arg_parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    arg_parser.add_argument("-r", "--ratio", type=float, help="Target ratio for subsampling")
    args = arg_parser.parse_args()
    df = pd.read_csv(args.file)

    debug = args.debug
    target_ratio = args.ratio

    if target_ratio is None or target_ratio <= 0 or target_ratio > 1:
        raise ValueError("Target ratio must be a float between 0 and 1.")

    if debug:
        df = subsample_to_ratio(df, target_col="TARGET", target_ratio=0.20, max_samples=100, seed=50)

    model_name = "ft_transformer"

    cont_cols, cat_cols = get_continuous_categorical_columns()

    param_grid = None
    model_config = None

    # If defining hyperparameter grid for custom loss, make sure to use floats
    param_grid = {
                "custom_loss": [[1.0,1.0], [0.1,0.9], [0.3,0.7]]
            }

    # Hyperparams determined by results of previous experiments
    model_config = FTTransformerConfig(
        task="classification",
        learning_rate=8e-5,
        input_embed_dim=64,
        num_heads=16,
        num_attn_blocks=2
    )

    logger.info(f"Running nested CV with model: {model_name}")

    model_save_path = "./output"
    os.makedirs(model_save_path, exist_ok=True)

    res = run_nested_cv(
        df=df,
        target_col="TARGET",
        continuous_cols=cont_cols,
        categorical_cols=cat_cols,
        target_ratio=target_ratio,
        outer_k=10,
        inner_k=3,
        param_grid=param_grid,
        debug=debug,
        model_config=model_config,
        save_path=model_save_path
    )