import numpy as np
import pandas as pd


def split_feature_and_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    X: pd.DataFrame = df.drop(columns=[target_column])
    y: pd.Series = df[target_column]
    return X, y

def get_continuous_categorical_columns() -> tuple[list[str], list[str]]:
    cont_cols = [
        'REGION_POPULATION_RELATIVE', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_2',
        'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
        'AGE_YEARS', 'YEARS_REGISTRATION', 'YEARS_ID_PUBLISH',
        'YEARS_LAST_PHONE_CHANGE', 'AMT_INCOME_TOTAL_SCALED',
        'AMT_CREDIT_SCALED', 'AMT_ANNUITY_SCALED', 'AMT_GOODS_PRICE_SCALED',
        'INCOME_CREDIT_PERC', 'INCOME_PER_PERSON_SCALED',
        'ANNUITY_INCOME_PERC', 'PAYMENT_RATE', 'EXT_SOURCE_MEAN'
    ]

    cat_cols = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
        'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
        'FLAG_EMAIL', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
        'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
        'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
        'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
        'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
        'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
        'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
        'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
        'NAME_CONTRACT_TYPE_Cash loans', 'NAME_CONTRACT_TYPE_Revolving loans',
        'NAME_TYPE_SUITE_Children', 'NAME_TYPE_SUITE_Family', 'NAME_TYPE_SUITE_Group of people',
        'NAME_TYPE_SUITE_Other_A', 'NAME_TYPE_SUITE_Other_B', 'NAME_TYPE_SUITE_Spouse partner',
        'NAME_TYPE_SUITE_Unaccompanied', 'NAME_INCOME_TYPE_Commercial associate',
        'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Working',
        'NAME_EDUCATION_TYPE_Higher education', 'NAME_EDUCATION_TYPE_Incomplete higher',
        'NAME_EDUCATION_TYPE_Lower secondary', 'NAME_EDUCATION_TYPE_Secondary  secondary special',
        'NAME_FAMILY_STATUS_Civil marriage', 'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated',
        'NAME_FAMILY_STATUS_Single  not married', 'NAME_FAMILY_STATUS_Widow',
        'NAME_HOUSING_TYPE_Co-op apartment', 'NAME_HOUSING_TYPE_House  apartment',
        'NAME_HOUSING_TYPE_Municipal apartment', 'NAME_HOUSING_TYPE_Office apartment',
        'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents',
        'OCCUPATION_TYPE_Accountants', 'OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff',
        'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers', 'OCCUPATION_TYPE_High skill tech staff',
        'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Laborers', 'OCCUPATION_TYPE_Low-skill Laborers',
        'OCCUPATION_TYPE_Managers', 'OCCUPATION_TYPE_Medicine staff', 'OCCUPATION_TYPE_Private service staff',
        'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff', 'OCCUPATION_TYPE_Secretaries',
        'OCCUPATION_TYPE_Security staff', 'OCCUPATION_TYPE_Waitersbarmen staff',
        'WEEKDAY_APPR_PROCESS_START_MONDAY', 'WEEKDAY_APPR_PROCESS_START_TUESDAY',
        'WEEKDAY_APPR_PROCESS_START_WEDNESDAY', 'WEEKDAY_APPR_PROCESS_START_THURSDAY',
        'WEEKDAY_APPR_PROCESS_START_FRIDAY', 'WEEKDAY_APPR_PROCESS_START_SATURDAY', 'WEEKDAY_APPR_PROCESS_START_SUNDAY'
    ]

    return cont_cols, cat_cols

def stratified_k_fold_split(y: np.ndarray, k=10, seed=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold splitting.
    
    Args:
        y (np.ndarray): Array of true binary labels (0 or 1).
        k (int, optional): Number of folds. Defaults to 10.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: List of tuples containing train and validation indices for each fold.
    """

    if seed is not None:
        np.random.seed(seed)

    # Separate indices by class
    idx_class0 = np.where(y == 0)[0]
    idx_class1 = np.where(y == 1)[0]

    # shuffle the class indices to ensure unbiased fold assignment
    np.random.shuffle(idx_class0)
    np.random.shuffle(idx_class1)

    # Split class indices into K chunks, reversing the order of class 1 chunks to ensure varied validation sets across folds
    splits_c0 = np.array_split(idx_class0, k)
    splits_c1 = np.array_split(idx_class1, k)[::-1]

    folds = []
    for i in range(k):
        val_idx_class0 = splits_c0[i]
        val_idx_class1 = splits_c1[i]
        # for k-th fold, the corresponding chunks of each class are used as the validation set
        # the remaining indices are used as the training set
        val_idx = np.concatenate([val_idx_class0, val_idx_class1])
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        folds.append((train_idx, val_idx))

    return folds

def subsample_to_ratio(df: pd.DataFrame, target_col: str, target_ratio: float, seed=None) -> pd.DataFrame:
    """Subsample the dataframe to achieve a target class ratio.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Name of the target column.
        target_ratio (float): Desired ratio of the positive class (0 < target_ratio < 1).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: Subsampled dataframe as a Pandas DataFrame.
    """

    if target_ratio <= 0 or target_ratio >= 1:
        raise ValueError("target_ratio must be between 0 and 1.")
    
    if seed is not None:
        np.random.seed(seed)

    df_pos = df[df[target_col] == 1]
    df_neg = df[df[target_col] == 0]

    n_pos = len(df_pos)

    if n_pos == 0:
        raise ValueError("No positive samples found in the dataframe.")
    
    n_neg_wanted = int(n_pos * (1 - target_ratio) / target_ratio)

    if n_neg_wanted > len(df_neg):
        raise ValueError("Not enough samples in class 0 to achieve the target ratio.")

    sampled_neg_indx = np.random.choice(np.arange(len(df_neg)), size=n_neg_wanted, replace=False)
    df_neg_sampled = df_neg.iloc[sampled_neg_indx]
    subsampled_df = pd.concat([df_pos, df_neg_sampled])
    subsampled_df = subsampled_df.sample(frac=1).reset_index(drop=True)

    return subsampled_df