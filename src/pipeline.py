from itertools import product
import numpy as np
import pandas as pd


def split_feature_and_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    X: pd.DataFrame = df.drop(columns=[target_column])
    y: pd.Series = df[target_column]
    return X, y

def get_continuous_categorical_columns() -> tuple[list[str], list[str]]:
    cont_cols = ["CNT_CHILDREN","REGION_POPULATION_RELATIVE","CNT_FAM_MEMBERS","REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","HOUR_APPR_PROCESS_START","EXT_SOURCE_2","OBS_30_CNT_SOCIAL_CIRCLE","DEF_30_CNT_SOCIAL_CIRCLE","OBS_60_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE","AGE_YEARS","YEARS_REGISTRATION","YEARS_ID_PUBLISH","YEARS_LAST_PHONE_CHANGE","AMT_INCOME_TOTAL_SCALED","AMT_CREDIT_SCALED","AMT_ANNUITY_SCALED","AMT_GOODS_PRICE_SCALED","INCOME_CREDIT_PERC","INCOME_PER_PERSON_SCALED","ANNUITY_INCOME_PERC","PAYMENT_RATE","EXT_SOURCE_MEAN","PREV_AMT_ANNUITY_SCALED_MIN","PREV_AMT_ANNUITY_SCALED_MAX","PREV_AMT_ANNUITY_SCALED_MEAN","PREV_AMT_APPLICATION_SCALED_MIN","PREV_AMT_APPLICATION_SCALED_MAX","PREV_AMT_APPLICATION_SCALED_MEAN","PREV_AMT_CREDIT_SCALED_MIN","PREV_AMT_CREDIT_SCALED_MAX","PREV_AMT_CREDIT_SCALED_MEAN","PREV_APP_CREDIT_PERC_MIN","PREV_APP_CREDIT_PERC_MAX","PREV_APP_CREDIT_PERC_MEAN","PREV_AMT_GOODS_PRICE_SCALED_MIN","PREV_AMT_GOODS_PRICE_SCALED_MAX","PREV_AMT_GOODS_PRICE_SCALED_MEAN","PREV_HOUR_APPR_PROCESS_START_MIN","PREV_HOUR_APPR_PROCESS_START_MAX","PREV_HOUR_APPR_PROCESS_START_MEAN","PREV_YEARS_DECISION_MIN","PREV_YEARS_DECISION_MAX","PREV_YEARS_DECISION_MEAN","PREV_CNT_PAYMENT_MEAN","PREV_CNT_PAYMENT_SUM","PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN","PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN","PREV_NAME_CONTRACT_TYPE_Revolving loans_MEAN","PREV_NAME_CONTRACT_TYPE_XNA_MEAN","PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN","PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN","PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN","PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN","PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN","PREV_WEEKDAY_APPR_PROCESS_START_TUESDAY_MEAN","PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_MEAN","PREV_FLAG_LAST_APPL_PER_CONTRACT_N_MEAN","PREV_FLAG_LAST_APPL_PER_CONTRACT_Y_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Business development_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Buying a garage_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home  land_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Buying a home_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Buying a used car_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Car repairs_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Education_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Everyday expenses_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Furniture_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Gasification  water supply_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Hobby_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Journey_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Medicine_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Money for a third person_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Payments on other loans_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Repairs_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Urgent needs_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_Wedding  gift  holiday_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_XAP_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN","PREV_NAME_CONTRACT_STATUS_Approved_MEAN","PREV_NAME_CONTRACT_STATUS_Canceled_MEAN","PREV_NAME_CONTRACT_STATUS_Refused_MEAN","PREV_NAME_CONTRACT_STATUS_Unused offer_MEAN","PREV_NAME_PAYMENT_TYPE_Cash through the bank_MEAN","PREV_NAME_PAYMENT_TYPE_Cashless from the account of the employer_MEAN","PREV_NAME_PAYMENT_TYPE_Non-cash from your account_MEAN","PREV_NAME_PAYMENT_TYPE_XNA_MEAN","PREV_CODE_REJECT_REASON_CLIENT_MEAN","PREV_CODE_REJECT_REASON_HC_MEAN","PREV_CODE_REJECT_REASON_LIMIT_MEAN","PREV_CODE_REJECT_REASON_SCO_MEAN","PREV_CODE_REJECT_REASON_SCOFR_MEAN","PREV_CODE_REJECT_REASON_SYSTEM_MEAN","PREV_CODE_REJECT_REASON_VERIF_MEAN","PREV_CODE_REJECT_REASON_XAP_MEAN","PREV_CODE_REJECT_REASON_XNA_MEAN","PREV_NAME_TYPE_SUITE_Children_MEAN","PREV_NAME_TYPE_SUITE_Family_MEAN","PREV_NAME_TYPE_SUITE_Group of people_MEAN","PREV_NAME_TYPE_SUITE_Other_A_MEAN","PREV_NAME_TYPE_SUITE_Other_B_MEAN","PREV_NAME_TYPE_SUITE_Spouse partner_MEAN","PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN","PREV_NAME_TYPE_SUITE_nan_MEAN","PREV_NAME_CLIENT_TYPE_New_MEAN","PREV_NAME_CLIENT_TYPE_Refreshed_MEAN","PREV_NAME_CLIENT_TYPE_Repeater_MEAN","PREV_NAME_CLIENT_TYPE_XNA_MEAN","PREV_NAME_GOODS_CATEGORY_Additional Service_MEAN","PREV_NAME_GOODS_CATEGORY_Animals_MEAN","PREV_NAME_GOODS_CATEGORY_AudioVideo_MEAN","PREV_NAME_GOODS_CATEGORY_Auto Accessories_MEAN","PREV_NAME_GOODS_CATEGORY_Clothing and Accessories_MEAN","PREV_NAME_GOODS_CATEGORY_Computers_MEAN","PREV_NAME_GOODS_CATEGORY_Construction Materials_MEAN","PREV_NAME_GOODS_CATEGORY_Consumer Electronics_MEAN","PREV_NAME_GOODS_CATEGORY_Direct Sales_MEAN","PREV_NAME_GOODS_CATEGORY_Education_MEAN","PREV_NAME_GOODS_CATEGORY_Fitness_MEAN","PREV_NAME_GOODS_CATEGORY_Furniture_MEAN","PREV_NAME_GOODS_CATEGORY_Gardening_MEAN","PREV_NAME_GOODS_CATEGORY_Homewares_MEAN","PREV_NAME_GOODS_CATEGORY_Insurance_MEAN","PREV_NAME_GOODS_CATEGORY_Jewelry_MEAN","PREV_NAME_GOODS_CATEGORY_Medical Supplies_MEAN","PREV_NAME_GOODS_CATEGORY_Medicine_MEAN","PREV_NAME_GOODS_CATEGORY_Mobile_MEAN","PREV_NAME_GOODS_CATEGORY_Office Appliances_MEAN","PREV_NAME_GOODS_CATEGORY_Other_MEAN","PREV_NAME_GOODS_CATEGORY_Photo  Cinema Equipment_MEAN","PREV_NAME_GOODS_CATEGORY_Sport and Leisure_MEAN","PREV_NAME_GOODS_CATEGORY_Tourism_MEAN","PREV_NAME_GOODS_CATEGORY_Vehicles_MEAN","PREV_NAME_GOODS_CATEGORY_Weapon_MEAN","PREV_NAME_GOODS_CATEGORY_XNA_MEAN","PREV_NAME_PORTFOLIO_Cards_MEAN","PREV_NAME_PORTFOLIO_Cars_MEAN","PREV_NAME_PORTFOLIO_Cash_MEAN","PREV_NAME_PORTFOLIO_POS_MEAN","PREV_NAME_PORTFOLIO_XNA_MEAN","PREV_NAME_PRODUCT_TYPE_XNA_MEAN","PREV_NAME_PRODUCT_TYPE_walk-in_MEAN","PREV_NAME_PRODUCT_TYPE_x-sell_MEAN","PREV_CHANNEL_TYPE_AP+ (Cash loan)_MEAN","PREV_CHANNEL_TYPE_Car dealer_MEAN","PREV_CHANNEL_TYPE_Channel of corporate sales_MEAN","PREV_CHANNEL_TYPE_Contact center_MEAN","PREV_CHANNEL_TYPE_Country-wide_MEAN","PREV_CHANNEL_TYPE_Credit and cash offices_MEAN","PREV_CHANNEL_TYPE_Regional  Local_MEAN","PREV_CHANNEL_TYPE_Stone_MEAN","PREV_NAME_SELLER_INDUSTRY_Auto technology_MEAN","PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN","PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN","PREV_NAME_SELLER_INDUSTRY_Construction_MEAN","PREV_NAME_SELLER_INDUSTRY_Consumer electronics_MEAN","PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN","PREV_NAME_SELLER_INDUSTRY_Industry_MEAN","PREV_NAME_SELLER_INDUSTRY_Jewelry_MEAN","PREV_NAME_SELLER_INDUSTRY_MLM partners_MEAN","PREV_NAME_SELLER_INDUSTRY_Tourism_MEAN","PREV_NAME_SELLER_INDUSTRY_XNA_MEAN","PREV_NAME_YIELD_GROUP_XNA_MEAN","PREV_NAME_YIELD_GROUP_high_MEAN","PREV_NAME_YIELD_GROUP_low_action_MEAN","PREV_NAME_YIELD_GROUP_low_normal_MEAN","PREV_NAME_YIELD_GROUP_middle_MEAN","PREV_PRODUCT_COMBINATION_Card Street_MEAN","PREV_PRODUCT_COMBINATION_Card X-Sell_MEAN","PREV_PRODUCT_COMBINATION_Cash_MEAN","PREV_PRODUCT_COMBINATION_Cash Street high_MEAN","PREV_PRODUCT_COMBINATION_Cash Street low_MEAN","PREV_PRODUCT_COMBINATION_Cash Street middle_MEAN","PREV_PRODUCT_COMBINATION_Cash X-Sell high_MEAN","PREV_PRODUCT_COMBINATION_Cash X-Sell low_MEAN","PREV_PRODUCT_COMBINATION_Cash X-Sell middle_MEAN","PREV_PRODUCT_COMBINATION_POS household with interest_MEAN","PREV_PRODUCT_COMBINATION_POS household without interest_MEAN","PREV_PRODUCT_COMBINATION_POS industry with interest_MEAN","PREV_PRODUCT_COMBINATION_POS industry without interest_MEAN","PREV_PRODUCT_COMBINATION_POS mobile with interest_MEAN","PREV_PRODUCT_COMBINATION_POS mobile without interest_MEAN","PREV_PRODUCT_COMBINATION_POS other with interest_MEAN","PREV_PRODUCT_COMBINATION_POS others without interest_MEAN","PREV_PRODUCT_COMBINATION_nan_MEAN","APPROVED_AMT_ANNUITY_SCALED_MIN","APPROVED_AMT_ANNUITY_SCALED_MAX","APPROVED_AMT_ANNUITY_SCALED_MEAN","APPROVED_AMT_APPLICATION_SCALED_MIN","APPROVED_AMT_APPLICATION_SCALED_MAX","APPROVED_AMT_APPLICATION_SCALED_MEAN","APPROVED_AMT_CREDIT_SCALED_MIN","APPROVED_AMT_CREDIT_SCALED_MAX","APPROVED_AMT_CREDIT_SCALED_MEAN","APPROVED_APP_CREDIT_PERC_MIN","APPROVED_APP_CREDIT_PERC_MAX","APPROVED_APP_CREDIT_PERC_MEAN","APPROVED_AMT_GOODS_PRICE_SCALED_MIN","APPROVED_AMT_GOODS_PRICE_SCALED_MAX","APPROVED_AMT_GOODS_PRICE_SCALED_MEAN","APPROVED_HOUR_APPR_PROCESS_START_MIN","APPROVED_HOUR_APPR_PROCESS_START_MAX","APPROVED_HOUR_APPR_PROCESS_START_MEAN","APPROVED_YEARS_DECISION_MIN","APPROVED_YEARS_DECISION_MAX","APPROVED_YEARS_DECISION_MEAN","APPROVED_CNT_PAYMENT_MEAN","APPROVED_CNT_PAYMENT_SUM","POS_MONTHS_BALANCE_MAX","POS_MONTHS_BALANCE_MEAN","POS_MONTHS_BALANCE_SIZE","POS_SK_DPD_MAX","POS_SK_DPD_MEAN","POS_SK_DPD_DEF_MAX","POS_SK_DPD_DEF_MEAN","POS_NAME_CONTRACT_STATUS_Active_MEAN","POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN","POS_NAME_CONTRACT_STATUS_Approved_MEAN","POS_NAME_CONTRACT_STATUS_Canceled_MEAN","POS_NAME_CONTRACT_STATUS_Completed_MEAN","POS_NAME_CONTRACT_STATUS_Demand_MEAN","POS_NAME_CONTRACT_STATUS_Returned to the store_MEAN","POS_NAME_CONTRACT_STATUS_Signed_MEAN","POS_NAME_CONTRACT_STATUS_XNA_MEAN","POS_COUNT","INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE","INSTAL_DPD_MAX","INSTAL_DPD_MEAN","INSTAL_DPD_SUM","INSTAL_DBD_MAX","INSTAL_DBD_MEAN","INSTAL_DBD_SUM","INSTAL_PAYMENT_PERC_MAX","INSTAL_PAYMENT_PERC_MEAN","INSTAL_PAYMENT_PERC_SUM","INSTAL_PAYMENT_PERC_VAR","INSTAL_PAYMENT_DIFF_SCALED_MAX","INSTAL_PAYMENT_DIFF_SCALED_MEAN","INSTAL_PAYMENT_DIFF_SCALED_SUM","INSTAL_PAYMENT_DIFF_SCALED_VAR","INSTAL_AMT_INSTALMENT_SCALED_MAX","INSTAL_AMT_INSTALMENT_SCALED_MEAN","INSTAL_AMT_INSTALMENT_SCALED_SUM","INSTAL_AMT_PAYMENT_SCALED_MIN","INSTAL_AMT_PAYMENT_SCALED_MAX","INSTAL_AMT_PAYMENT_SCALED_MEAN","INSTAL_AMT_PAYMENT_SCALED_SUM","INSTAL_YEARS_ENTRY_PAYMENT_MAX","INSTAL_YEARS_ENTRY_PAYMENT_MEAN","INSTAL_YEARS_ENTRY_PAYMENT_SUM","INSTAL_COUNT"]
    cat_cols = ["CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","FLAG_EMP_PHONE","FLAG_WORK_PHONE","FLAG_CONT_MOBILE","FLAG_PHONE","FLAG_EMAIL","REG_REGION_NOT_LIVE_REGION","REG_REGION_NOT_WORK_REGION","LIVE_REGION_NOT_WORK_REGION","REG_CITY_NOT_LIVE_CITY","REG_CITY_NOT_WORK_CITY","LIVE_CITY_NOT_WORK_CITY","FLAG_DOCUMENT_2","FLAG_DOCUMENT_3","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_6","FLAG_DOCUMENT_7","FLAG_DOCUMENT_8","FLAG_DOCUMENT_9","FLAG_DOCUMENT_10","FLAG_DOCUMENT_11","FLAG_DOCUMENT_12","FLAG_DOCUMENT_13","FLAG_DOCUMENT_14","FLAG_DOCUMENT_15","FLAG_DOCUMENT_16","FLAG_DOCUMENT_17","FLAG_DOCUMENT_18","FLAG_DOCUMENT_19","FLAG_DOCUMENT_20","FLAG_DOCUMENT_21","NAME_CONTRACT_TYPE_Cash loans","NAME_CONTRACT_TYPE_Revolving loans","NAME_TYPE_SUITE_Children","NAME_TYPE_SUITE_Family","NAME_TYPE_SUITE_Group of people","NAME_TYPE_SUITE_Other_A","NAME_TYPE_SUITE_Other_B","NAME_TYPE_SUITE_Spouse partner","NAME_TYPE_SUITE_Unaccompanied","NAME_INCOME_TYPE_Businessman","NAME_INCOME_TYPE_Commercial associate","NAME_INCOME_TYPE_Maternity leave","NAME_INCOME_TYPE_Pensioner","NAME_INCOME_TYPE_State servant","NAME_INCOME_TYPE_Student","NAME_INCOME_TYPE_Unemployed","NAME_INCOME_TYPE_Working","NAME_EDUCATION_TYPE_Academic degree","NAME_EDUCATION_TYPE_Higher education","NAME_EDUCATION_TYPE_Incomplete higher","NAME_EDUCATION_TYPE_Lower secondary","NAME_EDUCATION_TYPE_Secondary  secondary special","NAME_FAMILY_STATUS_Civil marriage","NAME_FAMILY_STATUS_Married","NAME_FAMILY_STATUS_Separated","NAME_FAMILY_STATUS_Single  not married","NAME_FAMILY_STATUS_Unknown","NAME_FAMILY_STATUS_Widow","NAME_HOUSING_TYPE_Co-op apartment","NAME_HOUSING_TYPE_House  apartment","NAME_HOUSING_TYPE_Municipal apartment","NAME_HOUSING_TYPE_Office apartment","NAME_HOUSING_TYPE_Rented apartment","NAME_HOUSING_TYPE_With parents","OCCUPATION_TYPE_Accountants","OCCUPATION_TYPE_Cleaning staff","OCCUPATION_TYPE_Cooking staff","OCCUPATION_TYPE_Core staff","OCCUPATION_TYPE_Drivers","OCCUPATION_TYPE_HR staff","OCCUPATION_TYPE_High skill tech staff","OCCUPATION_TYPE_IT staff","OCCUPATION_TYPE_Laborers","OCCUPATION_TYPE_Low-skill Laborers","OCCUPATION_TYPE_Managers","OCCUPATION_TYPE_Medicine staff","OCCUPATION_TYPE_Private service staff","OCCUPATION_TYPE_Realty agents","OCCUPATION_TYPE_Sales staff","OCCUPATION_TYPE_Secretaries","OCCUPATION_TYPE_Security staff","OCCUPATION_TYPE_Waitersbarmen staff","WEEKDAY_APPR_PROCESS_START_FRIDAY","WEEKDAY_APPR_PROCESS_START_MONDAY","WEEKDAY_APPR_PROCESS_START_SATURDAY","WEEKDAY_APPR_PROCESS_START_SUNDAY","WEEKDAY_APPR_PROCESS_START_THURSDAY","WEEKDAY_APPR_PROCESS_START_TUESDAY","WEEKDAY_APPR_PROCESS_START_WEDNESDAY","ORGANIZATION_TYPE_Advertising","ORGANIZATION_TYPE_Agriculture","ORGANIZATION_TYPE_Bank","ORGANIZATION_TYPE_Business Entity Type 1","ORGANIZATION_TYPE_Business Entity Type 2","ORGANIZATION_TYPE_Business Entity Type 3","ORGANIZATION_TYPE_Cleaning","ORGANIZATION_TYPE_Construction","ORGANIZATION_TYPE_Culture","ORGANIZATION_TYPE_Electricity","ORGANIZATION_TYPE_Emergency","ORGANIZATION_TYPE_Government","ORGANIZATION_TYPE_Hotel","ORGANIZATION_TYPE_Housing","ORGANIZATION_TYPE_Industry type 1","ORGANIZATION_TYPE_Industry type 10","ORGANIZATION_TYPE_Industry type 11","ORGANIZATION_TYPE_Industry type 12","ORGANIZATION_TYPE_Industry type 13","ORGANIZATION_TYPE_Industry type 2","ORGANIZATION_TYPE_Industry type 3","ORGANIZATION_TYPE_Industry type 4","ORGANIZATION_TYPE_Industry type 5","ORGANIZATION_TYPE_Industry type 6","ORGANIZATION_TYPE_Industry type 7","ORGANIZATION_TYPE_Industry type 8","ORGANIZATION_TYPE_Industry type 9","ORGANIZATION_TYPE_Insurance","ORGANIZATION_TYPE_Kindergarten","ORGANIZATION_TYPE_Legal Services","ORGANIZATION_TYPE_Medicine","ORGANIZATION_TYPE_Military","ORGANIZATION_TYPE_Mobile","ORGANIZATION_TYPE_Other","ORGANIZATION_TYPE_Police","ORGANIZATION_TYPE_Postal","ORGANIZATION_TYPE_Realtor","ORGANIZATION_TYPE_Religion","ORGANIZATION_TYPE_Restaurant","ORGANIZATION_TYPE_School","ORGANIZATION_TYPE_Security","ORGANIZATION_TYPE_Security Ministries","ORGANIZATION_TYPE_Self-employed","ORGANIZATION_TYPE_Services","ORGANIZATION_TYPE_Telecom","ORGANIZATION_TYPE_Trade type 1","ORGANIZATION_TYPE_Trade type 2","ORGANIZATION_TYPE_Trade type 3","ORGANIZATION_TYPE_Trade type 4","ORGANIZATION_TYPE_Trade type 5","ORGANIZATION_TYPE_Trade type 6","ORGANIZATION_TYPE_Trade type 7","ORGANIZATION_TYPE_Transport type 1","ORGANIZATION_TYPE_Transport type 2","ORGANIZATION_TYPE_Transport type 3","ORGANIZATION_TYPE_Transport type 4","ORGANIZATION_TYPE_University","ORGANIZATION_TYPE_XNA","FONDKAPREMONT_MODE_not specified","FONDKAPREMONT_MODE_org spec account","FONDKAPREMONT_MODE_reg oper account","FONDKAPREMONT_MODE_reg oper spec account","HOUSETYPE_MODE_block of flats","HOUSETYPE_MODE_specific housing","HOUSETYPE_MODE_terraced house","WALLSMATERIAL_MODE_Block","WALLSMATERIAL_MODE_Mixed","WALLSMATERIAL_MODE_Monolithic","WALLSMATERIAL_MODE_Others","WALLSMATERIAL_MODE_Panel","WALLSMATERIAL_MODE_Stone brick","WALLSMATERIAL_MODE_Wooden","EMERGENCYSTATE_MODE_No","EMERGENCYSTATE_MODE_Yes","PREV_NAME_CONTRACT_TYPE_nan_MEAN","PREV_WEEKDAY_APPR_PROCESS_START_nan_MEAN","PREV_FLAG_LAST_APPL_PER_CONTRACT_nan_MEAN","PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN","PREV_NAME_CONTRACT_STATUS_nan_MEAN","PREV_NAME_PAYMENT_TYPE_nan_MEAN","PREV_CODE_REJECT_REASON_nan_MEAN","PREV_NAME_CLIENT_TYPE_nan_MEAN","PREV_NAME_GOODS_CATEGORY_House Construction_MEAN","PREV_NAME_GOODS_CATEGORY_nan_MEAN","PREV_NAME_PORTFOLIO_nan_MEAN","PREV_NAME_PRODUCT_TYPE_nan_MEAN","PREV_CHANNEL_TYPE_nan_MEAN","PREV_NAME_SELLER_INDUSTRY_nan_MEAN","PREV_NAME_YIELD_GROUP_nan_MEAN"]

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

def subsample_to_ratio(df: pd.DataFrame, target_col: str, target_ratio: float, max_samples=None, seed=None) -> pd.DataFrame:
    """Subsample the dataframe to achieve a target class ratio.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Name of the target column.
        target_ratio (float): Desired ratio of the positive class (0 < target_ratio < 1).
        max_samples (int, optional): Maximum number of samples in the subsampled dataframe. Defaults to None.
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

    sampled_neg_indx = np.random.choice(len(df_neg), size=n_neg_wanted, replace=False)
    df_neg_sampled = df_neg.iloc[sampled_neg_indx]

    if max_samples is not None and n_pos + n_neg_wanted > max_samples:
        # Sample each class separately so the target ratio is preserved
        n_pos_keep = min(max(1, int(round(max_samples * target_ratio))), n_pos)
        n_neg_keep = min(max_samples - n_pos_keep, n_neg_wanted)
        df_pos = df_pos.sample(n=n_pos_keep, random_state=seed)
        df_neg_sampled = df_neg_sampled.sample(n=n_neg_keep, random_state=seed)

    subsampled_df = pd.concat([df_pos, df_neg_sampled], axis=0)

    # Shuffle the subsampled dataframe to ensure random order of samples
    subsampled_df = subsampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return subsampled_df

def generate_parameters(param_grid: dict) -> list[dict]:
    """Generate all combinations of parameters from a parameter grid.

    Args:
        param_grid (dict): Dictionary where keys are parameter names and values are lists of parameter settings to try.

    Returns:
        list[dict]: A list of dictionaries, each representing a unique combination of parameters.
    """
    
    items = sorted(param_grid.items())

    if not items:
        return [{}]

    keys, values = zip(*items)
    return [dict(zip(keys, v)) for v in product(*values)]


