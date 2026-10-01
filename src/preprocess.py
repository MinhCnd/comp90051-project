from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import gc
import time
import logging
import argparse
from contextlib import contextmanager
pd.set_option('display.max_columns', None)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Source: https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features

# `DAYS_EMPLOYED` contains an extreme outlier value: **365243** that Home Credit uses as a placeholder for retired, unemployed, or missing records).
PLACEHOLDER = 365243
DAYS_PER_YEAR = 365.25

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    logger.info("{} - done in {:.0f}s".format(title, time.time() - t0))

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if (df[col].dtype == 'object' or df[col].dtype == 'str')]
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category, dtype=int)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def application_train(folder_path: Path, num_rows=None, nan_as_category=False):
    df = pd.read_csv(folder_path / 'application_train.csv', nrows= num_rows)
    logger.info("Train samples: {}".format(len(df)))

    # Remove applications with XNA CODE_GENDER
    df = df[df['CODE_GENDER'] != 'XNA']

    # Remove FLAGMOBIL feature
    df = df.drop(columns=["FLAG_MOBIL"], errors="ignore")

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    days_employed = df['DAYS_EMPLOYED'].replace(PLACEHOLDER, np.nan)
    
    # Create new features using concat to avoid dataframe fragmentation
    new_time_cols = pd.DataFrame(
        {
            'YEARS_EMPLOYED': days_employed / -DAYS_PER_YEAR,
            'AGE_YEARS': df['DAYS_BIRTH'] / -DAYS_PER_YEAR,
            'YEARS_REGISTRATION': df['DAYS_REGISTRATION'] / -DAYS_PER_YEAR,
            'YEARS_ID_PUBLISH': df['DAYS_ID_PUBLISH'] / -DAYS_PER_YEAR,
            'YEARS_LAST_PHONE_CHANGE': df['DAYS_LAST_PHONE_CHANGE'] / -DAYS_PER_YEAR,
        },
        index=df.index,
    )
    df = pd.concat([df.drop(columns=['DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE'], errors='ignore'), new_time_cols], axis=1).copy()

    #AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY exhibit extreme range, apply natural log scaling
    features_to_scale = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    scaled_cols = pd.DataFrame(
        {f'{feature}_SCALED': np.log1p(df[feature]) for feature in features_to_scale},
        index=df.index,
    )
    df = pd.concat([df, scaled_cols], axis=1).copy()

    # Some simple new features (percentages)
    new_cols = pd.DataFrame(
        {
            'DAYS_EMPLOYED_PERC': days_employed / df['DAYS_BIRTH'],
            'INCOME_CREDIT_PERC': df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT'],
            'INCOME_PER_PERSON_SCALED': np.log1p(df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']),
            'ANNUITY_INCOME_PERC': df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'],
            'PAYMENT_RATE': df['AMT_ANNUITY'] / df['AMT_CREDIT'],
        },
        index=df.index,
    )

    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df]
    if ext:
        new_cols = pd.concat(
            [
                new_cols,
                pd.DataFrame(
                    {
                        "EXT_SOURCE_MEAN": df[ext].mean(axis=1),  # consensus external score
                        "EXT_SOURCE_STD": df[ext].std(axis=1),  # disagreement between sources
                    },
                    index=df.index,
                ),
            ],
            axis=1,
        )

    df = pd.concat([df, new_cols], axis=1).copy()

    #Drop unscaled features
    df = df.drop(columns=['DAYS_BIRTH'])
    df = df.drop(columns=[f'{feature}' for feature in features_to_scale], errors='ignore')

    gc.collect()
    return df

def bureau_and_balance(folder_path: Path, num_rows=None, nan_as_category=False):
    bureau = pd.read_csv(folder_path / 'bureau.csv', nrows = num_rows)

    # Rename Loan Type
    CREDIT_TYPE_NAME_MAP = {
    "Another type of loan": "Other",
    "Car loan": "Car",
    "Cash loan (non-earmarked)": "Cash",
    "Consumer credit": "Consumer Credit",
    "Credit card": "Credit Card",
    "Interbank credit": "Interbank credit",
    "Loan for business development": "Business Development",
    "Loan for purchase of shares (margin lending)": "Shares",
    "Loan for the purchase of equipment": "Equipment",
    "Loan for working capital replenishment": "Working Capital",
    "Microloan": "Microloan",
    "Mobile operator loan": "Mobile Operator",
    "Mortgage": "Mortgage",
    "Real estate loan": "Real Estate",
    "Unknown type of loan": "Unknown",
    }
    bureau["CREDIT_TYPE"] = bureau["CREDIT_TYPE"].map(CREDIT_TYPE_NAME_MAP)

    bb = pd.read_csv(folder_path / 'bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    del bb, bb_agg
    gc.collect()

    num_aggregations = {
        'SK_ID_BUREAU': ['count'],
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }

    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

def previous_applications(folder_path: Path, num_rows = None, nan_as_category = True):
    prev = pd.read_csv(f'{folder_path}/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'] = prev['DAYS_FIRST_DRAWING'].replace(PLACEHOLDER, np.nan)
    prev['DAYS_FIRST_DUE'] = prev['DAYS_FIRST_DUE'].replace(PLACEHOLDER, np.nan)
    prev['DAYS_LAST_DUE_1ST_VERSION'] = prev['DAYS_LAST_DUE_1ST_VERSION'].replace(PLACEHOLDER, np.nan)
    prev['DAYS_LAST_DUE'] = prev['DAYS_LAST_DUE'].replace(PLACEHOLDER, np.nan)
    prev['DAYS_TERMINATION'] = prev['DAYS_TERMINATION'].replace(PLACEHOLDER, np.nan)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    prev['DAYS_DECISION'] = prev['DAYS_DECISION'].replace(PLACEHOLDER, np.nan)
    prev['YEARS_DECISION'] = prev['DAYS_DECISION'] / -DAYS_PER_YEAR
    prev=prev.drop(['DAYS_DECISION'], axis=1)

    #ln Scale features
    features_to_scale = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE']

    scaled_cols = pd.DataFrame(
        {f'{feature}_SCALED': np.log1p(prev[feature]) for feature in features_to_scale},
        index=prev.index,
    )
    prev = pd.concat([prev, scaled_cols], axis=1).copy()
    #Remove unscaled features
    prev = prev.drop(columns=features_to_scale)
    
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY_SCALED': ['min', 'max', 'mean'],
        'AMT_APPLICATION_SCALED': ['min', 'max', 'mean'],
        'AMT_CREDIT_SCALED': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT_SCALED': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE_SCALED': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'YEARS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    

    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(folder_path: Path, num_rows = None, nan_as_category = False):
    pos = pd.read_csv(folder_path / 'POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

def installments_payments(folder_path: Path, num_rows = None, nan_as_category = False):
    ins = pd.read_csv(folder_path / 'installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    ins["PAY_RATIO"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    #Scale relevant features
    features_to_scale = ['AMT_INSTALMENT', 'AMT_PAYMENT']
    scaled_cols = pd.DataFrame(
        {f'{feature}_SCALED': np.log1p(ins[feature]) for feature in features_to_scale},
        index=ins.index,
    )
    #Use signed log transform for PAYMENT_DIFF
    ins['PAYMENT_DIFF_SCALED'] = ins['PAYMENT_DIFF'].apply(lambda x: np.sign(x) * np.log1p(abs(x)))
    ins = pd.concat([ins, scaled_cols], axis=1).copy()

    #Change days to years
    ins['YEARS_ENTRY_PAYMENT'] = ins['DAYS_ENTRY_PAYMENT'] / -DAYS_PER_YEAR
    ins = ins.drop(columns=['DAYS_ENTRY_PAYMENT'])
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF_SCALED': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT_SCALED': ['max', 'mean', 'sum'],
        'AMT_PAYMENT_SCALED': ['min', 'max', 'mean', 'sum'],
        'YEARS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()

    return ins_agg

def credit_card_balance(folder_path: Path, num_rows = None, nan_as_category = False):
    cc = pd.read_csv(folder_path / 'credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

def preprocess(folder_path: Path, num_rows=None, debug=False, drop_col_thres: float = 0.10, output_path: Optional[str] = None) -> pd.DataFrame:
    """Preprocess the data for machine learning models.
    
    Args:
        folder_path (Path): The path to the data folder.
        num_rows (int, optional): The number of rows to read from each file. Defaults to None.
        debug (bool): Whether to run in debug mode. Defaults to False.
        drop_col_thres (float): The percentage threshold for dropping columns with too many missing values. Defaults to 0.10.
        output_path (str, optional): The path to save the processed data. Defaults to None.
    
    Return:
        pd.DataFrame: The preprocessed DataFrame.
    """
    num_rows = 500 if debug else num_rows

    if drop_col_thres > 1 or drop_col_thres < 0:
        raise ValueError("drop_col_thres must be between 0 and 1")

    df = application_train(folder_path, num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(folder_path, num_rows)
        logger.info(f"Bureau df shape: {bureau.shape}")
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(folder_path, num_rows)
        logger.info(f"Previous applications df shape: {prev.shape}")
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(folder_path, num_rows)
        logger.info(f"Pos-cash balance df shape: {pos.shape}")
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(folder_path, num_rows)
        logger.info(f"Installments payments df shape: {ins.shape}")
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(folder_path, num_rows)
        logger.info(f"Credit card balance df shape: {cc.shape}")
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()

    #Dropping columns with too many missing
    logger.info(f"df before dropping columns: {df.shape}")
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > drop_col_thres].index.tolist()
    cols_to_drop = [col for col in cols_to_drop if col != 'TARGET']

    logger.info(f"Dropping {len(cols_to_drop)} columns with more than {drop_col_thres:.0%} missing values")
    df = df.drop(columns=cols_to_drop)

    #Drop rows with any missing cols
    logger.info(f"df before dropping rows: {df.shape}")
    df = df.dropna(axis=0, how='any')
    logger.info(f"df after dropping rows: {df.shape}")

    # Diagnostic prints
    logger.info(f"Updated df shape: {df.shape}")
    target_ratio = (df['TARGET'] == 1).sum() / len(df)
    logger.info(f"TARGET class ratio: {target_ratio:.2%}")

    # Count and percentage of rows with missing data
    rows_with_missing = df.isnull().any(axis=1).sum()
    pct_rows_with_missing = (rows_with_missing / len(df)) * 100
    logger.debug(f"Rows with missing data: {rows_with_missing} ({pct_rows_with_missing:.2f}%)")
    # 5 features with most missing data, and how many in percentage
    missing_counts = df.isnull().sum().sort_values(ascending=False)
    missing_percentages = (missing_counts / len(df)) * 100
    
    logger.debug("Top 5 features with most missing data:")
    for col, pct in missing_percentages.head(5).items():
        logger.debug(f"  {col}: {pct:.2f}%")

    #Rename columns with any json special characters
    df.columns = df.columns.str.replace('[\\\/\,\:]', '', regex=True)

    #Remove index column
    df = df.drop('SK_ID_CURR', axis=1)

    if output_path:
        df.to_csv(output_path, index=False)

    return df

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-f", "--folder_path", type=Path, default=Path("data"))
    argparser.add_argument("-t", "--drop_col_thres", type=float, default=0.10)
    argparser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    
    argparser.add_argument("-o", "--output_path", type=str, default="processed_data.csv")
    args = argparser.parse_args()

    preprocess(args.folder_path, debug=args.debug, drop_col_thres=args.drop_col_thres, output_path=args.output_path)