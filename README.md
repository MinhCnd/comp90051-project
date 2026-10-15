# comp90051-project

# Install dependencies
call `uv sync`

# Run scripts from virtual environment
call `uv run python .....` 

# Data gathering
Download dataset from `https://www.kaggle.com/competitions/home-credit-default-risk/data` and store in folder on local 

# Data Preprocessing
To generate preprocessed dataframe call `uv run python -m src.preprocess ` with required arguments, note for this experiment drop_col_thres should be set to 0.5

The output should be a dataset with 34,000+ rows. We stratified subsample this further down to 10,000 rows by calling `src.pipeline.subsample_to_ratio(df,target_col='TARGET', target_ratio=0.08, max_samples=10000)`, this can be done via Jupyter Notebook or script.

# Model training
## Transformer
Training transformer model with preset hyperparameters require a machine with min 50GB RAM and 50GB GPU mem, Google Colab's A-100 High RAM server was used. To train:
1. Start the Google Colab notebook environment
2. Upload all files from src to environment
3. Upload csv training data to environment
4. Call `pip install pytorch-tabular`
5. Call `python -m py_train -f <path to dataset> -r <sub-sampling ratio>`
6. Get comfortable, it will be a 6-7 hours wait
Alternatively you can train this on a High-RAM, GPU-enabled cloud server(AWS, Google, Azure)

## XGBoost and Logistic Regression
These models were able to be trained on a macboook air. Just call `uv run python -m pt_train -f <path to dataset> -r <sub-sampling ratio> -m <model choice, simple for log res, medium for XGBoost>`

# Corrected Resampled t-test
The t-tests ran through notebooks/t_test.R script. You need to change file paths at the start of the file and run the script