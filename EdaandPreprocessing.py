import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("default")
sns.set_theme(style="whitegrid")

# 1. SETTINGS AND DATA LOADING

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "cardio_labels_fixed.csv"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "cardio"
RANDOM_STATE = 42

# Set these according to your dataset and assignment.
SCALING_METHOD = "standard"  
HANDLE_OUTLIERS = False     
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Treat common placeholders as missing values.
df = df.replace(
    r"^\s*(\?|unknown|n/?a|null|none)\s*$",
    np.nan,
    regex=True
)

# Convert known numerical columns to numeric values when present.
numeric_candidates = [
    "age", "height", "weight", "ap_hi", "ap_lo",
    "cardio", "cholesterol", "gluc", "smoke",
    "alco", "active"
]

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' is missing. "
        f"Available columns: {df.columns.tolist()}"
    )

print("=" * 65)
print("HEART DISEASE DATASET: EDA AND PREPROCESSING")
print("=" * 65)

# 2. DATASET DIMENSIONS AND STRUCTURE

print("\n1. DATASET DIMENSIONS")
print("-" * 65)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nRandom five records:")
print(df.sample(min(5, len(df)), random_state=RANDOM_STATE))

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

print("\nNumerical descriptive statistics:")
print(df.describe().T)

print("\nAll-column descriptive statistics:")
print(df.describe(include="all").T)

print("\nInterpretation:")
print(
    "Dimensions show the dataset size. Descriptive statistics show "
    "central tendency, spread, minimum, maximum, and category counts."
)

# 3. MISSING-VALUE ANALYSIS

print("\n2. MISSING-VALUE ANALYSIS")
print("-" * 65)

missing_report = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (df.isnull().mean() * 100).round(2)
})

missing_report = missing_report.sort_values(
    "Missing_Percentage", ascending=False
)

print(missing_report)

missing_to_plot = missing_report[
    missing_report["Missing_Count"] > 0
]

if not missing_to_plot.empty:
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=missing_to_plot.index,
        y=missing_to_plot["Missing_Count"]
    )
    plt.title("Missing Values by Feature")
    plt.xlabel("Feature")
    plt.ylabel("Missing Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    print(
        "Interpretation: taller bars indicate columns requiring "
        "more attention during missing-value treatment."
    )
else:
    print("No missing values found.")
    print("Interpretation: no missing-value treatment is needed currently.")

# 4. DUPLICATE ANALYSIS

print("\n3. DUPLICATE ANALYSIS")
print("-" * 65)

duplicate_count = df.duplicated().sum()

print("Exact duplicate records:", duplicate_count)

plt.figure(figsize=(6, 4))
sns.barplot(
    x=["Unique rows", "Duplicate rows"],
    y=[len(df) - duplicate_count, duplicate_count]
)
plt.title("Unique vs Duplicate Records")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

print(
    "Interpretation: exact duplicates may represent repeated records. "
    "Check the dataset documentation before removing them."
)

# 5. NUMERICAL FEATURE DISTRIBUTIONS

print("\n4. NUMERICAL FEATURE DISTRIBUTIONS")
print("-" * 65)

numeric_features = [
    col for col in [
        "age", "height", "weight", "ap_hi", "ap_lo"
    ]
    if col in df.columns
]

for col in numeric_features:
    values = df[col].dropna()

    if values.empty:
        continue

    print(f"\n{col}")
    print("Mean:", round(values.mean(), 2))
    print("Median:", round(values.median(), 2))
    print("Standard deviation:", round(values.std(), 2))
    print("Minimum:", round(values.min(), 2))
    print("Maximum:", round(values.max(), 2))
    print("Skewness:", round(values.skew(), 2))

    plt.figure(figsize=(8, 4))
    sns.histplot(values, bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    if abs(values.skew()) < 0.5:
        print("Interpretation: distribution is approximately symmetric.")
    elif values.skew() > 0.5:
        print("Interpretation: distribution is positively skewed.")
    else:
        print("Interpretation: distribution is negatively skewed.")

# 6. CATEGORICAL FEATURE DISTRIBUTIONS

print("\n5. CATEGORICAL FEATURE DISTRIBUTIONS")
print("-" * 65)

categorical_candidates = [
    "gender", "cholesterol", "gluc",
    "smoke", "alco", "active"
]

categorical_features = [
    col for col in categorical_candidates if col in df.columns
]

for col in categorical_features:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts(dropna=False))

    plot_data = df[col].astype("string").fillna("Missing")

    plt.figure(figsize=(7, 4))
    sns.countplot(x=plot_data, order=plot_data.value_counts().index)
    plt.title(f"Category Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    print(
        "Interpretation: compare bar heights to identify common "
        "and uncommon categories and possible data-quality issues."
    )

# 7. TARGET DISTRIBUTION AND CLASS IMBALANCE

print("\n6. TARGET DISTRIBUTION")
print("-" * 65)

print("Target values, including missing values:")
print(df[TARGET].value_counts(dropna=False))

valid_target = df[TARGET].dropna()

print("\nTarget percentages:")
print((valid_target.value_counts(normalize=True) * 100).round(2))

plt.figure(figsize=(7, 4))
sns.countplot(x=df[TARGET].astype("string"))
plt.title("Heart Disease Target Distribution")
plt.xlabel("Cardio (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

if not valid_target.empty:
    percentages = valid_target.value_counts(normalize=True) * 100

    if len(percentages) >= 2:
        print(
            "Majority class percentage:",
            round(percentages.max(), 2), "%"
        )
        print(
            "Minority class percentage:",
            round(percentages.min(), 2), "%"
        )
        print(
            "Majority-to-minority ratio:",
            round(percentages.max() / percentages.min(), 2)
        )

        if percentages.max() - percentages.min() > 10:
            print(
                "Interpretation: class imbalance may be present. "
                "Evaluate its effect before choosing a balancing method."
            )
        else:
            print(
                "Interpretation: classes are relatively balanced "
                "under the 10-percentage-point heuristic."
            )
    else:
        print(
            "Interpretation: only one valid class was found. "
            "Check target labels and data quality."
        )

# 8. OUTLIER DETECTION USING IQR

print("\n7. OUTLIER DETECTION")
print("-" * 65)

outlier_summary = []

for col in numeric_features:
    values = df[col].dropna()

    if values.empty:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (values < lower_bound) | (values > upper_bound)
    outlier_count = int(outlier_mask.sum())

    outlier_summary.append({
        "Feature": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": outlier_count,
        "Outlier_Percentage": outlier_mask.mean() * 100
    })

    print(f"\nFeature: {col}")
    print("Lower bound:", round(lower_bound, 2))
    print("Upper bound:", round(upper_bound, 2))
    print("Potential outliers:", outlier_count)
    print("Outlier percentage:", round(outlier_mask.mean() * 100, 2), "%")

    plt.figure(figsize=(8, 3))
    sns.boxplot(x=values)
    plt.title(f"Box Plot: {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

    print(
        "Interpretation: points beyond the whiskers are potential "
        "outliers, not automatically incorrect observations."
    )

outlier_report = pd.DataFrame(outlier_summary)
outlier_report.to_csv(OUTPUT_DIR / "outlier_report.csv", index=False)

# 9. CORRELATION ANALYSIS

print("\n8. CORRELATION ANALYSIS")
print("-" * 65)

correlation_data = df.select_dtypes(include=np.number)
correlation_matrix = correlation_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.3
)
plt.title("Numerical Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

if TARGET in correlation_matrix.columns:
    print("\nCorrelation with target:")
    print(
        correlation_matrix[TARGET]
        .sort_values(ascending=False)
        .round(3)
    )

print(
    "Interpretation: values near +1 indicate positive linear "
    "association; values near -1 indicate negative linear association; "
    "values near 0 indicate weak linear association. Correlation "
    "does not establish causation."
)

# 10. FEATURE RELATIONSHIPS WITH TARGET

print("\n9. FEATURE-TARGET RELATIONSHIPS")
print("-" * 65)

# Keep only valid binary target values.
valid_df = df[df[TARGET].isin([0, 1])].copy()

print("Rows available for relationship analysis:", len(valid_df))
print("Target distribution:")
print(valid_df[TARGET].value_counts(dropna=False))

if valid_df.empty:
    print("ERROR: No valid target records are available.")
else:

    # Numerical features versus target
    for col in numeric_features:
        if col not in valid_df.columns:
            continue

        plot_df = valid_df[[TARGET, col]].dropna()

        if plot_df.empty:
            print(f"Skipping {col}: no usable values.")
            continue

        plt.figure(figsize=(7, 4))
        sns.boxplot(data=plot_df, x=TARGET, y=col)
        plt.title(f"{col} by Heart Disease Status")
        plt.xlabel("Cardio (0 = No Disease, 1 = Disease)")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

        print(f"\nMean {col} by target:")
        print(plot_df.groupby(TARGET)[col].mean().round(2))

    # Categorical features versus target
    for col in categorical_features:
        if col not in valid_df.columns:
            continue

        plot_df = valid_df[[col, TARGET]].dropna()

        if plot_df.empty:
            print(f"Skipping {col}: no usable values.")
            continue

        relationship = pd.crosstab(
            plot_df[col],
            plot_df[TARGET],
            normalize="index"
        ) * 100

        # Avoid plotting an empty table.
        if relationship.empty or relationship.shape[1] == 0:
            print(
                f"Skipping {col}: no valid feature-target "
                "relationships were found."
            )
            continue

        print(f"\nPercentage of target classes within {col}:")
        print(relationship.round(2))

        relationship.plot(
            kind="bar",
            figsize=(8, 4)
        )

        plt.title(f"Heart Disease Percentage by {col}")
        plt.xlabel(col)
        plt.ylabel("Percentage within Category")
        plt.legend(title=TARGET)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()


    print(
        "Interpretation: compare disease percentages across categories. "
        "Small groups may produce unstable percentages."
    )

# 11. CLEANING: DUPLICATES AND INVALID TARGETS

print("\n10. BASIC DATA CLEANING")
print("-" * 65)

rows_before = len(df)

# Remove exact duplicate rows.
df = df.drop_duplicates().copy()

# Exclude records without a valid binary target.
missing_target_count = int(df[TARGET].isna().sum())
invalid_target_count = int((~df[TARGET].isin([0, 1]) & df[TARGET].notna()).sum())

df = df[df[TARGET].isin([0, 1])].copy()
df[TARGET] = df[TARGET].astype(int)

print("Rows before cleaning:", rows_before)
print("Rows after cleaning:", len(df))
print("Total rows removed:", rows_before - len(df))
print("Missing-target rows before removal:", missing_target_count)
print("Invalid-target rows before removal:", invalid_target_count)
print("Remaining exact duplicates:", df.duplicated().sum())

if df.empty or df[TARGET].nunique() < 2:
    raise ValueError(
        "After cleaning, both target classes must be present."
    )

# 12. CHECK OBVIOUSLY INVALID MEASUREMENTS

print("\n11. DATA VALIDITY CHECKS")
print("-" * 65)

# Flag values for investigation; do not assume every extreme
# medical measurement is an error.
validity_checks = {
    "height": lambda s: s.le(0),
    "weight": lambda s: s.le(0),
    "age": lambda s: s.le(0),
    "ap_hi": lambda s: s.le(0),
    "ap_lo": lambda s: s.le(0)
}

for col, check in validity_checks.items():
    if col in df.columns:
        invalid_mask = df[col].notna() & check(df[col])
        print(f"{col}: {int(invalid_mask.sum())} non-positive values")

        # Treat impossible non-positive measurements as missing.
        # Positive but unusual measurements remain for investigation.
        df.loc[invalid_mask, col] = np.nan
        
# 13. FEATURE ENGINEERING

print("\n12. FEATURE ENGINEERING")
print("-" * 65)

if {"ap_hi", "ap_lo"}.issubset(df.columns):
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    print("Created pulse_pressure = ap_hi - ap_lo")

    print(df["pulse_pressure"].describe())

    plt.figure(figsize=(8, 4))
    sns.histplot(df["pulse_pressure"].dropna(), bins=30, kde=True)
    plt.title("Pulse Pressure Distribution")
    plt.xlabel("Pulse Pressure")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    print(
        "Interpretation: pulse pressure represents the difference "
        "between systolic and diastolic blood pressure. Check its "
        "validity and usefulness before including it in a model."
    )

# 14. MISSING-VALUE PREPROCESSING

print("\n13. MISSING-VALUE PREPROCESSING")
print("-" * 65)

target_values = df[TARGET].copy()

categorical_cols = [
    col for col in df.columns
    if col != TARGET and (
        df[col].dtype == "object"
        or str(df[col].dtype) in ["category", "string", "bool"]
        or col in categorical_candidates
    )
]

numeric_cols = [
    col for col in df.columns
    if col != TARGET and col not in categorical_cols
    and pd.api.types.is_numeric_dtype(df[col])
]

print("Numerical columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

imputation_report = []

for col in numeric_cols:
    missing_before = int(df[col].isna().sum())

    if missing_before > 0:
        median_value = df[col].median()

        if pd.notna(median_value):
            df[col] = df[col].fillna(median_value)

            imputation_report.append({
                "Feature": col,
                "Strategy": "Median",
                "Missing_Filled": missing_before
            })

            print(
                f"{col}: filled {missing_before} missing values "
                f"using median {median_value:.2f}"
            )

for col in categorical_cols:
    missing_before = int(df[col].isna().sum())

    if missing_before > 0:
        mode_values = df[col].mode(dropna=True)

        if not mode_values.empty:
            mode_value = mode_values.iloc[0]
            df[col] = df[col].fillna(mode_value)

            imputation_report.append({
                "Feature": col,
                "Strategy": "Most frequent category",
                "Missing_Filled": missing_before
            })

            print(
                f"{col}: filled {missing_before} missing values "
                f"using most frequent category {mode_value!r}"
            )

imputation_df = pd.DataFrame(imputation_report)
imputation_df.to_csv(
    OUTPUT_DIR / "imputation_report.csv",
    index=False
)

# Restore the target from the cleaned target values.
df[TARGET] = target_values.astype(int)

print("\nRemaining missing values:")
print(df.isnull().sum())

print(
    "\nInterpretation: median imputation is less sensitive to extreme "
    "values than mean imputation. Most-frequent imputation is a simple "
    "choice for categorical variables. These choices should be justified "
    "using the missingness pattern and dataset documentation."
)

# 15. OPTIONAL OUTLIER CAPPING

print("\n14. OUTLIER TREATMENT")
print("-" * 65)

if HANDLE_OUTLIERS:
    print(
        "Capping enabled. This uses full-dataset IQR bounds and is "
        "for exploratory preprocessing only, not leakage-safe evaluation."
    )

    for col in numeric_cols:
        if col not in df.columns or df[col].dropna().empty:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df[col] = df[col].clip(lower_bound, upper_bound)

        print(
            f"{col}: capped values outside "
            f"[{lower_bound:.2f}, {upper_bound:.2f}]"
        )
else:
    print(
        "No outliers automatically capped. This is deliberate: "
        "extreme clinical measurements should be investigated rather "
        "than removed or capped without domain justification."
    )

# 16. ENCODING AND SCALING PLAN

print("\n15. ENCODING AND SCALING")
print("-" * 65)

print(
    "Categorical columns identified for encoding:",
    categorical_cols
)
print(
    "Numerical columns identified for scaling:",
    numeric_cols
)

print(
    "\nEncoding recommendation: use one-hot encoding for nominal "
    "categories, unless the dataset documentation confirms a meaningful "
    "ordinal order. For example, cholesterol levels 1, 2, 3 are ordinal "
    "if they represent normal, above-normal, and well-above-normal."
)

print(
    "\nScaling recommendation: StandardScaler centers features and "
    "scales them to unit variance; MinMaxScaler maps features to a "
    "specified range, usually 0 to 1. Scaling is useful for many "
    "distance-based and gradient-based algorithms but is not necessary "
    "for every tree-based algorithm."
)

print(
    "\nNo scaler or encoder is fitted here. Without a train-test split, "
    "fitting learned preprocessing on all rows can leak information "
    "into later evaluation. Fit these transformations on training data "
    "only in the future modeling pipeline."
)

# 17. FEATURE SELECTION PLAN

print("\n16. FEATURE SELECTION")
print("-" * 65)

excluded_features = [
    col for col in [
        "id", "exam_date", "hospital_code", "bp_reading"
    ]
    if col in df.columns
]

print("Potential identifier/non-predictive columns:", excluded_features)

print(
    "Do not automatically drop a column just because it has a weak "
    "correlation with the target. Review its meaning, missingness, "
    "redundancy, and possible target leakage first."
)

print(
    "Supervised feature selection (such as SelectKBest) should be fitted "
    "on training data only after the future split."
)

# 18. CLASS IMBALANCE HANDLING PLAN

print("\n17. CLASS IMBALANCE HANDLING")
print("-" * 65)

print("Target counts:")
print(df[TARGET].value_counts())

print("Target percentages:")
print((df[TARGET].value_counts(normalize=True) * 100).round(2))

print(
    "No resampling is performed. If imbalance is substantial, compare "
    "class weights and resampling methods later. Any oversampling or "
    "undersampling must be applied to training data only, never to "
    "the complete dataset before evaluation."
)

# 19. FINAL DATA QUALITY CHECK

print("\n18. FINAL DATA QUALITY CHECK")
print("-" * 65)

print("Final dimensions:", df.shape)
print("Remaining duplicates:", df.duplicated().sum())
print("Remaining missing values:")
print(df.isnull().sum())

print("\nFinal target distribution:")
print(df[TARGET].value_counts())

# 20. SAVE OUTPUTS

clean_path = OUTPUT_DIR / "cardio_clean.csv"
df.to_csv(clean_path, index=False)

missing_path = OUTPUT_DIR / "missing_values_report.csv"
missing_report.to_csv(missing_path)

summary_path = OUTPUT_DIR / "descriptive_statistics.csv"
df.describe(include="all").T.to_csv(summary_path)

correlation_path = OUTPUT_DIR / "correlation_matrix.csv"
correlation_matrix = df.select_dtypes(include=np.number).corr()
correlation_matrix.to_csv(correlation_path)

print("\n" + "=" * 65)
print("EDA AND PREPROCESSING COMPLETED")
print("=" * 65)

print("Cleaned dataset:", clean_path)
print("Missing-value report:", missing_path)
print("Descriptive statistics:", summary_path)
print("Correlation matrix:", correlation_path)
print("Outlier report:", OUTPUT_DIR / "outlier_report.csv")
print("Imputation report:", OUTPUT_DIR / "imputation_report.csv")