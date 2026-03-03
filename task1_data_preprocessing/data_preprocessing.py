import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# loading and inspecting the data
print("Loading and Inspecting the Dataset")

df = pd.read_csv("dataset/churn-bigml-80.csv")

print(f"\nDataset Shape: {df.shape}")
print(f"  Rows   : {df.shape[0]}")
print(f"  Columns: {df.shape[1]}")

print("\nColumn Data Types")
print(df.dtypes)

print("\nFirst 5 Rows")
print(df.head())

print("\nStatistical Summary")
print(df.describe())

# handling the missing data
print("\nHandling Missing Data")

missing_before = df.isnull().sum()
total_missing = missing_before.sum()

print("\nMissing Values Per Column (Before)")
print(missing_before)
print(f"\nTotal missing values: {total_missing}")

if total_missing > 0:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled '{col}' missing values with median = {median_val}")

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  Filled '{col}' missing values with mode = '{mode_val}'")

    threshold = len(df.columns) * 0.5
    rows_before = len(df)
    df.dropna(thresh=int(threshold), inplace=True)
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(f"\n  Dropped {rows_dropped} rows with >50% missing values")

    missing_after = df.isnull().sum().sum()
    print(f"\nMissing Values After Handling: {missing_after}")
else:
    print("\nNo missing values found in the dataset!")
    print("  (Demonstrating the check is still important in a real pipeline)")

# encoding categorical variables
print("\nEncoding Categorical Variables")

print("\nCategorical Columns")
categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
print(categorical_cols)

print("\nLabel Encoding (Binary Columns)")
label_encoder = LabelEncoder()

binary_cols = ["International plan", "Voice mail plan", "Churn"]
for col in binary_cols:
    df[col] = label_encoder.fit_transform(df[col])
    print(f"  '{col}' -> Encoded as: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

print("\nOne-Hot Encoding (Multi-class Columns)")
print(f"  'State' has {df['State'].nunique()} unique values")

df = pd.get_dummies(df, columns=["State"], prefix="State", drop_first=True)
print(f"  After one-hot encoding: {df.shape[1]} total columns")

print("\nEncoded Dataset Sample (first 5 rows, first 15 cols)")
print(df.iloc[:5, :15])

# normalizing / standardizing numerical features
print("\nNormalizing / Standardizing Numerical Features")

target_col = "Churn"
original_numerical = [
    "Account length", "Area code", "Number vmail messages",
    "Total day minutes", "Total day calls", "Total day charge",
    "Total eve minutes", "Total eve calls", "Total eve charge",
    "Total night minutes", "Total night calls", "Total night charge",
    "Total intl minutes", "Total intl calls", "Total intl charge",
    "Customer service calls"
]

print(f"\nNumerical features to standardize: {len(original_numerical)}")
print(original_numerical)

scaler = StandardScaler()

print("\nBefore Scaling")
print(df[original_numerical].head(3))

df[original_numerical] = scaler.fit_transform(df[original_numerical])

print("\nAfter Scaling")
print(df[original_numerical].head(3))

print(f"\nStandardization applied -- Mean ~ 0, Std ~ 1")
print(f"  Example: '{original_numerical[0]}' -> mean={df[original_numerical[0]].mean():.4f}, std={df[original_numerical[0]].std():.4f}")

# splitting into training and testing sets
print("\nSplitting into Training and Testing Sets")

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Total samples : {len(df)}")
print(f"  Training set  : X_train={X_train.shape}, y_train={y_train.shape}")
print(f"  Testing set   : X_test={X_test.shape},  y_test={y_test.shape}")
print(f"  Train ratio   : {len(X_train)/len(df)*100:.1f}%")
print(f"  Test ratio    : {len(X_test)/len(df)*100:.1f}%")

print("\nTarget Class Distribution")
print(f"  Training set: {dict(y_train.value_counts())}")
print(f"  Testing set : {dict(y_test.value_counts())}")

# saving the processed data
print("\nSaving Processed Datasets")

os.makedirs("processed_dataset", exist_ok=True)

X_train.to_csv("processed_dataset/X_train.csv", index=False)
X_test.to_csv("processed_dataset/X_test.csv", index=False)
y_train.to_csv("processed_dataset/y_train.csv", index=False)
y_test.to_csv("processed_dataset/y_test.csv", index=False)
df.to_csv("processed_dataset/churn_preprocessed_full.csv", index=False)

print("  Saved the following files in the 'processed_dataset' folder:")
print("  - processed_dataset/X_train.csv")
print("  - processed_dataset/X_test.csv")
print("  - processed_dataset/y_train.csv")
print("  - processed_dataset/y_test.csv")
print("  - processed_dataset/churn_preprocessed_full.csv")
print(f"""
Summary:
  Dataset         : churn-bigml-80.csv
  Original shape  : (2667, 20)
  Processed shape : {df.shape}
  Missing data    : Handled (median/mode imputation)
  Encoding        : Label (binary) + One-Hot (State)
  Scaling         : StandardScaler on {len(original_numerical)} numerical features
  Train/Test split: 80/20 with stratification

  Features (X)    : {X.shape[1]} columns
  Target (y)      : '{target_col}' (0 = No Churn, 1 = Churn)
""")
