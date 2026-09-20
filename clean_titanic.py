"""
Data Cleaning & Preparation - Titanic Passenger Dataset
Task 1: Clean and prepare a raw public dataset for analysis.

Source: Titanic dataset (public, widely used for data-cleaning/ML practice)
Original source URL: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
"""

import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 1. LOAD RAW DATA
# ----------------------------------------------------------------------
df = pd.read_csv("titanic_raw.csv")
print("=== RAW DATA OVERVIEW ===")
print(f"Shape: {df.shape}")
print(df.dtypes)
print()

report_lines = []
report_lines.append("DATA CLEANING REPORT")
report_lines.append("=" * 50)
report_lines.append(f"Raw dataset shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

# ----------------------------------------------------------------------
# 2. IDENTIFY MISSING VALUES
# ----------------------------------------------------------------------
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_report = missing_report[missing_report["missing_count"] > 0]
print("=== MISSING VALUES ===")
print(missing_report)
report_lines.append("1. MISSING VALUES FOUND")
report_lines.append(missing_report.to_string())
report_lines.append("")

# ----------------------------------------------------------------------
# 3. IDENTIFY DUPLICATE RECORDS
# ----------------------------------------------------------------------
dup_count_full = df.duplicated().sum()
dup_count_id = df.duplicated(subset=["PassengerId"]).sum()
print(f"\n=== DUPLICATES ===\nFull-row duplicates: {dup_count_full}")
print(f"Duplicate PassengerId values: {dup_count_id}")
report_lines.append("2. DUPLICATE RECORDS")
report_lines.append(f"Full-row duplicates found: {dup_count_full}")
report_lines.append(f"Duplicate PassengerId values found: {dup_count_id}")
report_lines.append("")

# ----------------------------------------------------------------------
# 4. IDENTIFY INCORRECT / SUBOPTIMAL DATA TYPES
# ----------------------------------------------------------------------
report_lines.append("3. DATA TYPE ISSUES IDENTIFIED")
report_lines.append("- Survived, Pclass, Sex, Embarked were stored as generic int64/object")
report_lines.append("  but are categorical variables, not free numbers/text.")
report_lines.append("- Age was stored as float64 purely because of missing values mixed")
report_lines.append("  with whole-number ages; cleaned version keeps float but rounds sensibly.")
report_lines.append("- Fare and Age had no explicit range validation (e.g. negative/zero fares).")
report_lines.append("")

# ----------------------------------------------------------------------
# 5. IDENTIFY INCONSISTENT VALUES
# ----------------------------------------------------------------------
report_lines.append("4. INCONSISTENT / MESSY VALUES IDENTIFIED")
# Whitespace issues in Name
ws_issues = df["Name"].apply(lambda x: isinstance(x, str) and x != x.strip()).sum()
report_lines.append(f"- Names with leading/trailing whitespace: {ws_issues}")
# Sex casing consistency check
sex_variants = df["Sex"].dropna().str.lower().unique()
report_lines.append(f"- Unique Sex categories (after lowercasing): {list(sex_variants)}")
# Embarked variants
emb_variants = df["Embarked"].dropna().unique()
report_lines.append(f"- Unique Embarked codes: {list(emb_variants)} (S=Southampton, C=Cherbourg, Q=Queenstown)")
# Fare of 0 (likely crew/comp tickets, worth flagging not necessarily removing)
zero_fare = (df["Fare"] == 0).sum()
report_lines.append(f"- Rows with Fare == 0 (flagged as suspicious, kept but noted): {zero_fare}")
report_lines.append("")

# ----------------------------------------------------------------------
# 6. CLEANING STEPS
# ----------------------------------------------------------------------
clean = df.copy()

# 6a. Trim whitespace in text columns
for col in ["Name", "Sex", "Ticket", "Cabin", "Embarked"]:
    clean[col] = clean[col].astype("string").str.strip()

# 6b. Remove exact duplicate rows (if any)
before = len(clean)
clean = clean.drop_duplicates()
after_dupe_drop = len(clean)

# 6c. Handle missing values
# Age: impute with median age per Pclass+Sex group (more accurate than a single global median)
clean["Age"] = clean.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
# Any remaining (edge-case) NaNs -> overall median
clean["Age"] = clean["Age"].fillna(clean["Age"].median())
clean["Age"] = clean["Age"].round(1)

# Embarked: fill with the mode (most frequent port)
if clean["Embarked"].isnull().any():
    mode_port = clean["Embarked"].mode(dropna=True)[0]
    clean["Embarked"] = clean["Embarked"].fillna(mode_port)

# Cabin: mostly missing (structural missingness = passenger had no recorded cabin).
# Create a boolean flag instead of imputing a fake cabin value.
clean["HasCabinRecord"] = clean["Cabin"].notna()
clean["Cabin"] = clean["Cabin"].fillna("Unknown")

# 6d. Fix data types -> convert categorical columns properly
clean["Survived"] = clean["Survived"].astype("category")
clean["Pclass"] = clean["Pclass"].astype("category")
clean["Sex"] = clean["Sex"].str.lower().astype("category")
clean["Embarked"] = clean["Embarked"].str.upper().astype("category")

# 6e. Standardize inconsistent values
# Ensure Fare is non-negative
clean = clean[clean["Fare"] >= 0]

# ----------------------------------------------------------------------
# 7. SAVE CLEANED DATA
# ----------------------------------------------------------------------
clean.to_csv("titanic_cleaned.csv", index=False)

report_lines.append("5. CLEANING ACTIONS TAKEN")
report_lines.append(f"- Trimmed whitespace in text columns (Name, Sex, Ticket, Cabin, Embarked).")
report_lines.append(f"- Dropped exact duplicate rows: {before - after_dupe_drop} removed.")
report_lines.append(f"- Imputed missing Age using median Age within each Pclass+Sex group.")
report_lines.append(f"- Imputed {missing_report.loc['Embarked','missing_count'] if 'Embarked' in missing_report.index else 0} missing Embarked values with the mode (most common port).")
report_lines.append(f"- Cabin: created new boolean column 'HasCabinRecord'; filled missing Cabin with 'Unknown' "
                     f"instead of guessing a fake cabin (missingness here is meaningful, not random).")
report_lines.append(f"- Converted Survived, Pclass, Sex, Embarked to proper categorical dtypes.")
report_lines.append(f"- Standardized Sex to lowercase and Embarked to uppercase for consistency.")
report_lines.append(f"- Filtered out any rows with negative Fare (data validation).")
report_lines.append("")
report_lines.append(f"Final cleaned dataset shape: {clean.shape[0]} rows x {clean.shape[1]} columns")

with open("cleaning_report.txt", "w") as f:
    f.write("\n".join(str(x) for x in report_lines))

print("\n=== CLEANED DATA OVERVIEW ===")
print(f"Shape: {clean.shape}")
print(clean.dtypes)
print("\nMissing values after cleaning:")
print(clean.isnull().sum())
print("\nSaved: titanic_cleaned.csv, cleaning_report.txt")
