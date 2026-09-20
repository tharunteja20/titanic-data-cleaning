# Task 1: Data Cleaning & Preparation — Titanic Passenger Dataset

## Dataset
- **Source:** Titanic passenger manifest — a well-known public dataset used for data cleaning/ML practice.
- **Original file:** https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
- **Raw copy used here:** titanic_raw.csv (250-row extract of the classic 891-row Titanic dataset)
- **Tool used:** Python (pandas)

## What Was Wrong With the Raw Data
- Missing values: Age missing in ~19% of rows, Cabin missing in ~81%, Embarked missing in 1 row.
- Duplicate records: checked for full-row duplicates and duplicate PassengerIds — none found in this extract.
- Incorrect data types: Survived, Pclass, Sex, Embarked stored as plain numbers/strings instead of categorical variables.
- Inconsistent values: stray whitespace in a Name entry; a Fare of 0 flagged; port codes and sex labels needed standardized casing.

## Cleaning Steps Performed (see clean_titanic.py)
1. Trimmed whitespace from all text columns.
2. Removed exact duplicate rows (none present, but check is built in).
3. Imputed missing Age using median Age within each Pclass + Sex group.
4. Imputed missing Embarked with the mode (most frequent port).
5. Handled missing Cabin by adding a boolean column HasCabinRecord and filling blanks with "Unknown".
6. Converted Survived, Pclass, Sex, Embarked to proper categorical dtypes.
7. Standardized formatting: lowercased Sex, uppercased Embarked codes.
8. Removed any rows with negative Fare.

## Files in This Submission
- titanic_raw.csv — original, unmodified dataset extract
- clean_titanic.py — the cleaning script (fully reproducible)
- titanic_cleaned.csv — the cleaned output dataset
- cleaning_report.txt — auto-generated log of every issue found and fix applied
- README.md — this file

## How to Reproduce
pip install pandas
python clean_titanic.py
