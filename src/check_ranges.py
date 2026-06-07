import pandas as pd

df = pd.read_csv("data/final/final_model_dataset.csv")

print("=== SLIDER RANGES FOR UI ===")
print(f"volt      : {df['volt'].min():.2f}  to  {df['volt'].max():.2f}")
print(f"rotate    : {df['rotate'].min():.2f}  to  {df['rotate'].max():.2f}")
print(f"pressure  : {df['pressure'].min():.2f}  to  {df['pressure'].max():.2f}")
print(f"vibration : {df['vibration'].min():.2f}  to  {df['vibration'].max():.2f}")
print(f"age       : {df['age'].min()}  to  {df['age'].max()}")

print()
print("=== KEY VALUES FOR prepare_input() ===")
print(f"max_rotate (for production_load) : {df['rotate'].max():.6f}")

print()
print("=== ENCODINGS ===")
print("model_encoded values      :", sorted(df["model_encoded"].unique()))
print("age_category_encoded vals :", sorted(df["age_category_encoded"].unique()))

# Cross-check what model_encoded corresponds to original model column
# Load engineered dataset which still has model column
eng = pd.read_csv("data/processed/engineered_dataset.csv")
mapping = eng[["model","age_category"]].drop_duplicates().sort_values("model")

# Rebuild encodings same way pipeline did
eng["model_encoded"] = eng["model"].astype("category").cat.codes
eng["age_category_encoded"] = eng["age_category"].astype("category").cat.codes

print()
print("=== MODEL LABEL ENCODING ===")
model_map = eng[["model","model_encoded"]].drop_duplicates().sort_values("model_encoded")
print(model_map.to_string(index=False))

print()
print("=== AGE CATEGORY ENCODING ===")
age_map = eng[["age_category","age_category_encoded"]].drop_duplicates().sort_values("age_category_encoded")
print(age_map.to_string(index=False))
