from fanblower_app.feature_extractor import build_feature_row

df = build_feature_row([
    r"data\raw\100A1.csv",
    r"data\raw\100H1.csv",
    r"data\raw\100V1.csv",
])

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nValues:")
print(df.iloc[0])