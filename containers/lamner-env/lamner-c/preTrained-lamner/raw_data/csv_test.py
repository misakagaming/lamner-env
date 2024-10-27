import pandas as pd
for namee in ["test", "train", "validation"]:
    parquet_file = namee + "-00000-of-00001.parquet"
    df = pd.read_parquet(parquet_file)
    if namee == "validation":
        csv_file = "valid.csv"
    else:
        csv_file = namee + ".csv"
    df.to_csv(csv_file)