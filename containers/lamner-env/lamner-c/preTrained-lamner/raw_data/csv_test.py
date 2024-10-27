import pandas as pd

count = {
    "test":9000,
    "train":70000,
    "validation"9000:
}


for namee in ["test", "train", "validation"]:
    parquet_file = namee + "-00000-of-00001.parquet"
    df = pd.read_parquet(parquet_file)
    df = df.head(count[namee])
    if namee == "validation":
        csv_file = "valid.csv"
    else:
        csv_file = namee + ".csv"
    df.to_csv(csv_file, index=False)