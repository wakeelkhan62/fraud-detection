import pandas as pd

tx = pd.read_csv("data/raw/train_transaction.csv")
idn = pd.read_csv("data/raw/train_identity.csv")

print("Transactions shape:", tx.shape)
print("Identity shape:", idn.shape)
print("Fraud rate:", round(tx["isFraud"].mean(), 4))
print("Memory (MB):", round(tx.memory_usage(deep=True).sum() / 1e6, 1))
print(tx[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]].head())