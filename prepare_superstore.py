import pandas as pd
df = pd.read_csv("train.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)   # dd/mm/yyyy
monthly = (df.groupby(df["Order Date"].dt.to_period("M").dt.to_timestamp())["Sales"]
             .sum().reset_index().rename(columns={"Order Date": "Date"}))
monthly.to_csv("monthly_sales.csv", index=False)
print(monthly.shape, monthly["Date"].min().date(), "->", monthly["Date"].max().date())
