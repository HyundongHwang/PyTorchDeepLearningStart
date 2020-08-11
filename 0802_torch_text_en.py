import myutil as mu
import os.path

################################################################################
# 


import urllib.request
import pandas as pd

if not os.path.isfile(".IMDb_Reviews.csv"):
    urllib.request.urlretrieve(
        url="https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv",
        filename=".IMDb_Reviews.csv")

df = pd.read_csv(".IMDb_Reviews.csv", encoding="latin1")

mu.log("len(df)", len(df))
mu.log("df[:5]", df[:5])

train_df = df[:2500]
test_df = df[2500:]

train_df.to_csv(".train_data.csv", index=False)
test_df.to_csv(".test_data.csv", index=False)

################################################################################
# 

from torchtext import data

TEXT = torchtext.data.Field(
    sequential=True,
    use_vocab=True,
    tokenize=str.split,
    lower=True,
    batch_first=True,
    fix_length=20)

LABEL = torchtext.data.Field(
    sequential=False,
    use_vocab=False,
    batch_first=False,
    is_target=True
)

################################################################################
#

from torchtext.data import TabularDataset

train_data, test_data = TabularDataset.splits(
    path=".",
    train="train_data.csv",
    test="test_data.csv",
    format="csv",
    fields=[("text", TEXT), ("label", LABEL)],
    skip_header=True
)

mu.log("len(train_data)", len(train_data))
mu.log("len(test_data)", len(test_data))
