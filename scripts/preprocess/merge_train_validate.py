import pandas as pd

train = pd.read_csv("logp_wo_averaging_train.csv")
valid = pd.read_csv("logp_wo_averaging_validation.csv")

sum_data = pd.concat([train, valid])
sum_data.to_csv("logp_wo_averaging_cross.csv", index=False)