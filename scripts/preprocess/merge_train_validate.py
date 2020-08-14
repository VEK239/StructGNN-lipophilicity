import pandas as pd

DATA_PATH = "../../data/3_final_data/split_data/"

train = pd.read_csv(DATA_PATH + "logp_wo_averaging_train.csv")
valid = pd.read_csv(DATA_PATH + "logp_wo_averaging_validation.csv")

sum_data = pd.concat([train, valid])
sum_data.to_csv(DATA_PATH + "logp_wo_averaging_cross.csv", index=False)
