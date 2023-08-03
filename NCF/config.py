# dataset name
DATASET = "ml-1m"
assert DATASET in ["ml-1m", "pinterest-20"]

# model name
MODEL = "NeuMF-pre"
assert MODEL in ["GMF", "MLP", "NeuMF-end", "NeuMF-pre"]

# datasets path
DATA_PATH = "./datasets/"

# data path
TEST_NEGATIVE = DATA_PATH + "{}.test.negative".format(DATASET)
TEST_RATING = DATA_PATH + "{}.test.rating".format(DATASET)
TRAIN_RATING = DATA_PATH + "{}.train.rating".format(DATASET)

# MODEL PATH
MODEL_PATH = "./models/"

GMF_MODEL = MODEL_PATH + "GMF.pth"
MLP_MODEL = MODEL_PATH + "MLP.pth"
NeuMF_MODEL = MODEL_PATH + "NeuMF.pth"
