# dataset name
DATASET = "pinterest-20"
assert DATASET in ["ml-1m", "pinterest-20"]

# datasets path
DATA_PATH = "./datasets/"

# data path
TEST_NEGATIVE = DATA_PATH + "{}.test.negative".format(DATASET)
TEST_RATING = DATA_PATH + "{}.test.rating".format(DATASET)
TRAIN_RATING = DATA_PATH + "{}.train.rating".format(DATASET)

# MODEL PATH
MODEL_PATH = "./models/"

AMF_MODEL = MODEL_PATH + "AMF.pth"