# datasets path
DATA_PATH = "./datasets/"

TRAINING_PATH = DATA_PATH + "train.txt"
TESTING_PATH = DATA_PATH + "test.txt"

HANDLE_TRAINING_PATH = DATA_PATH + "handle_train.txt"
HANDLE_TESTING_PATH = DATA_PATH + "handle_test.txt"
HANDLE_FEATURES_SIZE_PATH = DATA_PATH + "features_size.txt"

CONTINUOUS_FEATURES = range(0, 13)
CATEGORIAL_FEATURES = range(13, 39)

# MODEL PATH
MODEL_PATH = "./models/"

DEEPFM_MODEL = MODEL_PATH + "deepFM.pth"