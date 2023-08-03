import pandas as pd
import config
import collections

continuous_features = range(1, 14)
categorial_features = range(14, 40)

# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
continuous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]

class CategoryDictGenerator:
    def __init__(self, num_features):
        self.num_features = num_features
        self.dicts = []
        for i in range(num_features):
            self.dicts.append(collections.defaultdict(int))

    def build(self, filename, categorial_features, cutoff=0):
        # mapping dicts
        with open(filename, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i, idx in enumerate(categorial_features):
                    if features[idx] != '':
                        self.dicts[i][features[idx]] += 1

        for i in range(0, self.num_features):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]["<unk>"] = 0

    def gen(self, val, idx):
        if val not in self.dicts[idx]:
            return self.dicts[idx]["<unk>"]
        return self.dicts[idx][val]

    def dict_size(self):
        return [len(d) for d in self.dicts]
    
class ContinuousDictGenerator:
    def __init__(self, continuous_clip):
        self.continuous_clip = continuous_clip

    def gen(self, val, idx):
        if val == '':
            return 0.0
        val = float(val)
        if val >= self.continuous_clip[idx]:
            return float(self.continuous_clip[idx])

        return val
    


if __name__ == '__main__':
    training_data = pd.read_csv(config.TRAINING_PATH, sep='\t', header=None)
    training_data = training_data.fillna(0.0)

    # category generator
    categoryDictGenerator = CategoryDictGenerator(len(categorial_features))
    categoryDictGenerator.build(config.TRAINING_PATH, categorial_features, cutoff=5)
    # continuous generator
    continuousDictGenerator = ContinuousDictGenerator(continuous_clip)

    # save category features' size
    dict_size = categoryDictGenerator.dict_size()
    with open(config.HANDLE_FEATURES_SIZE_PATH, 'w') as output:
        sizes = [1] * len(continuous_features) + dict_size
        sizes = [str(i) for i in sizes]
        output.write(','.join(sizes))

    # handle raw training data
    with open(config.HANDLE_TRAINING_PATH, 'w') as output:
        with open(config.TRAINING_PATH, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                # handle continuous features
                for i, idx in enumerate(continuous_features):
                    features[idx] = continuousDictGenerator.gen(features[idx], i)
                    features[idx] = str(features[idx])

                # handle categorial features
                for i, idx in enumerate(categorial_features):
                    features[idx] = categoryDictGenerator.gen(features[idx], i)
                    features[idx] = str(features[idx])

                label = list(features[0])
                features = features[1:] + label

                output.write(','.join(features) + '\n')

    # handle raw testing data
    with open(config.HANDLE_TESTING_PATH, 'w') as output:
        with open(config.TESTING_PATH, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                # handle continuous features
                for i, idx in enumerate(continuous_features):
                    idx -= 1
                    features[idx] = continuousDictGenerator.gen(features[idx], i)
                    features[idx] = str(features[idx])

                # handle categorial features
                for i, idx in enumerate(categorial_features):
                    idx -= 1
                    features[idx] = categoryDictGenerator.gen(features[idx], i)
                    features[idx] = str(features[idx])

                output.write(','.join(features) + '\n')