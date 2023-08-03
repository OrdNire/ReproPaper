import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import errno
from PIL import Image

class MovingMNIST(Dataset):

    # 存放原始数据以及分裂下载后的数据
    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "moving_mnist_train.pt"
    test_file = "moving_mnist_test.pt"
    # root/processed_folder/training_file
    # root/processed_folder/test_file
    # root/raw_folder/

    urls= ["https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz"]

    def __init__(self, root, train=True, split=1000, transform=None, target_transform=None, download=True, Norm=False):
        '''

        :param root:              存放数据根目录
        :param train:             训练集或者测试集
        :param split:             训练集和测试集分裂比例
        :param transform:         input transform
        :param target_transform:  target transform
        :param download:
        '''
        super(MovingMNIST, self).__init__()

        self.root = os.path.expanduser(root)
        self.train = train
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.Norm = Norm

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found! you can use Download=True.")

        if self.train:
            # train_data (Num, B, T, H, W)
            self.train_data = torch.load(os.path.join(self.root, self.processed_folder, self.training_file))
            self.train_data = self.train_data.unsqueeze(2)
            if self.Norm:
                self.train_data = (self.train_data / 255.0).contiguous().float()
        else:
            self.test_data = torch.load(os.path.join(self.root, self.processed_folder, self.test_file))
            self.test_data = self.test_data.unsqueeze(2)
            if self.Norm:
                self.test_data = (self.test_data / 255.0).contiguous().float()


    # data (B, T, H, W)
    def __getitem__(self, index):

        def _tranform_time(transform, data):
            new_data = None
            for i in range(data.size(0)):
                img = Image.fromarray(data[i].numpy(), mode='L')
                new_data = transform(img) if new_data is None else torch.cat([new_data, transform(img)], dim=0)
            return new_data

        if self.train:
            seq, target = self.train_data[index, :10], self.train_data[index, 10:]
        else:
            seq, target = self.test_data[index, :10], self.test_data[index, 10:]

        if self.transform is not None:
            seq = _tranform_time(self.transform, seq)
        if self.target_transform is not None:
            target = _tranform_time(self.target_transform, target)

        return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file))) and \
               (os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file)))

    def download(self):
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # 创建目录 root/raw_folder and root/processed
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # 下载文件
        for url in self.urls:
            print("Download {}".format(url))
            data = urllib.request.urlopen(url)
            file_name = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, file_name)
            with open(file_path, "wb") as f:
                f.write(data.read())
            with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        print("Processing....")

        '''
            mnist_test_seq.npy   (20, 10000. 64, 64) (T, B, W, H) -> (B, T, W, H)
        '''
        train_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, "mnist_test_seq.npy")).swapaxes(0, 1)[:-self.split]
        )
        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, "mnist_test_seq.npy")).swapaxes(0, 1)[-self.split:]
        )

        with open(os.path.join(self.root, self.processed_folder, self.training_file), "wb") as f:
            torch.save(train_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")

if __name__ == '__main__':
    train_set = MovingMNIST("datasets/mnist")
    train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    # seq (B, T, C, H, W)
    for seq, target in train_loader:
        print(seq.shape, target.shape)
        print(seq, target)
        break