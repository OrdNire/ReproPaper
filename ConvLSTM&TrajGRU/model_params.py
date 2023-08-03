from collections import OrderedDict
from ConvLSTM import ConvLSTMCell
from TrajGRU import TrajGRU

# conv [input_channel, output_channel, kernel_size, stride, padding]
convlstm_encoder_params = [
    [
        OrderedDict({"conv1_leaky_1": [1, 16, 3, 1, 1]}),
        # Downsample
        OrderedDict({"conv2_leaky_1": [64, 64, 3, 2, 1]}),
        # Downsample
        OrderedDict({"conv3_leaky_1": [96, 96, 3, 2, 1]})
    ],
    [
        ConvLSTMCell(shape=(64, 64), input_channel=16, kernel_size=5, num_features=64),
        ConvLSTMCell(shape=(32, 32), input_channel=64, kernel_size=5, num_features=96),
        ConvLSTMCell(shape=(16, 16), input_channel=96, kernel_size=5, num_features=96)
    ]
]

convlstm_forecaster_params = [
    [
        # Upsample
        OrderedDict({"deconv1_leaky_1": [96, 96, 4, 2, 1]}),
        # Upsample
        OrderedDict({"deconv2_leaky_1": [96, 96, 4, 2, 1]}),
        OrderedDict({
            "conv3_leaky_1": [64, 16, 3, 1, 1],
            "conv4_leaky_1": [16, 1, 1, 1, 0] # reshape
        })
    ],
    [
        ConvLSTMCell(shape=(16, 16), input_channel=96, kernel_size=5, num_features=96), #3
        ConvLSTMCell(shape=(32, 32), input_channel=96, kernel_size=5, num_features=96), #2
        ConvLSTMCell(shape=(64, 64), input_channel=96, kernel_size=5, num_features=64)  #1
    ]
]

trajgru_encoder_params = [
    [
        OrderedDict({"conv1_leaky_1": [1, 16, 3, 1, 1]}),
        OrderedDict({"conv2_leaky_1": [64, 64, 3, 2, 1]}),
        OrderedDict({"conv3_leaky_1": [96, 96, 3, 2, 1]})
    ],
    [
        TrajGRU((64, 64), input_channel=16, num_features=64, kernel_size=3, num_links=5),
        TrajGRU((32, 32), input_channel=64, num_features=96, kernel_size=3, num_links=5),
        TrajGRU((16, 16), input_channel=96, num_features=96, kernel_size=3, num_links=5)
    ]
]

trajgru_forecaster_params = [
    [
        OrderedDict({"deconv1_leaky_1": [96, 96, 4, 2, 1]}),
        OrderedDict({"deconv2_leaky_1": [96, 96, 4, 2, 1]}),
        OrderedDict({
            "conv3_leaky_1": [64, 16, 3, 1, 1],
            "conv4_leaky_1": [16, 1, 1, 1, 0]  # reshape
        })
    ],
    [
        TrajGRU((16, 16), input_channel=96, num_features=96, kernel_size=3, num_links=5),
        TrajGRU((32, 32), input_channel=96, num_features=96, kernel_size=3, num_links=5),
        TrajGRU((64, 64), input_channel=96, num_features=64, kernel_size=3, num_links=5)
    ]
]