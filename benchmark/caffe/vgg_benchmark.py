# %matplotlib inline
from matplotlib import pyplot
import numpy as np
import os
import shutil
import time

from caffe2.python import core, cnn, net_drawer, workspace, visualize

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
# set this where the root of caffe2 is installed
caffe2_root = "~/caffe2"
print("Necessities imported!")

current_folder = os.getcwd()

def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label
print("Input function created.")

def AddLeNetModel(model, data):
    conv1_1 = model.Conv(data, 'conv1_1', dim_in=3, dim_out=64, kernel=3, pad=1)
    conv1_1 = model.Relu(conv1_1, conv1_1)
    conv1_2 = model.Conv(conv1_1, 'conv1_2', dim_in=64, dim_out=64, kernel=3, pad=1)
    conv1_2 = model.Relu(conv1_2, conv1_2)
    pool1 = model.MaxPool(conv1_2, 'pool1', kernel=2, stride=2)
    # 224 x 224 -> 112 x 112

    conv2_1 = model.Conv(pool1, 'conv2_1', dim_in=64, dim_out=128, kernel=3, pad=1)
    conv2_1 = model.Relu(conv2_1, conv2_1)
    conv2_2 = model.Conv(conv2_1, 'conv2_2', dim_in=128, dim_out=128, kernel=3, pad=1)
    conv2_2 = model.Relu(conv2_2, conv2_2)
    pool2 = model.MaxPool(conv2_2, 'pool2', kernel=2, stride=2)
    # 112 x 112 -> 56 x 56

    conv3_1 = model.Conv(pool2, 'conv3_1', dim_in=128, dim_out=256, kernel=3, pad=1)
    conv3_1 = model.Relu(conv3_1, conv3_1)
    conv3_2 = model.Conv(conv3_1, 'conv3_2', dim_in=256, dim_out=256, kernel=3, pad=1)
    conv3_2 = model.Relu(conv3_2, conv3_2)
    conv3_3 = model.Conv(conv3_2, 'conv3_3', dim_in=256, dim_out=256, kernel=3, pad=1)
    conv3_3 = model.Relu(conv3_3, conv3_3)
    conv3_4 = model.Conv(conv3_3, 'conv3_4', dim_in=256, dim_out=256, kernel=3, pad=1)
    conv3_4 = model.Relu(conv3_4, conv3_4)
    pool3 = model.MaxPool(conv3_4, 'pool3', kernel=2, stride=2)
    # 56 x 56 -> 28 x 28

    conv4_1 = model.Conv(pool3, 'conv4_1', dim_in=256, dim_out=512, kernel=3, pad=1)
    conv4_1 = model.Relu(conv4_1, conv4_1)
    conv4_2 = model.Conv(conv4_1, 'conv4_2', dim_in=512, dim_out=512, kernel=3, pad=1)
    conv4_2 = model.Relu(conv4_2, conv4_2)
    conv4_3 = model.Conv(conv4_2, 'conv4_3', dim_in=512, dim_out=512, kernel=3, pad=1)
    conv4_3 = model.Relu(conv4_3, conv4_3)
    conv4_4 = model.Conv(conv4_3, 'conv4_4', dim_in=512, dim_out=512, kernel=3, pad=1)
    conv4_4 = model.Relu(conv4_4, conv4_4)
    pool4 = model.MaxPool(conv4_4, 'pool4', kernel=2, stride=2)
    # 28 x 28 -> 14 x 14

    conv5_1 = model.Conv(pool4, 'conv5_1', dim_in=512, dim_out=512, kernel=3, pad=1)
    conv5_1 = model.Relu(conv5_1, conv5_1)
    conv5_2 = model.Conv(conv5_1, 'conv5_2', dim_in=512, dim_out=512, kernel=3, pad=1)
    conv5_2 = model.Relu(conv5_2, conv5_2)
    conv5_3 = model.Conv(conv5_2, 'conv5_3', dim_in=512, dim_out=512, kernel=3, pad=1)
    conv5_3 = model.Relu(conv5_3, conv5_3)
    conv5_4 = model.Conv(conv5_3, 'conv5_4', dim_in=512, dim_out=512, kernel=3, pad=1)
    conv5_4 = model.Relu(conv5_4, conv5_4)
    pool5 = model.MaxPool(conv5_4, 'pool5', kernel=2, stride=2)
    # 14 x 14 -> 7 x 7

    fc6 = model.FC(pool5, 'fc6', dim_in = 7 * 7 * 512, dim_out=4096)
    fc6 = model.Relu(fc6, fc6)
    fc7 = model.FC(fc6, 'fc7', dim_in = 4096, dim_out=4096)
    fc7 = model.Relu(fc7, fc7)
    fc8 = model.FC(fc7, 'fc8', dim_in=4096, dim_out=1000)

    prob = model.Relu(fc8, fc8)

print("Model function created.")

train_model = cnn.CNNModelHelper(order="NCHW", name="vgg_train")

data, label = AddInput(
    train_model, batch_size=10,
    db= '/caffe/train',
    db_type='lmdb')
softmax = AddLeNetModel(train_model, data)

print('Created training model.')

# The parameter initialization network only needs to be run once.
workspace.RunNetOnce(train_model.param_init_net)
# creating the network
workspace.CreateNet(train_model.net)
# set the number of iterations and track the accuracy & loss
total_iters = 3

start = time.time()
print 'Start'
for i in range(total_iters):
    st = time.time()
    workspace.RunNet(train_model.net.Proto().name)
    e = time.time()
    print i + 1,': iteration time {}'.format(e - st)

end = time.time()
print('Time: {}'.format(end - start))
