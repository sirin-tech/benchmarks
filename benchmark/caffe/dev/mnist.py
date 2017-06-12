# %matplotlib inline
from matplotlib import pyplot
import numpy as np
import os
import shutil
import time
import utils

from caffe2.python import core, cnn, net_drawer, workspace, visualize

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
# set this where the root of caffe2 is installed
caffe2_root = "~/caffe2"
print("Necessities imported!")

# This section preps your image and test set in a leveldb
# if you didn't download the dataset yet go back to Models and Datasets and get it there
current_folder = os.getcwd()

data_folder = os.path.join(current_folder, 'mnist')
root_folder = os.path.join(current_folder, 'tutorial_mnist')
image_file_train = os.path.join(data_folder, "train-images-idx3-ubyte")
label_file_train = os.path.join(data_folder, "train-labels-idx1-ubyte")
image_file_test = os.path.join(data_folder, "t10k-images-idx3-ubyte")
label_file_test = os.path.join(data_folder, "t10k-labels-idx1-ubyte")

# Get the dataset if it is missing
#def DownloadDataset(url, path):
#    import requests, zipfile, StringIO
#    print "Downloading... ", url, " to ", path
#    r = requests.get(url, stream=True)
#    z = zipfile.ZipFile(StringIO.StringIO(r.content))
#    z.extractall(path)
#if not os.path.exists(data_folder):
#    os.makedirs(data_folder)
#if not os.path.exists(label_file_train):
#    DownloadDataset("https://s3.amazonaws.com/caffe2/datasets/mnist/mnist.zip", data_folder)
#
#def GenerateDB(image, label, name):
#    name = os.path.join(data_folder, name)
#    print 'DB name: ', name
#    syscall = "/usr/local/binaries/make_mnist_db --channel_first --db leveldb --image_file " + image + " --label_file " + label + " --output_file " + name
#    print "Creating database with: ", syscall
#    os.system(syscall)
#
## (Re)generate the leveldb database (known to get corrupted...)
#GenerateDB(image_file_train, label_file_train, "mnist-train-nchw-leveldb")
#GenerateDB(image_file_test, label_file_test, "mnist-test-nchw-leveldb")
#
#if os.path.exists(root_folder):
#    print("Looks like you ran this before, so we need to cleanup those old workspace files...")
#    shutil.rmtree(root_folder)
#
#os.makedirs(root_folder)
#workspace.ResetWorkspace(root_folder)
#
#print("training data folder:"+data_folder)
#print("workspace root folder:"+root_folder)

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
    # fc6 = model.Dropout(fc6, fc6)
    fc7 = model.FC(fc6, 'fc7', dim_in = 4096, dim_out=4096)
    fc7 = model.Relu(fc7, fc7)
    # fc7 = model.Dropout(fc7, fc7)
    fc8 = model.FC(fc7, 'fc8', dim_in=4096, dim_out=1000)

    prob = model.Relu(fc8, fc8)

    # conv1 = model.Conv(data, 'conv1', 1, 20, 5)
    # pool1 = model.MaxPool(conv1, 'pool1', kernel=2, stride=2)
    # conv2 = model.Conv(pool1, 'conv2', 20, 50, 5)
    # pool2 = model.MaxPool(conv2, 'pool2', kernel=2, stride=2)
    # fc3 = model.FC(pool2, 'fc3', 50 * 4 * 4, 500)
    # fc3 = model.Relu(fc3, fc3)
    # pred = model.FC(fc3, 'pred', 500, 10)
    # softmax = model.Relu(fc3, fc3) # model.Softmax(pred, 'softmax')
    # return softmax
print("Model function created.")

def AddAccuracy(model, softmax, label):
    accuracy = model.Accuracy([softmax, label], "accuracy")
    return accuracy
print("Accuracy function created.")

def AddTrainingOperators(model, softmax, label):
    # something very important happens here
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = model.Iter("iter")
    # set the learning rate schedule
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - CNNModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)
    # let's checkpoint every 20 iterations, which should probably be fine.
    # you may need to delete tutorial_files/tutorial-mnist to re-run the tutorial
    model.Checkpoint([ITER] + model.params, [],
                   db="vgg_lenet_checkpoint_%05d.leveldb",
                   db_type="leveldb", every=20)
print("Training function created.")

def AddBookkeepingOperators(model):
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
    # Now, if we really want to be verbose, we can summarize EVERY blob
    # that the model produces; it is probably not a good idea, because that
    # is going to take time - summarization do not come for free. For this
    # demo, we will only show how to summarize the parameters and their
    # gradients.
print("Bookkeeping function created")

train_model = cnn.CNNModelHelper(order="NCHW", name="vgg_train")

data, label = AddInput(
    train_model, batch_size=10,
    db= '/projects/caffe/cnn_mnist/train', # os.path.join(data_folder, 'mnist-train-nchw-leveldb'),
    db_type='lmdb')
softmax = AddLeNetModel(train_model, data)
#AddTrainingOperators(train_model, softmax, label)

# Testing model. We will set the batch size to 100, so that the testing
# pass is 100 iterations (10,000 images in total).
# For the testing model, we need the data input part, the main LeNetModel
# part, and an accuracy part. Note that init_params is set False because
# we will be using the parameters obtained from the train model.

#test_model = cnn.CNNModelHelper(
#    order="NCHW", name="vgg_test", init_params=False)
#data, label = AddInput(
#    test_model, batch_size=1,
#    db='/projects/caffe/cnn_mnist/train', # os.path.join(data_folder, 'mnist-test-nchw-leveldb'),
#    db_type='lmdb')
#softmax = AddLeNetModel(test_model, data)
# AddAccuracy(test_model, softmax, label)

# Deployment model. We simply need the main LeNetModel part.
#deploy_model = cnn.CNNModelHelper(
#    order="NCHW", name="vgg_deploy", init_params=False)
#AddLeNetModel(deploy_model, "data")
# You may wonder what happens with the param_init_net part of the deploy_model.
# No, we will not use them, since during deployment time we will not randomly
# initialize the parameters, but load the parameters from the db.

print('Created training and deploy models.')

# print(str(train_model.param_init_net.Proto())[:400] + '\n...')

#with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fid:
#    fid.write(str(train_model.net.Proto()))
#with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:
#    fid.write(str(train_model.param_init_net.Proto()))
#with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fid:
#    fid.write(str(test_model.net.Proto()))
#with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fid:
#    fid.write(str(test_model.param_init_net.Proto()))
#with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
#    fid.write(str(deploy_model.net.Proto()))
#print("Protocol buffers files have been created in your root folder: "+root_folder)

# The parameter initialization network only needs to be run once.
workspace.RunNetOnce(train_model.param_init_net)
# creating the network
workspace.CreateNet(train_model.net)
# set the number of iterations and track the accuracy & loss
total_iters = 3
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)
# Now, we will manually run the network for 200 iterations.
#for j in range(10):
start = time.time()
print start
for i in range(total_iters):
    workspace.RunNet(train_model.net.Proto().name)
    #accuracy = workspace.FetchBlob('accuracy')
    #loss = workspace.FetchBlob('loss')
    #print '{}: accuracy - {}, loss - {}'.format(i, accuracy, loss)

end = time.time()
print('Time: {}'.format(end - start))
