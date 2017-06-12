import numpy as np
import lmdb
import caffe
# from caffe.proto import caffe_pb2
from caffe2.proto import caffe2_pb2
import cv2
import glob
import skimage
import skimage.io
import skimage.transform
from PIL import Image

N = 40

# Let's pretend this is interesting data
X = np.zeros((N, 3, 224, 224), dtype=np.uint8)
y = np.zeros(N, dtype=np.int64)

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = X.nbytes * 10


train_data = [img for img in glob.glob("/projects/caffe/cnn_mnist/dataset/flowers/*jpg")]
for i , img_path in enumerate(train_data):
    img = cv2.imread(img_path)

    # img = skimage.io.imread(img_path)
    # img = img / 255.0

    # load image:
    # - as np.uint8 {0, ..., 255}
    #im = np.array(Image.open(img_path)) # or load whatever ndarray you need
    # if gray images, reshape to HxWx1
    #if len(im.shape) == 2:
    #    im = np.reshape(im, ([im.shape[0], im.shape[1], 1]))
    # - in BGR (switch from RGB)
    #im = im[:,:,::-1]
    # - in Channel x Height x Width order (switch from H x W x C)
    img = img.transpose((2,0,1))
    # - Turn to caffe object
    #im_dat = caffe.io.array_to_datum(im)

    X[i]=img
    y[i]=i%2


env = lmdb.open('train', map_size=map_size)
print X
print y
with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        #datum = caffe.proto.caffe_pb2.Datum()
        #datum.channels = X.shape[1]
        #datum.height = X.shape[2]
        #datum.width = X.shape[3]
        #datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        #print 'a ' + str(X[i])
        #datum.label = int(y[i])
        #print 'b ' + str(datum.label)
        #str_id = '{:08}'.format(i)

        #txn.put(str_id.encode('ascii'), datum.SerializeToString())
        label = i % 10 #[1 if i == 292 else 0 for i in range(1000)]
        width = 224
        height = 224

        img_data = np.random.rand(3, width, height)
        # ...

        # Create TensorProtos
        tensor_protos = caffe2_pb2.TensorProtos()
        img_tensor = tensor_protos.protos.add()
        img_tensor.dims.extend(img_data.shape)
        img_tensor.data_type = 1

        flatten_img = img_data.reshape(np.prod(img_data.shape))
        img_tensor.float_data.extend(flatten_img)

        label_tensor = tensor_protos.protos.add()
        label_tensor.data_type = 2
        label_tensor.int32_data.append(label)
        txn.put(
            '{}'.format(i).encode('ascii'),
            tensor_protos.SerializeToString()
        )
