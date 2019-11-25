import os
import tensorflow as tf
import numpy as np
import pickle

# from SANC_optimizer import SANCOptimizer
# from first_order_optimizers import SGDOptimizer, ADAMOptimizer
# from SCR_optimizer import SCROptimizer,SCRFIXEDOptimizer
from optimizers import SANCOptimizer, SGDOptimizer, CROptimizer, SCROptimizer, NCDOptimizer, CRMOptimizer



def conv2d(x, W, b, strides=1):
  # Conv2D wrapper, with bias and relu activation
  x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
  x = tf.nn.bias_add(x, b)
  x = tf.nn.tanh(x)
  # x = tf.nn.relu(x)
  return x
  # return tf.nn.relu(x)
  # return tf.nn.tanh(x)


def maxpool2d(x, k=2):
  # MaxPool2D wrapper
  return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                        padding='SAME')


# Create model
def conv_net(x, weights, biases, data):
  # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
  # Reshape to match picture format [Height x Width x Channel]
  # Tensor input become 4-D: [Batch Size, Height, Width, Channel]

  #   x = tf.reshape(x, shape=[-1, 28, 28, 1])
  # elif data == 'CIFAR':
  #   x = tf.reshape(x, shape=[-1, 28, 28, 1])

  # Convolution Layer
  conv1 = conv2d(x, weights['wc1'], biases['bc1'])
  # Max Pooling (down-sampling)
  conv1 = maxpool2d(conv1, k=2)

  # Convolution Layer
  conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
  # Max Pooling (down-sampling)
  conv2 = maxpool2d(conv2, k=2)

  # Fully connected layer
  # Reshape conv2 output to fit fully connected layer input
  # fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
  fc1 = tf.contrib.layers.flatten(conv2)

  fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
  fc1 = tf.nn.tanh(fc1)
  # fc1 = tf.nn.relu(fc1)
  
  # Apply Dropout
  # fc1 = tf.nn.dropout(fc1, dropout)

  # Output, class prediction
  out = tf.add(tf.matmul(fc1, weights['wout']), biases['bout'])
  return out


seed = 5
tf.set_random_seed(seed)

def compute_loss(X,Y,alpha,dtype,num_classes,num_channel,data):
  if data == 'MNIST':
    wd1 = tf.get_variable("wd1",[7*7*64,1024],initializer = tf.contrib.layers.xavier_initializer(),dtype=dtype)
  elif data == 'CIFAR':
    wd1 = tf.get_variable("wd1",[8*8*64,1024],initializer = tf.contrib.layers.xavier_initializer(),dtype=dtype)

  weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.get_variable("wc1",[5,5,num_channel,32],initializer = tf.contrib.layers.xavier_initializer(),dtype=dtype),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.get_variable("wc2",[5,5,32,64],initializer = tf.contrib.layers.xavier_initializer(),dtype=dtype),
    # fully connected, 7*7*64 inputs, 1024 outputs :MNIST
    # fully connected, 8*8*64 inputs, 1024 outputs :CIFAR
    'wd1': wd1,
    # 1024 inputs, 10 outputs (class prediction)
    'wout': tf.get_variable("wout",[1024,num_classes],initializer = tf.contrib.layers.xavier_initializer(),dtype=dtype)
  }

  biases = {
    'bc1': tf.get_variable("bc1",[32],initializer = tf.contrib.layers.xavier_initializer(),dtype=dtype),
    'bc2': tf.get_variable("bc2",[64],initializer = tf.contrib.layers.xavier_initializer(),dtype=dtype),
    'bd1': tf.get_variable("bd1",[1024],initializer = tf.contrib.layers.xavier_initializer(),dtype=dtype),
    'bout': tf.get_variable("bout",[num_classes],initializer = tf.contrib.layers.xavier_initializer(),dtype=dtype)
  }

  # Construct model
  logits = conv_net(X, weights, biases, data)
  # prediction = tf.nn.softmax(logits)

  # Define loss and optimizer
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

  reg = tf.nn.l2_loss(weights['wc1'])+tf.nn.l2_loss(weights['wc2'])+tf.nn.l2_loss(weights['wd1'])+tf.nn.l2_loss(weights['wout'])
  
  loss = loss + alpha*reg
  return loss

def get_mnist_datasets():
  from tensorflow.examples.tutorials.mnist import input_data
  dataset = input_data.read_data_sets("./data/MNIST/", one_hot=True,validation_size=0)
  dataset2 = input_data.read_data_sets("./data/MNIST/", one_hot=True,validation_size=0)
  dataset3 = input_data.read_data_sets("./data/MNIST/", one_hot=True,validation_size=0)
  return dataset,dataset2,dataset3


cifar10_dataset_folder_path = os.path.join('.','data','cifar-10-batches-py')

def load_cifar_batch(batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels


def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def one_hot_encode(x,num_classes):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), num_classes))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded

    
class CIFAR_DataSet(object):
  def __init__(self,x_train,y_train):
    self._images = x_train
    self._labels = y_train

    self._num_examples = self._images.shape[0]

    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels


  def next_batch(self,batch_size,shuffle=True):
    start = self._index_in_epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0] 

    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate(
          (images_rest_part, images_new_part), axis=0), np.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]



def get_cifar_datasets(num_classes):
  n_batches = 5

  for batch_i in range(1, n_batches + 1):
    x_train, y_train = load_cifar_batch(batch_i)
    x_train = normalize(x_train)
    y_train = one_hot_encode(y_train,num_classes)
    
    # print(x_train.shape,y_train.shape)
    if batch_i == 1:
      x_train_ = x_train
      y_train_ = y_train
    else:
      x_train_ = np.concatenate((x_train_,x_train),axis = 0)
      y_train_ = np.concatenate((y_train_,y_train),axis = 0)
      # print('x_train_.shape',x_train_.shape)
      # print('y_train_.shape',y_train_.shape)

  cifar_dataset = CIFAR_DataSet(x_train_,y_train_)
  cifar_dataset2 = CIFAR_DataSet(x_train_,y_train_)
  cifar_dataset3 = CIFAR_DataSet(x_train_,y_train_)
  # dataset = (x_train_,y_train_)  
  return cifar_dataset,cifar_dataset2,cifar_dataset3


def get_data(data,num_classes):
  if data == 'MNIST':
    return get_mnist_datasets()
  elif data == 'CIFAR':
    return get_cifar_datasets(num_classes)


def get_train_set(dataset,data):
  if data=='MNIST':
    x_train = dataset.train.images
    y_train = np.asarray(dataset.train.labels, dtype=np.int32)
    # x_train = np.reshape(x_train, shape=[-1, 28, 28, 1])
    x_train = x_train.reshape((-1, 28, 28, 1))

    return x_train,y_train
  elif data=='CIFAR':
    x_train = dataset.images
    y_train = dataset.labels
    return x_train,y_train


def get_next_batch(dataset,batch_size,data):
  if data =='MNIST':
    x_batch, y_batch = dataset.train.next_batch(batch_size)
    # x_batch = tf.reshape(x_batch, shape=[-1, 28, 28, 1])
    x_batch = x_batch.reshape((-1, 28, 28, 1))
    return x_batch,y_batch
  elif data == 'CIFAR':
    x_batch,y_batch = dataset.next_batch(batch_size)
    return x_batch,y_batch


def _main_cnn(method, data, opt):
  dtype = opt['dtype']
  batch_size = opt['batch_size']
  # num_input = 784 # MNIST data input (img shape: 28*28)
  num_classes = 10 # MNIST total classes (0-9 digits), CIFAR: 10
  
  f_vals_avg = None
  g_norms_avg = None

  repetition = opt['repetition']
  oracles_max = []

  for i in range(repetition):
    dataset, dataset2, dataset3 = get_data(data,num_classes)
    x_train,y_train = get_train_set(dataset,data)
    
    # print(type(x_train),x_train.shape)
    num_examples = int(x_train.shape[0])

    if data == 'MNIST':
      num_channel = 1
      num_dim = 28
      
    elif data == 'CIFAR':
      num_channel = 3
      num_dim = 32

    X = tf.placeholder(dtype, [None, num_dim, num_dim, num_channel])
    Y = tf.placeholder(dtype, [None, num_classes])
    
    alpha = opt['CNN_alpha']
    loss = compute_loss(X,Y,alpha,dtype,num_classes,num_channel,data)

    oracles = []
    f_vals = []
    g_norms = []

    # Session start
    with tf.Session() as sess:
      if method == 'SANC':
        optimizer = SANCOptimizer(sess, loss, opt, dtype=dtype)
      elif method == 'SCR':
        optimizer = SCROptimizer(sess, loss, opt, dtype=dtype)
      elif method == 'CR':
        optimizer = CROptimizer(sess, loss, opt, dtype=dtype)
      elif method == 'SGD':
        optimizer = SGDOptimizer(sess,loss,learning_rate = opt['SGD_learning_rate'])
      elif method == 'NCD':
        optimizer = NCDOptimizer(sess, loss, opt, dtype=dtype)
      elif method == 'CRM':
        optimizer = CRMOptimizer(sess, loss, opt, dtype=dtype)


      # Run the initializer
      init = tf.global_variables_initializer()
      sess.run(init)
      x_batch, y_batch = get_next_batch(dataset,batch_size,data)
      x_batch2, y_batch2 = get_next_batch(dataset2,batch_size,data)
      x_batch3, y_batch3 = get_next_batch(dataset3,batch_size,data)
      
      total_oracle_call = 0
      n_batch_fraction = 0

      numset = 6
      num = int(num_examples/numset)

      c = []
      for k in range(numset):
        x_train_i = x_train[num*k:num*(k+1)]
        y_train_i = y_train[num*k:num*(k+1)]
        # print("x_train_i",x_train_i.shape,x_train_i[0,:])
        # print("TEST",type(x_train_i),x_train_i.shape)
        c.append(sess.run(loss, feed_dict={X: x_train_i,Y: y_train_i}))
      c = np.mean(np.asarray(c))

      oracles.append(0)
      f_vals.append(c)
      while total_oracle_call < opt['oraclecall_limit']:
        num_oracle,g_norm = optimizer.minimize(X,Y,x_batch3,y_batch3,x_batch,y_batch,x_batch2,y_batch2, debug_print = True)
        
        c = []
        for k in range(numset):
          x_train_i = x_train[num*k:num*(k+1),:]
          y_train_i = y_train[num*k:num*(k+1)]
          # print("TEST",type(x_train_i))
          c.append(sess.run(loss, feed_dict={X: x_train_i,Y: y_train_i}))
        c = np.mean(np.asarray(c))


        total_oracle_call += num_oracle*batch_size

        oracles.append(total_oracle_call)
        f_vals.append(c)
        print('cost: ',c,'oracle calls:',total_oracle_call)

        x_batch, y_batch = get_next_batch(dataset,batch_size,data)
        x_batch2, y_batch2 = get_next_batch(dataset2,batch_size,data)
        x_batch3, y_batch3 = get_next_batch(dataset3,batch_size,data)
        
        # if x_batch.shape[0]!=batch_size:
        #   x_batch, y_batch = dataset.train.next_batch(batch_size)
        #   x_batch2, y_batch2 = dataset2.train.next_batch(batch_size)
        #   x_batch3, y_batch3 = dataset3.train.next_batch(batch_size)

    tf.reset_default_graph()
    if i == 0:
      f_vals_avg = np.asarray(f_vals).reshape(1,-1)
    else:
      f_vals_avg = np.vstack((f_vals_avg,np.asarray(f_vals)))
    if len(oracles_max)<len(oracles):
      oracles_max = oracles

  print("Optimization Finished!")
  print(f_vals_avg)
  f_vals_avg = np.mean(f_vals_avg,axis = 0)
  print(f_vals_avg)
  return (oracles_max,f_vals_avg,g_norms_avg)
  