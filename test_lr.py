import math
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_svmlight_file

from optimizers import SANCOptimizer, SGDOptimizer, CROptimizer, SCROptimizer, NCDOptimizer


seed_val = 28173
tf.set_random_seed(seed_val)

def train_input_fn(x_train,y_train,batch_size):
    '''
    take the data from tensor_slices i.e. an array of datapoints in simple words.
    '''
    dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)) 
    
    dataset = dataset.shuffle(buffer_size=batch_size) \
                .batch(batch_size).repeat().make_one_shot_iterator()    
    return dataset.get_next()


def train_input_fn2(x_train,y_train,batch_size,sess):
  """
  can run into the 2GB limit for the tf.GraphDef protocol buffer.
  """
  x_train_placeholder = tf.placeholder(x_train.dtype, x_train.shape)
  y_train_placeholder = tf.placeholder(y_train.dtype, y_train.shape)
  dataset = tf.data.Dataset.from_tensor_slices((x_train_placeholder, y_train_placeholder))
  dataset = dataset.shuffle(buffer_size=batch_size).batch(batch_size).repeat()
  iterator = dataset.make_initializable_iterator()
  sess.run(iterator.initializer, feed_dict={x_train_placeholder: x_train,y_train_placeholder: y_train})
  return iterator.get_next()


def _main_lr(method, data, opt):
  dtype = opt['dtype']
  npidtype = np.int32
  if dtype == tf.float32:
    npfdtype = np.float32
  elif dtype == tf.float64:
    npfdtype = np.float64


  # Preparation of data
  from os.path import join
  suffix = '.txt'
  datapath = join('.','data',data+suffix)
  x_train, y_train = load_svmlight_file(datapath,dtype=npfdtype)
  x_train = x_train.toarray().astype(dtype=npfdtype)
  y_train = y_train.reshape(-1,1).astype(dtype=npidtype)
  if data == 'covtype':
    y_train[y_train==2.]=0.
  else:
    y_train[y_train==-1.]=0.
  num_examples, n_inputs = x_train.shape
  np.set_printoptions(suppress=True)
  
  f_vals_avg = None
  g_norms_avg = None
  repetition = opt['repetition']
  oracles_max = []
  for i in range(repetition):
    print('repetition: ', i)
    X = tf.placeholder(dtype, shape=(None, n_inputs), name='input')
    Y = tf.placeholder(dtype, shape=(None,1), name='label')

    batch_fraction = opt['batch_fraction']
    batch_size = math.floor(num_examples/batch_fraction)
    
    # Preparation of learning model
    cost = logistic_regression(n_inputs, X, Y, dtype = dtype, alpha = opt['LR_lambda'], nonconvex = True)
  
    oracles = []
    f_vals = []
    g_norms = []

    # Session start
    with tf.Session() as sess:
      if data == 'higgs' or data == 'covtype':
        next_element = train_input_fn2(x_train,y_train,batch_size,sess)
        next_element2 = train_input_fn2(x_train,y_train,batch_size,sess)
      else:
        next_element = train_input_fn(x_train,y_train,batch_size)
        next_element2 = train_input_fn(x_train,y_train,batch_size)

      if method == 'SANC':
        optimizer = SANCOptimizer(sess, cost, opt, dtype=dtype)
      elif method == 'SCR':
        optimizer = SCROptimizer(sess, cost, opt, dtype=dtype)
      elif method == 'CR':
        optimizer = CROptimizer(sess, cost, opt, dtype=dtype)
      elif method == 'SGD':
        optimizer = SGDOptimizer(sess,cost,learning_rate = opt['SGD_learning_rate'])
      elif method == 'NCD':
        optimizer = NCDOptimizer(sess, cost, opt, dtype=dtype)

      # Run the initializer
      init = tf.global_variables_initializer()
      sess.run(init)
      x_batch,y_batch = sess.run(next_element)
      x_batch2,y_batch2 = sess.run(next_element2)

      total_oracle_call = 0
      n_batch_fraction = 0
      itr = 0
      c = sess.run(cost, feed_dict={X: x_train, Y: y_train})
      oracles.append(0)
      f_vals.append(c)
      print('cost: ',c,'oracle calls:',total_oracle_call)
      while total_oracle_call < opt['oraclecall_limit']:

        num_oracle,g_norm = optimizer.minimize(X,Y,x_train,y_train,x_batch,y_batch,x_batch2,y_batch2, debug_print = False)
        c = sess.run(cost, feed_dict={X: x_train, Y: y_train})
        total_oracle_call += num_oracle*batch_size  
        oracles.append(total_oracle_call)
        f_vals.append(c)

        print('cost: ',c,'oracle calls:',total_oracle_call)

        x_batch,y_batch = sess.run(next_element)
        x_batch2,y_batch2 = sess.run(next_element2)
        
        if x_batch.shape[0]!=batch_size:
          x_batch,y_batch = sess.run(next_element)
          x_batch2,y_batch2 = sess.run(next_element2)
          
        itr += 1        

    tf.reset_default_graph()
    if i == 0:
      f_vals_avg = np.asarray(f_vals).reshape(1,-1)
    else:
      f_vals_avg = np.vstack((f_vals_avg,np.asarray(f_vals)))
    if len(oracles_max)<len(oracles):
      oracles_max = oracles

  print("Optimization Finished!")
  f_vals_avg = np.mean(f_vals_avg,axis = 0)
  return (oracles_max,f_vals_avg,g_norms_avg)


def logistic_regression(n_inputs, X, Y, dtype, alpha=0.1, nonconvex = True):
  """ Constructing logitic regression with nonconvex regularization """
  W = tf.get_variable("W", [n_inputs, 1], initializer = tf.constant_initializer(1.),dtype=dtype)
  Z = tf.matmul(X,W)
  cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y))

  reg = None
  if nonconvex is True:
    reg = tf.reduce_sum(tf.square(W)/(1+tf.square(W)))
  else:
    reg = tf.reduce_sum(tf.square(W))

  cost = cost + alpha * reg
  return cost