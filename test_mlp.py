import os
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_svmlight_file

# from SANC_optimizer import SANCOptimizer
# from first_order_optimizers import SGDOptimizer, ADAMOptimizer
# from SCR_optimizer import SCROptimizer,SCRFIXEDOptimizer

from test_lr import train_input_fn
# from test_cnn import one_hot_encode

from optimizers import SANCOptimizer, SGDOptimizer, CROptimizer, SCROptimizer, NCDOptimizer, CRMOptimizer

# def train_input_fn(x_train,y_train,batch_size):
#     ##Here we are using dataset API.
#     '''
#     take the data from tensor_slices i.e. an array of datapoints in simple words.
#     '''
#     dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)) 
    
#     dataset = dataset.shuffle(buffer_size=batch_size) \
#                 .batch(batch_size).repeat().make_one_shot_iterator()
#     # dataset = dataset.map(lambda x,y:_parse_(x,y)).shuffle(buffer_size=batch_size) \
#     #             .batch(batch_size).make_one_shot_iterator()
    
#     return dataset.get_next()

# def one_hot_encode(x):
#     """
#         argument
#             - x: a list of labels
#         return
#             - one hot encoding matrix (number of labels, number of class)
#     """
#     encoded = np.zeros((len(x), 10))

#     for idx, val in enumerate(x):
#         encoded[idx][val] = 1

#     return encoded


def compute_loss(X,Y,alpha,dtype,num_classes,n_inputs,data):
  """ Constructing simple neural network """
  n_hidden1 = 300
  n_hidden2 = 500
  n_outputs = num_classes
  w_1 = tf.get_variable("w_1",(n_inputs, n_hidden1),initializer = tf.contrib.layers.xavier_initializer(),dtype = dtype)
  b_1 = tf.get_variable("b_1",(n_hidden1),initializer =  tf.contrib.layers.xavier_initializer(),dtype = dtype)
  y_1 = tf.nn.tanh(tf.matmul(X, w_1) + b_1)
  # y_1 = tf.nn.relu(tf.matmul(X, w_1) + b_1)
  n_inputs_out = int(y_1.get_shape()[1])

  w_2 = tf.get_variable("w_2",(n_inputs_out, n_hidden2),initializer = tf.contrib.layers.xavier_initializer(),dtype = dtype)
  b_2 = tf.get_variable("b_2",(n_hidden2),initializer =  tf.contrib.layers.xavier_initializer(),dtype = dtype)
  y_2 = tf.nn.tanh(tf.matmul(y_1, w_2) + b_2)
  # y_2 = tf.nn.relu(tf.matmul(y_1,w_2)+b_2)
  n_inputs_out = int(y_2.get_shape()[1])


  w_out = tf.get_variable("w_out",(n_inputs_out, n_outputs),initializer = tf.contrib.layers.xavier_initializer(),dtype = dtype)
  b_out = tf.get_variable("b_out",(n_outputs),initializer =  tf.contrib.layers.xavier_initializer(),dtype = dtype)
  y_out = tf.matmul(y_2, w_out) + b_out
  y_out_sm = tf.nn.softmax(y_out)

  Y_one_hot = tf.one_hot(Y, n_outputs, dtype=tf.int32)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_out_sm, labels=Y_one_hot))
  reg = tf.nn.l2_loss(w_1)+tf.nn.l2_loss(w_out)
  loss = loss +alpha*reg
  
  return loss


def _main_mlp(method, data, opt):
  dtype = opt['dtype']
  npidtype = np.int32
  
  if dtype == tf.float32:
    npfdtype = np.float32
    
  elif dtype == tf.float64:
    npfdtype = np.float64


  batch_size = opt['batch_size']
  num_classes = opt['num_classes']

  # Preparation of data
  # data = 'protein1'
  from os.path import join
  suffix = '.txt'
  datapath = join('.','data',data+suffix)
  print('datapath',datapath)
  x_train, y_train = load_svmlight_file(datapath,dtype=npfdtype)
  x_train = x_train.toarray().astype(dtype=npfdtype)
  y_train = y_train.reshape(-1,1).astype(dtype=npidtype)
  # y_train = one_hot_encode(y_train,num_classes)

  num_examples, n_inputs = x_train.shape

  f_vals_avg = None
  g_norms_avg = None
  repetition = opt['repetition']
  oracles_max = []
  for i in range(repetition):
    X = tf.placeholder(dtype, shape=(None, n_inputs), name='input')
    Y = tf.placeholder(tf.int32, shape=(None), name='label')

    # Preparation of learning model
    alpha = opt["MLP_alpha"]

    loss = compute_loss(X,Y,alpha,dtype,num_classes,n_inputs,data)
  
    oracles = []
    f_vals = []
    g_norms = []
    # Session start
    with tf.Session() as sess:
      next_element = train_input_fn(x_train,y_train,batch_size)
      next_element2 = train_input_fn(x_train,y_train,batch_size)

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
      x_batch,y_batch = sess.run(next_element)
      x_batch2,y_batch2 = sess.run(next_element2)

      total_oracle_call = 0
      n_batch_fraction = 0
      itr = 0
      c = sess.run(loss, feed_dict={X: x_train, Y: y_train})
      # c_reg = sess.run(cost_reg, feed_dict={X: x_train, Y: y_train})
      # print("initial c and c_reg", c, c_reg)
      oracles.append(0)
      f_vals.append(c)
      while total_oracle_call < opt['oraclecall_limit']:

        num_oracle,g_norm = optimizer.minimize(X,Y,x_train,y_train,x_batch,y_batch,x_batch2,y_batch2, debug_print = True)

        c = sess.run(loss, feed_dict={X: x_train, Y: y_train})
  
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
      # g_norms_avg = np.asarray(g_norms).reshape(1,-1)
    else:
      f_vals_avg = np.vstack((f_vals_avg,np.asarray(f_vals)))
      # g_norms_avg = np.vstack((g_norms_avg,np.asarray(g_norms)))
    if len(oracles_max)<len(oracles):
      oracles_max = oracles

  print("Optimization Finished!")
  print(f_vals_avg)
  # print(g_norms_avg)
  f_vals_avg = np.mean(f_vals_avg,axis = 0)
  # g_norms_avg = np.mean(g_norms_avg,axis = 0)
  print(f_vals_avg)
  # print(g_norms_avg)
  return (oracles_max,f_vals_avg,g_norms_avg)




  # f_vals_avg = None
  # g_norms_avg = None

  # repetition = opt['repetition']
  # oracles_max = []

  # for i in range(repetition):
  #   dataset, dataset2, dataset3 = get_data(data,num_classes)
  #   x_train,y_train = get_train_set(dataset,data)
    
  #   # print(type(x_train),x_train.shape)
  #   num_examples = int(x_train.shape[0])

  #   X = tf.placeholder(dtype, (None, n_inputs))
  #   Y = tf.placeholder(dtype, (None, num_classes))
    
  #   alpha = 0.1
  #   loss = compute_loss(X,Y,alpha,dtype,num_classes,num_channel,data)

  #   oracles = []
  #   f_vals = []
  #   g_norms = []

  #   # Session start
  #   with tf.Session() as sess:
  #     if method == 'SANC':
  #       optimizer = SANCOptimizer(sess, loss, opt, dtype=dtype)
  #     elif method == 'SCR':
  #       optimizer = SCROptimizer(sess, loss, opt, dtype=dtype)
  #     elif method == 'SCR_FIXED':
  #       optimizer = SCRFIXEDOptimizer(sess, loss, opt, dtype=dtype)
  #     elif method == 'SGD':
  #       optimizer = SGDOptimizer(sess,loss,learning_rate = opt['SGD_learning_rate'])
  #     elif method == 'ADAM':
  #       optimizer = ADAMOptimizer(sess,loss,learning_rate = opt['ADAM_learning_rate'])

  #     # Run the initializer
  #     init = tf.global_variables_initializer()
  #     sess.run(init)
  #     x_batch, y_batch = get_next_batch(dataset,batch_size,data)
  #     x_batch2, y_batch2 = get_next_batch(dataset2,batch_size,data)
  #     x_batch3, y_batch3 = get_next_batch(dataset3,batch_size,data)
      
  #     total_oracle_call = 0
  #     n_batch_fraction = 0

  #     numset = 6
  #     num = int(num_examples/numset)

  #     c = []
  #     for k in range(numset):
  #       x_train_i = x_train[num*k:num*(k+1)]
  #       y_train_i = y_train[num*k:num*(k+1)]
  #       # print("x_train_i",x_train_i.shape,x_train_i[0,:])
  #       # print("TEST",type(x_train_i),x_train_i.shape)
  #       c.append(sess.run(loss, feed_dict={X: x_train_i,Y: y_train_i}))
  #     c = np.mean(np.asarray(c))

  #     oracles.append(0)
  #     f_vals.append(c)
  #     while total_oracle_call < opt['oraclecall_limit']:
  #       num_oracle,g_norm = optimizer.minimize(X,Y,x_batch3,y_batch3,x_batch,y_batch,x_batch2,y_batch2, debug_print = True)
        
  #       c = []
  #       for k in range(numset):
  #         x_train_i = x_train[num*k:num*(k+1),:]
  #         y_train_i = y_train[num*k:num*(k+1)]
  #         # print("TEST",type(x_train_i))
  #         c.append(sess.run(loss, feed_dict={X: x_train_i,Y: y_train_i}))
  #       c = np.mean(np.asarray(c))


  #       total_oracle_call += num_oracle*batch_size

  #       oracles.append(total_oracle_call)
  #       f_vals.append(c)
  #       print('cost: ',c,'oracle calls:',total_oracle_call)

  #       x_batch, y_batch = get_next_batch(dataset,batch_size,data)
  #       x_batch2, y_batch2 = get_next_batch(dataset2,batch_size,data)
  #       x_batch3, y_batch3 = get_next_batch(dataset3,batch_size,data)
        
  #       # if x_batch.shape[0]!=batch_size:
  #       #   x_batch, y_batch = dataset.train.next_batch(batch_size)
  #       #   x_batch2, y_batch2 = dataset2.train.next_batch(batch_size)
  #       #   x_batch3, y_batch3 = dataset3.train.next_batch(batch_size)

  #   tf.reset_default_graph()
  #   if i == 0:
  #     f_vals_avg = np.asarray(f_vals).reshape(1,-1)
  #   else:
  #     f_vals_avg = np.vstack((f_vals_avg,np.asarray(f_vals)))
  #   if len(oracles_max)<len(oracles):
  #     oracles_max = oracles

  # print("Optimization Finished!")
  # print(f_vals_avg)
  # f_vals_avg = np.mean(f_vals_avg,axis = 0)
  # print(f_vals_avg)
  # return (oracles_max,f_vals_avg,g_norms_avg)
  # 