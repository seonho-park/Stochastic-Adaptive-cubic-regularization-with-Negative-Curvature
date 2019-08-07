import math
import random
import numpy as np
import tensorflow as tf

from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize
from scipy.sparse import diags

LANCZOS_NUMERICAL_ERROR_STOP_FLOAT32 = 1e-20
LANCZOS_NUMERICAL_ERROR_STOP_FLOAT64 = 1e-80

random.seed(21663)

def local_cubic_model_fn(u,T,gamma0,sigma):
  return gamma0*u[0]+0.5*np.inner((T@u).flatten(),u.flatten())+1/3*sigma*(np.linalg.norm(u)**3)

class CROptimizerBase(object):
  def __init__(self, sess, loss, dtype=tf.float64):
    self.sess = sess
    self.loss = loss

    self.lanczos_num_err = LANCZOS_NUMERICAL_ERROR_STOP_FLOAT32
    if dtype == tf.float64:
      self.lanczos_num_err = LANCZOS_NUMERICAL_ERROR_STOP_FLOAT64

    self.W = tf.trainable_variables()
    self.newton_step = [tf.placeholder(dtype=dtype,shape=w.get_shape(),name='newton_step_'+str(i)) for i,w in enumerate(self.W)]
    self.grads = tf.gradients(self.loss, self.W)

    self.lanczos_q = [tf.placeholder(dtype=dtype,shape=w.get_shape(),name='lanczos_q_'+str(i)) for i,w in enumerate(self.W)]
    self.Hv = self.Hv_def(self.grads,self.lanczos_q)


  def compute_Newton_step(self, X,Y,x_batch,y_batch,x_batch2, y_batch2):
    """
      solve the Lanczos method and return the Newton step, 's'
    """
    Q = []
    deltas = []
    gammas = []

    grad_vals = self.sess.run(self.grads,feed_dict={X:x_batch,Y:y_batch})
    t = grad_vals
    q_prev = [np.zeros(t.shape) for t in t]
    for i in range(self.lanczos_max_iters):
      gamma = [np.sum(np.power(t,2)) for t in t]
      gamma = math.sqrt(np.sum(np.asarray(gamma)))
      gammas.append(gamma)
      
      if i>0:
        q_prev=[q for q in q]
      q = [np.divide(t,gamma) for t in t]
      Q.append(q)
      newton_step_dict = {X:x_batch2, Y:y_batch2}
      newton_step_dict_ = {self.lanczos_q[i]: q for i,q in enumerate(q)}
      newton_step_dict.update(newton_step_dict_)
      Aq = self.sess.run(self.Hv,feed_dict = newton_step_dict)
      delta = np.sum(np.asarray([np.sum(q * aq) for q, aq in zip(q, Aq)]))
      deltas.append(delta)
      t = [Aq[i]-delta*q-gamma*q_prev[i] for i,q in enumerate(q)]

    T = diags((np.asarray(deltas),np.asarray(gammas[1:]),np.asarray(gammas[1:])),[0,-1,1]).toarray()
    gamma0 = gammas[0]
    u = np.zeros((T.shape[0],1))
    result = minimize(local_cubic_model_fn,u,args = (T,gamma0,self.sigma), method = 'CG')
    u_opt = result.x

    s = self.retrieve_whole_dimension_vector(Q,u_opt)
    return s,grad_vals,deltas,gammas,Q

  def Hv_def(self, grads, vec):
    """ Computes Hessian vector product.

    grads: list of Tensorflow tensor objects
        Network gradients.
    vec: list of Tensorflow tensor objects
        Vector that is multiplied by the Hessian.

    return: list of Tensorflow tensor objects
        Result of multiplying Hessian by vec. """

    grad_v = [tf.reduce_sum(g * v) for g, v in zip(grads, vec)]
    Hv = tf.gradients(grad_v, self.W, stop_gradients=vec)
    Hv = [hv for hv, v in zip(Hv, vec)]

    return Hv

  def train_newton_op(self):
    """ Performs main training operation, i.e. updates weights

    return: list of Tensorflow tensor objects
        Main training operations """

    update_ops = []
    steps_and_vars = list(zip(self.newton_step, self.W))
    for s, w in reversed(steps_and_vars):
      with tf.control_dependencies(update_ops):
        update_ops.append(tf.assign(w, w + s))
    training_op = tf.group(*update_ops)

    return training_op


  def traceback_newton_op(self):
    update_ops = []
    steps_and_vars = list(zip(self.newton_step, self.W))
    for s, w in reversed(steps_and_vars):
      with tf.control_dependencies(update_ops):
        update_ops.append(tf.assign(w, w - s))
    training_op = tf.group(*update_ops)

    return training_op


  def retrieve_whole_dimension_vector(self,Q,vec):
    vec = vec.flatten()
    vec_ = [np.zeros(q.shape) for q in Q[0]]
    for i,q in enumerate(Q):
      for j,qq in enumerate(q):
        vec_[j]=vec_[j]+vec[i]*qq
    return vec_

  def train_nc_op(self):
    """ Performs main training operation, i.e. updates weights with negative curvature

    return: list of Tensorflow tensor objects
        Main training operations """

    update_ops = []
    step_size = 2.*self.z*abs(self.eigval)/self.L2_nc

    for w, v in reversed(list(zip(self.W, self.eigvec))):
      with tf.control_dependencies(update_ops):
        update_ops.append(tf.assign(w, w - step_size*v))

    training_op = tf.group(*update_ops)

    return training_op


  def train_sgd_op(self):
    update_ops = []
    step_size = 1./self.L1_nc

    for w, g in reversed(list(zip(self.W, self.grads))):
      with tf.control_dependencies(update_ops):
        update_ops.append(tf.assign(w, w - step_size*g))

    training_op = tf.group(*update_ops)

    return training_op


class SANCOptimizer(CROptimizerBase):
  """ 
  Methods to use:
  __init__:
      Creates Tensorflow graph and variables.
  minimize:
      Perfoms SANC optimization. """

  def __init__(self, sess, loss, opt, dtype=tf.float64):
    """ Creates Tensorflow graph and variables.

    sess: Tensorflow session object
        Used for conjugate gradient computation.
    loss: Tensorflow tensor object
        Loss function of the neural network.
    L2_nc: float
        Estimated L2 constant used for negative curvature update
    lanczos_max_iters: int
        Number of maximum iterations of Lanczos computations.
    sigma_init: float
        initial coefficient for the cubic regularization term
    eta1: float
        eta1 parameter for Newton step update
    eta2: float
        eta2 parameter for Newton step update
    gamma: float
        gamma parameter for Newton step update
    epsilon: float
        epsilon parameter
    dtype: Tensorflow type
        Type of Tensorflow variables. """

    super(SANCOptimizer, self).__init__(sess,loss,dtype)

    self.L1_nc = opt.get('SANC_L1_nc',1.0)
    self.L2_nc = opt.get('SANC_L2_nc',1.0)
    self.sigma = opt.get('SANC_sigma_init',1.)
    self.eta1 = opt.get('SANC_eta1',0.2)
    self.eta2 = opt.get('SANC_eta2',0.8)
    self.gamma = opt.get('SANC_gamma',2.)
    self.epsilon = opt.get('SANC_epsilon',0.0001)
    self.lanczos_max_iters = opt.get('SANC_lanczos_max_iters',5)
    
    ranval = random.uniform(0.,1.)
    self.z = 1. if ranval<0.5 else -1 # Rademacher random variable

    with tf.name_scope('nc_vars'):
      self.eigval = tf.placeholder(dtype=dtype,shape=[1],name='eigval')
      self.eigvec = [tf.placeholder(dtype=dtype,shape=w.get_shape(),name='eigvec_'+str(i)) for i,w in enumerate(self.W)]

    self.ops = {
      'train_newton': self.train_newton_op(),
      'traceback_newton': self.traceback_newton_op(),
      'train_nc': self.train_nc_op(),
      'train_sgd': self.train_sgd_op()
    }


  def minimize(self, X, Y, x_train, y_train, x_batch, y_batch, x_batch2, y_batch2, debug_print=False):
    s,grad_vals,deltas,gammas,Q = self.compute_Newton_step(X,Y,x_batch,y_batch,x_batch2,y_batch2)
    eigval_min,eigvec_min_ = eigh_tridiagonal(np.asarray(deltas),np.asarray(gammas[1:]),select='i',select_range=(0,0)) 
    
    fx = self.sess.run(self.loss,feed_dict={X:x_train,Y:y_train})

    norm_s = 0.
    for ss in s:
      norm_s += np.linalg.norm(ss)**2
    norm_s = math.sqrt(norm_s)

    gs = np.sum(np.asarray([np.dot(g.flatten(),s.flatten()) for g,s in zip(grad_vals,s)]))
    newton_step_dict = {X:x_batch2, Y:y_batch2}
    newton_step_dict_ = {self.lanczos_q[i]: s for i,s in enumerate(s)}
    newton_step_dict.update(newton_step_dict_)
    Bs = self.sess.run(self.Hv,feed_dict = newton_step_dict)
    sBs = np.sum(np.asarray([np.dot(Bs.flatten(),s.flatten()) for Bs,s in zip(Bs,s)]))
    mval = fx + gs + 0.5*sBs + 1/3*self.sigma*norm_s**3

    newton_step_dict = {self.newton_step[i]: s for i,s, in enumerate(s)}
    self.sess.run(self.ops['train_newton'], feed_dict = newton_step_dict)

    fxs  = self.sess.run(self.loss,feed_dict={X:x_train,Y:y_train})

    self.sess.run(self.ops['traceback_newton'],feed_dict = newton_step_dict)
    rho = (fx-fxs)/(fx-mval)
    if debug_print:
      print("fx, fxs, mval: ",fx,fxs,mval)

    newton_step = True
    if rho > self.eta2:
      if debug_print:
        print("[VERY SUCCESSFUL ITERATION] rho: ",rho)
      self.sigma = max(min(self.sigma,gammas[0]),self.lanczos_num_err)
      newton_step = True
    elif rho >= self.eta1:
      if debug_print:
        print("[SUCCESSFUL ITERATION] rho: ",rho)
      newton_step = True
    else:
      if debug_print:
        print("[UNSUCCESSFUL ITERATION] rho: ",rho)
      self.sigma = self.gamma*self.sigma
      newton_step = False

    if debug_print:
      print("SIGMA UPDATED VALUE: ", self.sigma," LEFT-MOST EIGENVALUE: ", eigval_min)

    if newton_step is True:
      self.sess.run(self.ops['train_newton'], newton_step_dict)
    else:
      if eigval_min < -self.epsilon:
        ranval = random.uniform(0.,1.)
        self.z = 1. if ranval<0.5 else -1
        eigvec = self.retrieve_whole_dimension_vector(Q,eigvec_min_)
        feed_dict = {}
        feed_dict.update({self.eigval:eigval_min})
        eigvec_dict = {self.eigvec[i]: v for i,v in enumerate(eigvec)}
        feed_dict.update(eigvec_dict)

        g_norm = gammas[0]
        feed_dict = {X:x_batch2, Y:y_batch2}
        feed_dict_ = {self.lanczos_q[i]: v for i,v in enumerate(eigvec)}
        feed_dict.update(feed_dict_)
        Bv = self.sess.run(self.Hv,feed_dict = feed_dict)
        vBv = np.sum(np.asarray([np.dot(Bv.flatten(),v.flatten()) for Bv,v in zip(Bv,eigvec)]))

        eps2 = self.epsilon
        eps_g = self.epsilon/4. 

        cond1 = -2*vBv**3/(3*self.L2_nc**2)- eps2*vBv**2/(6*self.L2_nc**2)
        cond2 = g_norm**2/(4*self.L1_nc)-eps_g/self.L1_nc
        if cond1>cond2:
          ranval = random.uniform(0.,1.)
          self.z = 1. if ranval<0.5 else -1
          feed_dict = {}
          feed_dict.update({self.eigval:eigval_min})
          eigvec_dict = {self.eigvec[i]: v for i,v in enumerate(eigvec)}
          feed_dict.update(eigvec_dict)
          self.sess.run(self.ops['train_nc'], feed_dict)
        else:
          self.sess.run(self.ops['train_sgd'], feed_dict)

    num_oraclecall = self.lanczos_max_iters+1
    return num_oraclecall, gammas[0]


class SCROptimizer(CROptimizerBase):
  def __init__(self, sess, loss, opt, dtype=tf.float64):
    super(SCROptimizer, self).__init__(sess,loss,dtype)

    self.sigma = opt.get('SCR_sigma_init',1.)
    self.eta1 = opt.get('SCR_eta1',0.2)
    self.eta2 = opt.get('SCR_eta2',0.8)
    self.gamma = opt.get('SCR_gamma',2.)
    self.lanczos_max_iters = opt.get('SCR_lanczos_max_iters',5)

    self.ops = {
      'train_newton': self.train_newton_op(),
      'traceback_newton': self.traceback_newton_op()
    }

  def minimize(self, X, Y, x_train, y_train, x_batch, y_batch, x_batch2, y_batch2, debug_print=False):
    s,grad_vals,deltas,gammas,Q = self.compute_Newton_step(X,Y,x_batch,y_batch,x_batch2,y_batch2)
    fx = self.sess.run(self.loss,feed_dict={X:x_train,Y:y_train})
    
    norm_s = 0.
    for ss in s:
      norm_s += np.linalg.norm(ss)**2
    norm_s = math.sqrt(norm_s)
    
    gs = np.sum(np.asarray([np.dot(g.flatten(),s.flatten()) for g,s in zip(grad_vals,s)]))
    newton_step_dict = {X:x_batch2, Y:y_batch2}
    newton_step_dict_ = {self.lanczos_q[i]: s for i,s in enumerate(s)}
    newton_step_dict.update(newton_step_dict_)
    Bs = self.sess.run(self.Hv,feed_dict = newton_step_dict)
    sBs = np.sum(np.asarray([np.dot(Bs.flatten(),s.flatten()) for Bs,s in zip(Bs,s)]))

    mval = fx + gs + 0.5*sBs + 1/3*self.sigma*norm_s**3

    newton_step_dict = {self.newton_step[i]: s for i,s, in enumerate(s)}
    self.sess.run(self.ops['train_newton'], feed_dict = newton_step_dict)
    
    fxs  = self.sess.run(self.loss,feed_dict={X:x_train,Y:y_train})
    
    self.sess.run(self.ops['traceback_newton'],feed_dict = newton_step_dict)
    rho = (fx-fxs)/(fx-mval)
    if debug_print:
      print("fx, fxs, mval: ",fx,fxs,mval)

    newton_step = True
    if rho > self.eta2:
      if debug_print:
        print("[VERY SUCCESSFUL ITERATION] rho: ",rho)
      self.sigma = max(min(self.sigma,gammas[0]),self.lanczos_num_err)
      newton_step = True

    elif rho >= self.eta1:
      if debug_print:
        print("[SUCCESSFUL ITERATION] rho: ",rho)
      newton_step = True
    else:
      if debug_print:
        print("[UNSUCCESSFUL ITERATION] rho: ",rho)
      self.sigma = self.gamma*self.sigma
      newton_step = False

    if debug_print:
      print("SIGMA UPDATED VALUE: ", self.sigma)

    if newton_step is True:
      self.sess.run(self.ops['train_newton'], newton_step_dict)

    num_oraclecall = self.lanczos_max_iters+1
    return num_oraclecall, gammas[0]



class CROptimizer(CROptimizerBase):
  def __init__(self, sess, loss, opt, dtype=tf.float64):
    super(CROptimizer, self).__init__(sess,loss,dtype)

    self.sigma = opt.get('CR_sigma',5.)
    self.lanczos_max_iters = opt.get('CR_lanczos_max_iters',5)
    
    self.ops = {
      'train_newton': self.train_newton_op()
    }



  def minimize(self, X, Y, x_train, y_train, x_batch, y_batch, x_batch2, y_batch2, debug_print=False):
    s,grad_vals,deltas,gammas,Q = self.compute_Newton_step(X,Y,x_batch,y_batch,x_batch2,y_batch2)
    fx = self.sess.run(self.loss,feed_dict={X:x_train,Y:y_train})
    norm_s = 0.
    for ss in s:
      norm_s += np.linalg.norm(ss)**2
    norm_s = math.sqrt(norm_s)
    
    newton_step_dict = {self.newton_step[i]: s for i,s, in enumerate(s)}
    self.sess.run(self.ops['train_newton'], newton_step_dict)

    num_oraclecall = self.lanczos_max_iters+1
    return num_oraclecall, gammas[0]



class NCDOptimizer(CROptimizerBase):
  def __init__(self, sess, loss, opt, dtype=tf.float64):
    super(NCDOptimizer, self).__init__(sess,loss,dtype)

    self.L1_nc = opt.get('NCD_L1_nc',1.0)
    self.L2_nc = opt.get('NCD_L2_nc',1.0)
    self.epsilon = opt.get('SANC_epsilon',0.0001)
    self.lanczos_max_iters = opt.get('NCD_lanczos_max_iters',5)

    ranval = random.uniform(0.,1.)
    self.z = 1. if ranval<0.5 else -1

    with tf.name_scope('nc_vars'):
      self.eigval = tf.placeholder(dtype=dtype,shape=[1],name='eigval')
      self.eigvec = [tf.placeholder(dtype=dtype,shape=w.get_shape(),name='eigvec_'+str(i)) for i,w in enumerate(self.W)]

    self.ops = {
      'train_sgd': self.train_sgd_op(),
      'train_nc': self.train_nc_op()
    }

  def minimize(self, X, Y, x_train, y_train, x_batch, y_batch, x_batch2, y_batch2, debug_print=False):
    Q = []
    deltas = []
    gammas = []

    grad_vals = self.sess.run(self.grads,feed_dict={X:x_batch,Y:y_batch})
    t = grad_vals
    q_prev = [np.zeros(t.shape) for t in t]
    for i in range(self.lanczos_max_iters):
      gamma = [np.sum(np.power(t,2)) for t in t]
      gamma = math.sqrt(np.sum(np.asarray(gamma)))
      gammas.append(gamma)
      
      if i>0:
        q_prev=[q for q in q]
      q = [np.divide(t,gamma) for t in t]
      Q.append(q)
      newton_step_dict = {X:x_batch2, Y:y_batch2}
      newton_step_dict_ = {self.lanczos_q[i]: q for i,q in enumerate(q)}
      newton_step_dict.update(newton_step_dict_)
      Aq = self.sess.run(self.Hv,feed_dict = newton_step_dict)
      delta = np.sum(np.asarray([np.sum(q * aq) for q, aq in zip(q, Aq)]))
      deltas.append(delta)
      t = [Aq[i]-delta*q-gamma*q_prev[i] for i,q in enumerate(q)]

    eigval_min,eigvec_min_ = eigh_tridiagonal(np.asarray(deltas),np.asarray(gammas[1:]),select='i',select_range=(0,0)) 

    eigvec = self.retrieve_whole_dimension_vector(Q,eigvec_min_)
    g_norm = gammas[0]

    feed_dict = {X:x_batch2, Y:y_batch2}
    feed_dict_ = {self.lanczos_q[i]: v for i,v in enumerate(eigvec)}
    feed_dict.update(feed_dict_)
    Bv = self.sess.run(self.Hv,feed_dict = feed_dict)
    vBv = np.sum(np.asarray([np.dot(Bv.flatten(),v.flatten()) for Bv,v in zip(Bv,eigvec)]))

    eps2 = self.epsilon
    eps_g = self.epsilon/4.
    cond1 = -2*vBv**3/(3*self.L2_nc**2)- eps2*vBv**2/(6*self.L2_nc**2)
    cond2 = g_norm**2/(4*self.L1_nc)-eps_g/self.L1_nc
    if cond1>cond2:
      ranval = random.uniform(0.,1.)
      self.z = 1. if ranval<0.5 else -1
      feed_dict = {}
      feed_dict.update({self.eigval:eigval_min})
      eigvec_dict = {self.eigvec[i]: v for i,v in enumerate(eigvec)}
      feed_dict.update(eigvec_dict)
      self.sess.run(self.ops['train_nc'], feed_dict)
    else:
      self.sess.run(self.ops['train_sgd'], feed_dict)

    num_oraclecall = self.lanczos_max_iters+1
    return num_oraclecall, g_norm


class SGDOptimizer(object):
  def __init__(self, sess, loss, learning_rate):
    self.sess = sess
    self.W = tf.trainable_variables()
    self.loss = loss
    self.grad = tf.gradients(self.loss, self.W)
    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  def minimize(self, X, Y, x_train, y_train, x_batch, y_batch, x_batch2, y_batch2, debug_print=False):
    _, c = self.sess.run([self.optimizer, self.loss], feed_dict={X:x_batch,Y:y_batch})
    grad_vals = self.sess.run(self.grad,feed_dict = {X:x_batch,Y:y_batch})
    gnorm = 0.
    for grads in grad_vals:
      gnorm += np.linalg.norm(grads)**2
    gnorm = math.sqrt(gnorm)
    return 1, gnorm

