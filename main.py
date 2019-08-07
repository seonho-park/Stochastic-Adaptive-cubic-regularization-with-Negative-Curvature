import tensorflow as tf

from utilities import save_figures, save_options, save_result_data
from test_lr import _main_lr

import datetime

seed_val = 22345
tf.set_random_seed(seed_val)

def main():
  opt = {}
  # Basic setting  
  opt['batch_fraction'] = 20 # use for LIBSVM data
  opt['dtype'] = tf.float32
  opt['repetition'] = 1 # the number of executions

  # Logistic regression parameter
  opt['LR_lambda'] = 1. # regularization coefficient

  # SANC parameter setting
  opt['SANC_eta1'] = 0.2
  opt['SANC_eta2'] = 0.8
  opt['SANC_gamma'] = 2.
  opt['SANC_epsilon'] = 1e-4
  opt['SANC_lanczos_max_iters'] = 5
  opt['SANC_L1_nc'] = 10.
  opt['SANC_L2_nc'] = 10.

  # CR parameter setting
  opt['CR_sigma'] = 5.
  opt['CR_lanczos_max_iters'] = 5

  # SCR parameter setting
  opt['SCR_eta1'] = 0.2
  opt['SCR_eta2'] = 0.8
  opt['SCR_gamma'] = 2.
  opt['SCR_lanczos_max_iters'] = 5

  #  NCD parameter setting
  opt['NCD_lanczos_max_iters'] = 5
  opt['NCD_L1_nc'] = 10.
  opt['NCD_L2_nc'] = 10.
  opt['NCD_epsilon'] = 1e-4

  # # SGD parameter setting
  opt['SGD_learning_rate'] = 0.01

  data = 'w1a' # dataset 
  
  opt['oraclecall_limit'] = 30000
  
  _main = _main_lr


  # Test for figure 1 in the paper
  opt['SANC_sigma_init'] = 0.001
  opt['SCR_sigma_init'] = 0.001
  fig_data = {}
  fig_data['SANC'] = _main(method = 'SANC', data = data, opt = opt) 
  fig_data['SCR'] = _main(method = 'SCR', data = data, opt = opt) 
  filename_suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  save_figures(fig_data, dataname = data, fn_prefix=filename_suffix, log_scale = False, show = True)
  save_options(opt, dataname = data, fn_prefix=filename_suffix)
  save_result_data(fig_data, dataname= data, fn_prefix = filename_suffix)

  # Test for figure 2 in the paper
  opt['SANC_sigma_init'] = 1.
  opt['SCR_sigma_init'] = 1.
  fig_data2 = {}
  fig_data2['SANC'] = _main(method = 'SANC', data = data, opt = opt) 
  fig_data2['SCR'] = _main(method = 'SCR', data = data, opt = opt) 
  fig_data2['CR'] = _main(method = 'CR', data = data, opt = opt) 
  fig_data2['SGD'] = _main(method = 'SGD', data = data, opt = opt) 
  fig_data2['NCD'] = _main(method = 'NCD', data = data, opt = opt) 
  filename_suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  save_figures(fig_data2, dataname = data, fn_prefix=filename_suffix, log_scale = False, show = True)
  save_options(opt, dataname = data, fn_prefix=filename_suffix)
  save_result_data(fig_data2, dataname= data, fn_prefix = filename_suffix)


if __name__ == '__main__':
  main()






