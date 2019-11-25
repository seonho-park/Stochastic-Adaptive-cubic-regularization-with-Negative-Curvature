import tensorflow as tf

from utilities import save_figures, save_options, save_result_data
from test_lr import _main_lr
from test_cnn import _main_cnn
from test_mlp import _main_mlp

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

  # CRM parameter setting
  opt['CRM_beta'] = 12

  data = 'w1a' # dataset 
  
  opt['oraclecall_limit'] = 30000
  
  _main = _main_lr
  


  # Test for figure 1 in the paper
  # opt['SANC_sigma_init'] = 0.001
  # opt['SCR_sigma_init'] = 0.001
  
  # fig_data['SANC'] = _main(method = 'SANC', data = data, opt = opt) 
  # fig_data['SCR'] = _main(method = 'SCR', data = data, opt = opt) 
  # fig_data['CR'] = _main(method = 'CR', data = data, opt = opt) 
  # fig_data['CRM'] = _main(method = 'CRM', data = data, opt = opt) 
  # filename_suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  # save_figures(fig_data, dataname = data, fn_prefix=filename_suffix, log_scale = False, show = True)
  # save_options(opt, dataname = data, fn_prefix=filename_suffix)
  # save_result_data(fig_data, dataname= data, fn_prefix = filename_suffix)

  # # Test for figure 1(a) top in the paper
  opt['SANC_sigma_init'] = 1.
  opt['SCR_sigma_init'] = 1.
  opt['oraclecall_limit'] = 50000
  opt['repetition'] = 1 # the number of executions

  # fig_data2 = {}
  # fig_data2['SANC'] = _main(method = 'SANC', data = data, opt = opt) 
  # fig_data2['SCR'] = _main(method = 'SCR', data = data, opt = opt) 
  # fig_data2['CR'] = _main(method = 'CR', data = data, opt = opt) 
  # fig_data2['SGD'] = _main(method = 'SGD', data = data, opt = opt) 
  # fig_data2['NCD'] = _main(method = 'NCD', data = data, opt = opt) 
  # fig_data2['CRM'] = _main(method = 'CRM', data = data, opt = opt) 
  # filename_suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  # save_figures(fig_data2, dataname = data, fn_prefix=filename_suffix, log_scale = False, show = True)
  # save_options(opt, dataname = data, fn_prefix=filename_suffix)
  # save_result_data(fig_data2, dataname= data, fn_prefix = filename_suffix)

  # figure 1(a) bottom
  # data  = 'higgs'
  # opt['SANC_sigma_init'] = 1.
  # opt['SCR_sigma_init'] = 1.
  # opt['oraclecall_limit'] = 60000000
  # opt['repetition'] = 1 # the number of executions
  # opt['SANC_L1_nc'] = 10.
  # opt['SANC_L2_nc'] = 100.
  # fig_data = {}
  # fig_data['SANC'] = _main(method = 'SANC', data = data, opt = opt) 
  # fig_data['SCR'] = _main(method = 'SCR', data = data, opt = opt) 
  # fig_data['CR'] = _main(method = 'CR', data = data, opt = opt) 
  # fig_data['SGD'] = _main(method = 'SGD', data = data, opt = opt) 
  # fig_data['NCD'] = _main(method = 'NCD', data = data, opt = opt) 
  # fig_data['CRM'] = _main(method = 'CRM', data = data, opt = opt) 
  # filename_suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  # save_figures(fig_data, dataname = data, fn_prefix=filename_suffix, log_scale = False, show = True)
  # save_options(opt, dataname = data, fn_prefix=filename_suffix)
  # save_result_data(fig_data, dataname= data, fn_prefix = filename_suffix)


  # exit()
  opt["MLP_alpha"] = 0.05
  # figure 1(b) top 
  _main = _main_mlp
  # fig_data = {}
  # opt['SANC_L1_nc'] = 100.
  # opt['SANC_L2_nc'] = 100.
  # opt['SANC_eta1'] = 0.1
  # opt['SANC_eta2'] = 0.3
  
  # opt['SCR_eta1'] = 0.1
  # opt['SCR_eta2'] = 0.3

  # opt['NCD_L1_nc'] = 100.
  # opt['NCD_L2_nc'] = 100.

  # opt['batch_size'] = 128
  # opt['oraclecall_limit'] = 40000
  
  # data = 'seismic'
  # opt['num_classes'] = 3
  # opt['CRM_beta'] = 2
  # opt['repetition'] = 10 # the number of executions
  # fig_data['CR'] = _main(method = 'CR', data = data, opt = opt) 
  # fig_data['CRM'] = _main(method = 'CRM', data = data, opt = opt) 
  # fig_data['NCD'] = _main(method = 'NCD', data = data, opt = opt) 
  # fig_data['SCR'] = _main(method = 'SCR', data = data, opt = opt) 
  # fig_data['SANC'] = _main(method = 'SANC', data = data, opt = opt) 
  # fig_data['SGD'] = _main(method = 'SGD', data = data, opt = opt) 
  # filename_suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  # save_figures(fig_data, dataname = data, fn_prefix=filename_suffix, log_scale = False, show = True)
  # save_options(opt, dataname = data, fn_prefix=filename_suffix)
  # save_result_data(fig_data, dataname= data, fn_prefix = filename_suffix)


  # figure 1(b) bottom

  data = 'segment'
  opt['num_classes'] = 7
  opt["MLP_alpha"] = 0.05

  opt['SANC_L1_nc'] = 100.
  opt['SANC_L2_nc'] = 100.
  opt['SANC_eta1'] = 0.1
  opt['SANC_eta2'] = 0.3
  
  opt['SCR_eta1'] = 0.1
  opt['SCR_eta2'] = 0.3
  
  opt['NCD_L1_nc'] = 100.
  opt['NCD_L2_nc'] = 100.

  opt['CRM_beta'] = 2 
  
  opt['repetition'] = 10 # the number of executions
  opt['batch_size'] = 128
  opt['oraclecall_limit'] = 50000
  fig_data = {}
  fig_data['CR'] = _main(method = 'CR', data = data, opt = opt) 
  fig_data['CRM'] = _main(method = 'CRM', data = data, opt = opt) 
  fig_data['NCD'] = _main(method = 'NCD', data = data, opt = opt) 
  fig_data['SCR'] = _main(method = 'SCR', data = data, opt = opt) 
  fig_data['SANC'] = _main(method = 'SANC', data = data, opt = opt) 
  fig_data['SGD'] = _main(method = 'SGD', data = data, opt = opt) 
  filename_suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  save_figures(fig_data, dataname = data, fn_prefix=filename_suffix, log_scale = False, show = True)
  save_options(opt, dataname = data, fn_prefix=filename_suffix)
  save_result_data(fig_data, dataname= data, fn_prefix = filename_suffix)


  # _main = _main_cnn
  # fig_data = {}
  # data = 'MNIST'

  # opt['batch_size'] = 128
  # opt['CNN_alpha'] = 0.01 # regularization coefficient
  # opt['SANC_L1_nc'] = 100.
  # opt['SANC_L2_nc'] = 100.
  # opt['oraclecall_limit'] = 10000
  # # fig_data['SANC'] = _main(method = 'SANC', data = data, opt = opt) 
  # # fig_data['CRM'] = _main(method = 'CRM', data = data, opt = opt) 
  # # fig_data['CR'] = _main(method = 'CR', data = data, opt = opt) 
  # # fig_data['SGD'] = _main(method = 'SGD', data = data, opt = opt) 
  # filename_suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  # save_figures(fig_data, dataname = data, fn_prefix=filename_suffix, log_scale = False, show = True)
  # save_options(opt, dataname = data, fn_prefix=filename_suffix)
  # save_result_data(fig_data, dataname= data, fn_prefix = filename_suffix)



if __name__ == '__main__':
  main()







