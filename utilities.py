import os
import matplotlib.pyplot as plt
import csv
import _pickle as pickle
import numpy as np

def preprocess(f_vals):
  min_f_val = 1000
  for f_vals_ in f_vals:
    min_f_val = min(f_vals_) if (min(f_vals_) <= min_f_val) else min_f_val
  print(min_f_val)

  eps = min_f_val * 1e-6
  for i in range(len(f_vals)):
      f_vals[i] = [f_val - min_f_val + eps for f_val in f_vals[i]]
  return f_vals


def save_figures(fig_data, dataname, fn_prefix, fval = True, dnorm = False, log_scale = True, show = True):
  """ make and save plots"""

  x_oracles = []
  g_norms = []
  f_vals = []
  line_style = []
  colors = []
  legends = []

  oracle_scale = 10000

  if 'SGD' in fig_data:
    oracles,f_vals_avg,g_norms_avg = fig_data['SGD']
    oracles = (np.asarray(oracles)/oracle_scale).tolist()
    x_oracles.append(oracles)
    g_norms.append(g_norms_avg)
    f_vals.append(f_vals_avg)
    line_style.append('-')
    colors.append('#5cb85c') 
    legends.append('SGD')
  
  if 'CR' in fig_data:
    oracles,f_vals_avg,g_norms_avg = fig_data['CR']
    oracles = (np.asarray(oracles)/oracle_scale).tolist()
    x_oracles.append(oracles)
    g_norms.append(g_norms_avg)
    f_vals.append(f_vals_avg)
    line_style.append('-')
    colors.append('#be29ec') 
    legends.append('CR')
  
  if 'SCR' in fig_data:
    oracles,f_vals_avg,g_norms_avg = fig_data['SCR']
    oracles = (np.asarray(oracles)/oracle_scale).tolist()
    x_oracles.append(oracles)
    g_norms.append(g_norms_avg)
    f_vals.append(f_vals_avg)
    line_style.append('-')
    colors.append('#5bc0de') 
    legends.append('SCR')
  
  if 'NCD' in fig_data:
    oracles,f_vals_avg,g_norms_avg = fig_data['NCD']
    oracles = (np.asarray(oracles)/oracle_scale).tolist()
    x_oracles.append(oracles)
    g_norms.append(g_norms_avg)
    f_vals.append(f_vals_avg)
    line_style.append('-')
    colors.append('#ffc425') 
    legends.append('NCD')

  if 'CRM' in fig_data:
    oracles,f_vals_avg,g_norms_avg = fig_data['CRM']
    oracles = (np.asarray(oracles)/oracle_scale).tolist()
    x_oracles.append(oracles)
    g_norms.append(g_norms_avg)
    f_vals.append(f_vals_avg)
    line_style.append('-')
    colors.append('#5d6d7e') 
    legends.append('CRM')

  if 'SANC' in fig_data:
    oracles,f_vals_avg,g_norms_avg = fig_data['SANC']
    oracles = (np.asarray(oracles)/oracle_scale).tolist()
    x_oracles.append(oracles)
    g_norms.append(g_norms_avg)
    f_vals.append(f_vals_avg)
    line_style.append('--')
    colors.append('#d9534f') 
    legends.append('SANC(ours)')

  if log_scale == True:
    f_vals = preprocess(f_vals)

  # Function value plot 
  if fval is True:
    for i in range(len(f_vals)):
      plt.plot(x_oracles[i], f_vals[i], line_style[i], color=colors[i], linewidth=2.)
    plt.legend(legends, fontsize=12, loc=1)

    if log_scale == True: 
      plt.yscale('log')
      plt.ylabel('$\log(f-f^*)$', fontsize=12)
    else:
      plt.yscale('linear')
      plt.ylabel('$f$', fontsize=12)
    plt.xlabel('number of oracle calls', fontsize=12)

    fn = fn_prefix+'_fval_'+dataname+'.png'
    path = os.path.join('.','figures_backup')
    if not os.path.exists(path):
      os.makedirs(path)
    subpath = os.path.join('.','figures_backup',dataname)
    if not os.path.exists(subpath):
      os.makedirs(subpath)


    fullfn = os.path.join(subpath,fn)
    plt.savefig(fullfn)
    
    if show:
      plt.show()

    plt.close()

  # Gradient norm plot
  if dnorm is True:
    for i,_ in enumerate(list(data_keys)):
      plt.plot(x_oracles[i], g_norms[i], line_style[i], color=colors[i], linewidth=2.)
    plt.legend(legends, fontsize=12, loc=1)

    if log_scale == True: 
      plt.yscale('log')
      plt.ylabel('$||g||$', fontsize=12)
    else:
      plt.yscale('linear')
      plt.ylabel('$||g||$', fontsize=12)
    plt.xlabel('number of oracle calls', fontsize=12)

    fn = fn_prefix+'_gnorm_'+dataname+'.png'
    fullfn = os.path.join(subpath,fn)
    plt.savefig(fullfn)
    
    if show:
      plt.show()

    plt.close()
  return


def save_options(opt, dataname, fn_prefix):
  path = os.path.join('.','options_backup')
  subpath = os.path.join('.','options_backup',dataname)
  if not os.path.exists(path):
    os.makedirs(path)
  if not os.path.exists(subpath):
    os.makedirs(subpath)

  fn = fn_prefix+'.csv'
  fullfn = os.path.join(subpath,fn)

  w = csv.writer(open(fullfn, "w"))
  for key, val in sorted(opt.items()):
    w.writerow([key, val])
  return 


def save_result_data(fig_data, dataname, fn_prefix):
  path = os.path.join('.','results_backup')
  subpath = os.path.join('.','results_backup',dataname)
  if not os.path.exists(path):
    os.makedirs(path)
  if not os.path.exists(subpath):
    os.makedirs(subpath)

  fn = fn_prefix+'.pkl'
  fullfn = os.path.join(subpath,fn)
  pickle.dump(fig_data,open(fullfn,'wb'))
