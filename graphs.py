import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.stats as sps

import matplotlib as mpl
import matplotlib.pyplot as plt

#from sklearn import linear_model

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:#
#mpl.rc('font',**{'family':'serif', 'size':8.0})
#rc('text', usetex=True)

def smooth(arr,width):
    l = arr.shape[0]
    smoothed = np.zeros(arr.shape)
    for i in xrange(l):
        lend = min(i+width,l)
        acc = arr[i:lend].sum()
        cnt = lend - i
        smoothed[i] = acc / cnt
    return smoothed

def smooth_filter(v, radius=50):
    # try to average equally on both sides, but if not enough, slide it around
    length = v.shape[0]
    smoothed = np.zeros((length,))
    for i in xrange(length):
        left_avail = i
        right_avail = length - 1 - i
        left_over = max(0,radius - right_avail)
        right_over = max(0,radius - i)
        left_idx = max(0, i - (radius + left_over))
        right_idx = min(length, i + radius + right_over)
        smoothed[i] = np.sum(v[left_idx:right_idx]) / (right_idx - left_idx)
    return smoothed

def graph_rewards(infilebase, num_reps, epochs, lbl, cm=False):
    rs = np.zeros((epochs,))
    for i in range(num_reps):
        rs += np.loadtxt(infilebase + str(i) + '.txt')[:epochs]
    rs /= num_reps
    
#    labels = [str(n) + ' MDP' for n in nums*len(infiles)]
#    colors = ['black', 'green', 'blue', 'red', 'magenta', 'cyan', 'pink', 'orange']
    
#    exs = [cdata[i*epochs : (i+1)*epochs, np.newaxis] for i,_ in enumerate(nums) for cdata in data]
    
#    plt.figure(figsize=(6,3), dpi=300)
#    plt.figure()
#    ax = plt.axes([0., 0., 1., 1.])
#    ax.set_xscale('log')
    
#    for ex, clr, lbl in zip(exs, colors, labels):
    if not cm:
        rs = smooth_filter(rs,radius=50)
    else:
        rs = rs.cumsum()
    plt.plot(rs[:], label=lbl)
#    plt.plot(smooth_filter(rs), color=clr, label=lbl)
#        plt.plot([0,epochs],[ex[-1], ex[-1]], 'k:')
    
#    plt.xlim(xmin=200,xmax=epochs)
#    plt.ylim(ymax=0.5, ymin=-0.5)
#    plt.xlabel('Time step')
#    plt.ylabel('Immediate Reward')
#    plt.legend(loc='lower right',prop={'size':6})
    
#    plt.savefig(outfile, bbox_inches='tight', dpi=300)

def graph_mat(infile, lbl, cm=False):
    plotdata = sio.loadmat(infile)['plotdata']
    xs = plotdata[0,0][0]
    ys = plotdata[0,1][0]
    if cm:
        diffs = np.diff(xs)
        diffs = np.insert(diffs, 0, xs[0])
        ys = (diffs*ys).cumsum()
    else:
        ys = smooth_filter(ys,radius=50)
    plt.plot(xs, ys, label=lbl)

if True:
    eps = 50
    
    plt.figure()
    plt.plot([0, eps*50],[1.80852, 1.80852], label='0.9') # acc=0.9
    plt.plot([0, eps*50],[1.08381, 1.08381], label='0.85') # acc=0.85
#    graph_rewards('emkeep_start100/2sensortiger8590_ci0.0_episodic_rewards_everystep', 20, eps*50, 'em ci 0.0', cm=False)
    graph_rewards('emkeep_start100/2sensortiger8590_ci0.1_episodic_rewards_everystep', 20, eps*50, 'em ci 0.1', cm=False)
#    graph_rewards('emkeep_start100/2sensortiger8590_ci0.3_episodic_rewards_everystep', 20, eps*50, 'em ci 0.3', cm=False)
#    graph_rewards('emkeep_start100/2sensortiger8590_ci0.5_episodic_rewards_everystep', 20, eps*50, 'em ci 0.5', cm=False)
    graph_rewards('emkeep_start100/2sensortiger8590_ci1.0_episodic_rewards_everystep', 20, eps*50, 'em ci 1.0', cm=False)
    graph_rewards('emkeep_start100/2sensortiger8590_ci1.0_sample_episodic_rewards_everystep', 20, eps*50, 'em ci 1.0 sample 20', cm=False)
    graph_rewards('emkeep_start100/2sensortiger8590_ci0.1_sample_episodic_rewards_everystep', 20, eps*50, 'em ci 0.1 sample 20', cm=False)
    graph_rewards('emkeep_start0/2sensortiger8590_ci1.0_episodic_rewards_everystep', 20, eps*50, 'em ci 1.0 no rand', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_train_2sensortiger_ffbs_epsilon_greedy_.mat', 'epsilon_greedy', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_rep_test_2sensortiger_ffbs_epsilon_greedy_.mat', 'epsilon_greedy ct', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_train_2sensortiger_ffbs_weighted_stochastic_.mat', 'weighted_stochastic', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_rep_test_2sensortiger_ffbs_weighted_stochastic_.mat', 'weighted_stochastic ct', cm=False)
    plt.ylim([-10.0, 2.0])
    plt.legend(loc='lower right')
    plt.xlabel('Time step')
    plt.ylabel('Immediate Reward')
#    plt.plot([0, eps*50],[0.6, 0.6]) # for acc=0.8
    
    plt.figure()
    plt.plot([0, 50*50],[0.72, 0.72]) # for tworoom
#    graph_rewards('emkeep_start0/tworoom_ci1.0_episodic_rewards_everystep', 20, eps*50, 'em ci 1.0 no rand', cm=False)
#    graph_rewards('emkeep_start0/tworoom_ci2.0_episodic_rewards_everystep', 20, eps*50, 'em ci 2.0 no rand', cm=False)
    graph_rewards('emkeep_start100/tworoom_ci1.0_episodic_rewards_everystep', 20, eps*50, 'em ci 1.0', cm=False)
#    graph_rewards('emkeep_start100/tworoom_ci2.0_episodic_rewards_everystep', 20, eps*50, 'em ci 2.0', cm=False)
    graph_rewards('emkeep_start100/tworoom_ci3.0_episodic_rewards_everystep', 20, eps*50, 'em ci 3.0', cm=False)
    graph_rewards('emkeep_start100/tworoom_ci1.0_sample_episodic_rewards_everystep', 20, eps*50, 'em ci 1.0 sample 20', cm=False)
    graph_rewards('emkeep_start100/tworoom_ci3.0_sample_episodic_rewards_everystep', 20, eps*50, 'em ci 3.0 sample 20', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_epsilon_greedy_.mat', 'epsilon_greedy', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_rep_test_tworoom_ffbs_epsilon_greedy_.mat', 'epsilon_greedy ct', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_weighted_stochastic_.mat', 'weighted_stochastic', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_rep_test_tworoom_ffbs_weighted_stochastic_.mat', 'weighted_stochastic ct', cm=False)

    plt.legend(loc='lower right')
    plt.xlabel('Time step')
    plt.ylabel('Immediate Reward')
    pass

# old stuffs
if False:
    data = np.loadtxt('../optimistic_rewards_eps.txt')[:-1]
    data2 = smooth(np.loadtxt('../optimistic_rewards_eps.txt')[:-1],10)
    plt.figure()
    plt.plot(data, '-g', label='Optimistic')
    plt.plot(data2, '-k', label='Smoothed Optimistic')
#    plt.legend(loc='lower right',prop={'size':8})
    plt.xlabel('Time step')
    plt.ylabel('Avg. Reward')
    plt.xlim(xmin=0,xmax=10000)
    plt.ylim(ymin=-75,ymax=10)
#    plt.savefig('rewards.png', bbox_inches='tight', dpi=300)

# plot reward results
if False:
    horizon=5100
    data = np.loadtxt('optimistic_rewards_h1100.txt')[:1100]
    data2 = np.loadtxt('mean_rewards_h1100.txt')[:1100]
    data3 = np.loadtxt('mean_rewards_f100_h5100.txt')[:horizon]
    data4 = np.loadtxt('mean_rewards_f200_h5100.txt')[:horizon]
    data5 = np.loadtxt('mean_rewards_f300_h5100.txt')[:horizon]
#    data3 = np.loadtxt('mean_rewards_f10_er_bem_easy_20p_50rep.txt')[:-1]
    plt.figure()
#    plt.figure(figsize=(3.25,1.0), dpi=300)
#    ax = plt.axes([0., 0., 1., 1.])
    plt.plot(data, '-k', label='Optimistic')
    plt.plot(data2, '--k', label='Non-Optimistic')
    plt.plot(data3, '-b', label='Egreedy f100')
    plt.plot(data4, '-r', label='Egreedy f200')
    plt.plot(data5, '-k', label='Egreedy f300')
    plt.legend(loc='lower right',prop={'size':8})
    plt.xlabel('Time step')
    plt.ylabel('Avg. Reward')
#    plt.xlim(xmin=1,xmax=100)
    plt.ylim(ymin=0,ymax=1)
#    plt.savefig('rewards.png', bbox_inches='tight', dpi=300)

# plot EM
if False:
    start = 000
    data = np.loadtxt('l2_out_tr.txt')[start:]
    data2 = np.loadtxt('linf_out_tr.txt')[start:]
    data3 = np.loadtxt('l2_out_err_tr.txt')[start:]
    data4 = np.loadtxt('linf_out_err_tr.txt')[start:]
    
    data5 = np.loadtxt('l2_out_ro.txt')[start:]
    data6 = np.loadtxt('linf_out_ro.txt')[start:]
    data7 = np.loadtxt('l2_out_ro_err.txt')[start:]
    data8 = np.loadtxt('linf_out_ro_err.txt')[start:]    
    
    data9 = np.loadtxt('l2_out.txt')[start:]
    data10 = np.loadtxt('linf_out.txt')[start:]
#    data11 = np.loadtxt('l2_out_err.txt')[start:]
#    data12 = np.loadtxt('linf_out_err.txt')[start:]    
    
    plt.figure()
#    plt.figure(figsize=(3.25,1.0), dpi=300)
#    ax = plt.axes([0., 0., 1., 1.])
    plt.plot(data, '-r', label='l2')
    plt.plot(data2, '-b', label='linf')
    plt.plot(data3, '--r', label='l2 ci')
    plt.plot(data4, '--b', label='linf ci')
    
    plt.plot(data5, '-m', label='l2 ro')
    plt.plot(data6, '-g', label='linf ro')
    plt.plot(data7, '--m', label='l2 ro ci')
    plt.plot(data8, '--g', label='linf ro ci') 
    
    plt.plot(data9, '-k', label='l2 no r')
    plt.plot(data10, '-c', label='linf no r')
#    plt.plot(data11, '--k', label='l2 no r c1i')
#    plt.plot(data12, '--c', label='linf no r ci')
    
    plt.legend(loc='upper right',prop={'size':8})
    plt.xlabel('Time step')
    plt.ylabel('Avg. Dist')
#    plt.xlim(xmin=1,xmax=100)
#    plt.ylim(ymin=0,ymax=11)
#    plt.savefig('rewards.png', bbox_inches='tight', dpi=300)
