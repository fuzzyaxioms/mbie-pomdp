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

# global color to make sure graphs cycle through colors
global_colors = ['k', 'b', 'r', 'g', 'c', 'm' ,'y']
global_color_ix = 0

def curr_color():
    return global_colors[global_color_ix]
def next_color():
    global global_color_ix
    global_color_ix = (global_color_ix + 1) % len(global_colors)
    return curr_color()

def save_color():
    global global_color_ix
    return global_color_ix

def restore_color(ix):
    global global_color_ix
    global_color_ix = ix

def smooth(arr,width):
    l = arr.shape[0]
    smoothed = np.zeros(arr.shape)
    for i in xrange(l):
        lend = min(i+width,l)
        acc = arr[i:lend].sum()
        cnt = lend - i
        smoothed[i] = acc / cnt
    return smoothed

def smooth_filter(v, width=100):
    # average from width back up to current ix
    # does all rows in parallel
    length = v.shape[1]
    smoothed = np.zeros(v.shape)
    for i in xrange(length):
        left_idx = max(0, i - width + 1)
        smoothed[:,i] = np.sum(v[:,left_idx:(i+1)],axis=1) / (i - left_idx + 1)
    return smoothed

def load_runs(infilebase, num_reps, epochs):
    ys = np.zeros((num_reps,epochs))
    for i in range(num_reps):
        ys[i,:] = np.loadtxt(infilebase + str(i) + '.txt')[:epochs]
    return ys

def graph_smoothed_matrix(ys, lbl, cm=False):
    ys = smooth_filter(ys, width=100)
    ys_mean = np.mean(ys, axis=0)
    ys_sem = sps.sem(ys, axis=0) *1.96 # 1.96 for 95% confidence
#    plt.figure(figsize=(6,3), dpi=300)
#    plt.figure()
#    ax = plt.axes([0., 0., 1., 1.])
#    ax.set_xscale('log')
    
    if not cm:
        ys_final = ys_mean
    else:
        ys_final = ys.cumsum()
    xs = np.arange(0, ys.shape[1])
    plt.plot(xs, ys_final, label=lbl, color=next_color())
#    plt.plot(xs, rs_final, xs, rs_final-rs_sem, xs, rs_final+rs_sem, label=lbl)
    rand_start = np.random.randint(0, 7) * 5
    skips = xs[rand_start::75]
    ys_skips = ys_final[rand_start::75]
    yerrs = ys_sem[rand_start::75]
    plt.errorbar(skips, ys_skips, yerr=yerrs, marker='.', ls=' ', color=curr_color())
#    plt.xlim(xmin=200,xmax=epochs)
#    plt.ylim(ymax=0.5, ymin=-0.5)
#    plt.xlabel('Time step')
#    plt.ylabel('Immediate Reward')
#    plt.legend(loc='lower right',prop={'size':6})
    
#    plt.savefig(outfile, bbox_inches='tight', dpi=300)

def graph_rewards(infilebase, num_reps, epochs, lbl, cm=False):
    ys = load_runs(infilebase, num_reps, epochs)
    graph_smoothed_matrix(ys, lbl, cm)


def graph_actions_dist(infilebase, num_reps, epochs, lbl):
    xs = np.arange(0, epochs)
    ys = np.zeros((num_reps,epochs))
    for i in range(num_reps):
        ys[i,:] = np.loadtxt(infilebase + str(i) + '.txt')[:epochs]
    
    xs_all = np.tile(xs, (ys.shape[0],))
    ys_all = ys.flatten('C')
#    x_noise = (np.random.rand(*xs_all.shape) - 0.5) * 5
#    y_noise = (np.random.rand(*ys_all.shape) - 0.5) * 0.07
    y_noise = np.random.randn(*ys_all.shape) * 0.05
    plt.plot(xs_all, ys_all+y_noise, color=next_color(), ls='',marker=',',alpha=0.3,label=lbl)
    plt.ylim(ymin=0,ymax=1)

def graph_rsa(infilebase, num_reps, lbl, pick=[1,0,0,0,0,0]):
    pick_ix = transpose(np.matrix(pick))
    reps = [None] * num_reps
    for i in range(num_reps):
        tmp = np.matrix(np.loadtxt(infilebase + str(i) + '.txt'))
        # average the ones that are picked
        reps[i] = np.squeeze(np.array(tmp * pick_ix)) / np.sum(pick)
        epochs = reps[i].shape[0]
    ys_all = np.concatenate(reps)
    xs_all = np.tile(np.arange(0, epochs), num_reps)
    
    x_noise = np.random.randn(*xs_all.shape) * 0.1
    y_noise = (np.random.randn(*ys_all.shape))
    plt.plot(xs_all+x_noise, ys_all+y_noise, color=next_color(), ls='',marker='.',alpha=0.2,label=lbl)
    
        
    
#    xs_all = np.tile(xs, (ys.shape[0],))
#    ys_all = ys.flatten('C')
#    y_noise = np.random.randn(*ys_all.shape) * 0.05
#    plt.plot(xs_all, ys_all+y_noise, color=next_color(), ls='',marker=',',alpha=0.3,label=lbl)
#    plt.ylim(ymin=0,ymax=1)

def graph_mat(infile, lbl, cm=False):
    plotdata = sio.loadmat(infile)['plotdata']
    xs = plotdata[0,0][0]
    ys = np.asarray(plotdata[0,1], dtype=np.float)
    ys = smooth_filter(ys,width=100)
    
    ys_mean = np.mean(ys, axis=0)
    ys_sem = sps.sem(ys, axis=0) *1.96 # 1.96 for 95% confidence
    if cm:
        diffs = np.diff(xs)
        diffs = np.insert(diffs, 0, xs[0])
        ys_mean = (diffs*ys_mean).cumsum()
    else:
#        ys_mean = smooth_filter(ys_mean[np.newaxis,:],width=100)[0]
        pass
#    plt.plot(xs, ys_mean, xs, ys_mean-ys_sem, xs, ys_mean+ys_sem, label=lbl)
    plt.plot(xs, ys_mean, label=lbl, color=next_color())
    rand_start = np.random.randint(0, 7) * 5
    skips = xs[rand_start::75]
    ys_skips = ys_mean[rand_start::75]
    yerrs = ys_sem[rand_start::75]
    plt.errorbar(skips, ys_skips, yerr=yerrs, marker='.', ls=' ', color=curr_color())

def graph_mat_dist(infile, lbl, cm=False):
    plotdata = sio.loadmat(infile)['plotdata']
    xs = plotdata[0,0][0]
    ys = plotdata[0,1]
    xs_all = np.tile(xs, (ys.shape[0],))
    ys_all = ys.flatten('C')
#    x_noise = (np.random.rand(*xs_all.shape) - 0.5) * 5
#    y_noise = (np.random.rand(*ys_all.shape) - 0.5) * 0.07
    y_noise = np.random.randn(*ys_all.shape) * 0.05
    plt.plot(xs_all, ys_all+y_noise, color=next_color(), ls='',marker=',', alpha=0.3,label=lbl)
    plt.ylim(ymin=0,ymax=1)

def show_2sensor_rewards():
    plt.figure()
    plt.plot([0, 40*50],[1.80852, 1.80852], label='0.9') # acc=0.9
    plt.plot([0, 40*50],[1.08381, 1.08381], label='0.85') # acc=0.85
#    graph_rewards('emkeep_start100/2sensortiger8590_ci0.0_episodic_rewards_everystep', 20, eps*50, 'em ci 0.0', cm=False)
    graph_rewards('emkeep_start100/2sensortiger8590_ci0.1_episodic_rewards_everystep', 20, 40*50, 'em ci 0.1', cm=False)
#    graph_rewards('emkeep_start100/2sensortiger8590_ci0.1_sample_episodic_rewards_everystep', 20, eps*50, 'em ci 0.1 sample 20', cm=False)
#    graph_rewards('emkeep_start100/2sensortiger8590_ci0.3_episodic_rewards_everystep', 20, eps*50, 'em ci 0.3', cm=False)
#    graph_rewards('emkeep_start100/2sensortiger8590_ci0.5_episodic_rewards_everystep', 20, eps*50, 'em ci 0.5', cm=False)
#    graph_rewards('emkeep_start100/2sensortiger8590_ci1.0_episodic_rewards_everystep', 20, eps*50, 'em ci 1.0', cm=False)
#    graph_rewards('emkeep_start100/2sensortiger8590_ci1.0_sample_episodic_rewards_everystep', 20, eps*50, 'em ci 1.0 sample 20', cm=False)
#    graph_rewards('emkeep_start0/2sensortiger8590_ci1.0_episodic_rewards_everystep', 20, eps*50, 'em ci 1.0 no rand', cm=False)
    
#    graph_mat('iPOMDP/pomdp-util/simulation_train_2sensortiger_ffbs_epsilon_greedy_.mat', 'epsilon_greedy', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_train_2sensortiger_ffbs_weighted_stochastic_.mat', 'weighted_stochastic', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_2sensortiger_ffbs_beb_.mat', 'beb', cm=False)
#    plt.ylim([-5.0, 2.0])
    plt.legend(loc='lower right')
    plt.xlabel('Time step')
    plt.ylabel('Immediate Reward')
#    plt.plot([0, eps*50],[0.6, 0.6]) # for acc=0.8
    pass

def show_tworoom_rewards():
    plt.figure()
    plt.plot([0, 2000],[0.72, 0.72], label='Optimal') # for tworoom
#    graph_rewards('emkeep_s100u100e2000/tworoom_ci0.2_rand100_rewards', 100, 2000, 'ci 0.2', cm=False)
#    graph_rewards('emkeep_s100u100e2000/tworoom_ci2.0_rand100_rewards', 100, 2000, 'ci 0.5', cm=False)
#    graph_rewards('emkeep_s100u100e2000/tworoom_ci2.0_sample_episodic_rewards_everystep', 100, 2000, 'ci 2 ', cm=False)
#    graph_rewards('emkeep_s100u100e2000/tworoom_ci1.0_sampling_rewards', 100, 2000, 'ci 1 sampling ', cm=False)
    graph_rewards('emkeep_s100u100e2000/tworoom_ci0.30_r_ci1.00_sampling_no_opt_ro_rand100_rewards', 100, 2000, 'ci 0.3 r_ci 1.0 no opt ro sampling', cm=False)
    graph_rewards('emkeep_s100u100e2000/tworoom_ci0.30_r_ci1.00_sampling_rand100_rewards', 100, 2000, 'ci 0.3 r_ci 1.0 sampling', cm=False)
    graph_rewards('emkeep_s100u100e2000/tworoom_ci0.30_r_ci1.00_sampling_spl_opt_ro_rand100_rewards', 100, 2000, 'ci 0.3 r_ci 1.0 sampling spl opt ro', cm=False)
    
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_epsilon_greedy_keep_s100u100e2000.mat', 'PLUS', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_weighted_stochastic_keep_s100u100e2000.mat', 'Thompson Sampling', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_boss_keep_s100u100e2000d0.mat', 'boss d0', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_boss_keep_s100u100e2000d1.mat', 'boss d1', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_boss_keep_s100u100e2000d2.mat', 'boss d2', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_boss_keep_s100u100e2000d3.mat', 'boss d3', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_boss_keep_s100u100e2000.mat', 'boss', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_beb_keep_s100u100e2000d0.mat', 'beb d0', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_beb_keep_s100u100e2000d1.mat', 'beb d1', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_beb_keep_s100u100e2000d2.mat', 'beb d2', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_beb_keep_s100u100e2000d3.mat', 'beb d3', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_beb_keep_s100u100e2000d4.mat', 'beb d4', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_stochastic_forward_search_keep_s100u100e2000.mat', 'stochastic forward search d4', cm=False)
    
    plt.ylim([0.6, 0.75])
    plt.xlim([0, 40*50])
    plt.legend(loc='lower right')
    plt.xlabel('Time step')
    plt.ylabel('Immediate Reward')
    pass

def show_tworoom_actions_dist():
    state = save_color()    
    
    plt.figure()
        
    restore_color(state)
    plt.subplot(1,3,1)
    plt.title('ci 0.3')
    graph_actions_dist('emkeep_s100u100e2000/tworoom_ci0.30_r_ci1.00_sampling_rand100_actions', 100, 2000, '')
    plt.xlabel('Time step')
    plt.ylabel('Frequency of not taking the first action')
    
    restore_color(state)
    plt.subplot(1,3,2)
    plt.title('ci 0.3 spl opt ro')
    graph_actions_dist('emkeep_s100u100e2000/tworoom_ci0.30_r_ci1.00_sampling_spl_opt_ro_rand100_actions', 100, 2000, '')
    plt.xlabel('Time step')
    plt.ylabel('Frequency of not taking the first action')
    
    restore_color(state)
    plt.subplot(1,3,3)
    plt.title('ci 0.3 no opt ro')
    graph_actions_dist('emkeep_s100u100e2000/tworoom_ci0.30_r_ci1.00_sampling_no_opt_ro_rand100_actions', 100, 2000, '')
    plt.xlabel('Time step')
    plt.ylabel('Frequency of not taking the first action')
    
    pass

def show_tiger_rewards():
    plt.figure()
    plt.plot([0, 20*50],[1.08381, 1.08381], color=next_color(), label='Optimal') # acc=0.85
#    graph_rewards('emkeep_s100u100e1000/tiger_ci1.00_r_ci1.00_no_opt_ro_no_weissman_rand100_rewards', 100, 1000, 'ci 1.0 r_ci 1.0 no opt ro no weismann', cm=False)
#    graph_rewards('emkeep_s100u100e1000/tiger_ci1.00_r_ci1.00_no_weissman_rand100_rewards', 100, 1000, 'ci 1.0 r_ci 1.0 no weismann', cm=False)
#    graph_rewards('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_no_opt_ro_rand100_rewards', 100, 1000, 'ci 0.3 r_ci 1.0 no opt ro', cm=False)
    graph_rewards('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_no_opt_ro_rand100_rewards', 100, 1000, 'ci 0.3 r_ci 1.0 no opt ro sampling', cm=False)
#    graph_rewards('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_rand100_rewards', 100, 1000, 'ci 0.3 r_ci 1.0', cm=False)
    graph_rewards('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_rand100_rewards', 100, 1000, 'ci 0.3 r_ci 1.0 sampling', cm=False)
    graph_rewards('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_spl_opt_ro_rand100_rewards', 100, 1000, 'ci 0.3 r_ci 1.0 sampling spl opt ro', cm=False)
    
    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_epsilon_greedy_keep_s100u100e1000.mat', 'PLUS', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_weighted_stochastic_keep_s100u100e1000.mat', 'Thompson Sampling', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_stochastic_forward_search_keep_s100u100e1000d4.mat', 'Stochastic Forward Search', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_boss_keep_s100u100e1000d0.mat', 'boss d0', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_boss_keep_s100u100e1000d1.mat', 'boss d1', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_boss_keep_s100u100e1000d2.mat', 'boss d2', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_boss_keep_s100u100e1000d3.mat', 'boss d3', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_boss_keep_s100u100e1000.mat', 'boss d4', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_beb_keep_s100u100e1000d0.mat', 'beb d0', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_beb_keep_s100u100e1000d1.mat', 'beb d1', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_beb_keep_s100u100e1000d2.mat', 'beb d2', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_beb_keep_s100u100e1000d3.mat', 'beb d3', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_beb_keep_s100u100e1000d4.mat', 'beb d4', cm=False)
    plt.ylim([-10, 1.5])
    plt.xlim(xmin=000)
    plt.legend(loc='lower right')
    plt.xlabel('Time step')
    plt.ylabel('Immediate Reward')
    pass

def show_actions():
    plt.figure()
#    graph_rewards('emkeep_start0/tworoom_ci1.0_episodic_rewards_everystep', 20, eps*50, 'em ci 1.0 no rand', cm=False)
#    graph_rewards('emkeep_start0/tworoom_ci2.0_episodic_rewards_everystep', 20, eps*50, 'em ci 2.0 no rand', cm=False)
#    graph_rewards('emkeep_every10/tworoom_ci0.5_up10_restarts50_episodic_rewards_everystep', 20, 20*50, 'em ci 0.5 every 10', cm=False)
#    graph_rewards('emkeep_every10/tworoom_ci2.0_up10_restarts50_episodic_rewards_everystep', 20, 20*50, 'em ci 2.0 every 10', cm=False)
#    graph_rewards('emkeep_every10/tworoom_ci3.0_up10_restarts50_episodic_rewards_everystep', 20, 20*50, 'em ci 3.0 every 10', cm=False)
#    graph_rewards('emkeep_every10/tworoom_ci5.0_up10_restarts50_episodic_rewards_everystep', 20, 20*50, 'em ci 5.0 every 10', cm=False)
#    graph_rewards('emkeep_s100u100e2000/tworoom_ci1.0_rewards', 100, 2000, 'em ci 1.0', cm=False)
#    graph_rewards('emkeep_s100u100e2000/tworoom_ci1.0_sampling_rewards', 100, 2000, 'em ci 1.0 sample', cm=False)
    graph_actions_dist('emkeep_s100u100e2000/tworoom_ci1.0_sampling_actions', 100, 2000, 'em ci 1.0 sample')
#    graph_actions_dist('emkeep_s100u100e2000/tworoom_ci1.0_sampling_alt_actions', 100, 2000, 'em ci 1.0 sample opt r')
    
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_epsilon_greedy_keep_s100u100e2000d4_ar.mat', 'PLUS', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_weighted_stochastic_keep_s100u100e2000d4_ar.mat', 'Thompson Sampling', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_boss_keep_s100u100e2000d0_ar.mat', 'boss d0', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_boss_keep_s100u100e2000d1_ar.mat', 'boss d1', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_boss_keep_s100u100e2000d2_ar.mat', 'boss d2', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_boss_keep_s100u100e2000d3_ar.mat', 'boss d3', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_boss_keep_s100u100e2000d4_ar.mat', 'boss d4', cm=False)
#    graph_mat_dist('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_beb_keep_s100u100e2000d0_ar.mat', 'beb d0', cm=False)
#    graph_mat_dist('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_beb_keep_s100u100e2000d1_ar.mat', 'beb d1', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_beb_keep_s100u100e2000d2_ar.mat', 'beb d2', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_beb_keep_s100u100e2000d3_ar.mat', 'beb d3', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_beb_keep_s100u100e2000d4_ar.mat', 'beb d4', cm=False)
#    graph_mat('iPOMDP/pomdp-util/simulation_train_tworoom_ffbs_stochastic_forward_search_keep_s100u100e2000d4_ar.mat', 'stochastic forward search d4', cm=False)
    plt.legend(loc='lower right')
    plt.xlabel('Time step')
    plt.ylabel('Frequency of not taking the first action')
    pass

def show_tiger_actions_dist():
    state = save_color()    
    
    plt.figure()
        
    restore_color(state)
    plt.subplot(1,3,1)
    plt.title('ci 0.3')
    graph_actions_dist('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_rand100_actions', 100, 1000, '')
    plt.xlabel('Time step')
    plt.ylabel('Frequency of not taking the first action')
    
    restore_color(state)
    plt.subplot(1,3,2)
    plt.title('ci 0.3 spl opt ro')
    graph_actions_dist('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_spl_opt_ro_rand100_actions', 100, 1000, '')
    plt.xlabel('Time step')
    plt.ylabel('Frequency of not taking the first action')
    
    restore_color(state)
    plt.subplot(1,3,3)
    plt.title('ci 0.3 no opt ro')
    graph_actions_dist('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_no_opt_ro_rand100_actions', 100, 1000, '')
    plt.xlabel('Time step')
    plt.ylabel('Frequency of not taking the first action')
    
    pass

def show_rsa():
    state = save_color()    
    
    plt.figure()
    
    plt.subplot(1,3,1)
    restore_color(state)
    plt.title('ci 0.3')
    graph_rsa('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_rand100_rsa', 100, 'listen', pick=(1,1,0,0,0,0))
    graph_rsa('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_rand100_rsa', 100, 'correct', pick=(0,0,1,0,0,1))
    graph_rsa('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_rand100_rsa', 100, 'wrong', pick=(0,0,0,1,1,0))
    plt.xlabel('Model Update')
    plt.ylabel('Upper bound on the reward')
    plt.ylim(-100,20)
    
    plt.subplot(1,3,2)
    restore_color(state)
    plt.title('ci 0.3 spl opt ro')
    graph_rsa('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_spl_opt_ro_rand100_rsa', 100, 'listen', pick=(1,1,0,0,0,0))
    graph_rsa('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_spl_opt_ro_rand100_rsa', 100, 'correct', pick=(0,0,1,0,0,1))
    graph_rsa('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_spl_opt_ro_rand100_rsa', 100, 'wrong', pick=(0,0,0,1,1,0))
    plt.xlabel('Model Update')
    plt.ylabel('Upper bound on the reward')
    plt.ylim(-100,20)
    
    plt.subplot(1,3,3)
    restore_color(state)
    plt.title('ci 0.3 no opt ro')
    graph_rsa('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_no_opt_ro_rand100_rsa', 100, 'listen', pick=(1,1,0,0,0,0))
    graph_rsa('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_no_opt_ro_rand100_rsa', 100, 'correct', pick=(0,0,1,0,0,1))
    graph_rsa('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_no_opt_ro_rand100_rsa', 100, 'wrong', pick=(0,0,0,1,1,0))
    plt.xlabel('Model Update')
    plt.ylabel('Upper bound on the reward')
    plt.ylim(-100,20)

    pass

def tiger_keep_optimistic():
    def graph_good(filebase, lbl, threshold):
        pick = [0,0,1,0,0,1]
        pick_ix = transpose(np.matrix(pick))
        num_reps = 100
        num_ups = 10
        reps = [None] * num_reps
        good_ix = []
        for i in range(num_reps):
            tmp = np.matrix(np.loadtxt(filebase + '_rsa' + str(i) + '.txt'))
            # average the ones that are picked
            reps[i] = np.squeeze(np.array(tmp * pick_ix)) / np.sum(pick)
            epochs = reps[i].shape[0]
            if np.all(reps[i][1:] >= threshold):
                good_ix.append(i)
        print(len(good_ix))
    #    reps = np.array(reps)
    #    ys_all = np.concatenate(reps)
    #    xs_all = np.tile(np.arange(0, epochs), num_reps)
    #    
    #    x_noise = np.random.randn(*xs_all.shape) * 0.1
    #    y_noise = (np.random.randn(*ys_all.shape))
    #    plt.figure()
    #    plt.plot(xs_all+x_noise, ys_all+y_noise, color=next_color(), ls='',marker='.',alpha=0.2,label='')
    #    plt.ylim(-100,20)
    #    
    #    num_good_reps = len(good_ix)
    #    ys_all_good = np.concatenate(reps[np.array(good_ix)][:])
    #    xs_all_good = np.tile(np.arange(0, epochs), num_good_reps)
    #    
    #    x_noise = np.random.randn(*xs_all_good.shape) * 0.1
    #    y_noise = (np.random.randn(*ys_all_good.shape))
    #    plt.figure()
    #    plt.plot(xs_all_good+x_noise, ys_all_good+y_noise, color=next_color(), ls='',marker='.',alpha=0.2,label='')
    #    plt.ylim(-100,20)
        epochs = 1000
        num_reps = 100
        
        infilebase = filebase + '_rewards'
        ys = load_runs(infilebase, num_reps, epochs)[good_ix,:]
        graph_smoothed_matrix(ys, lbl)

    plt.figure()
    
    plt.subplot(3,1,1)
    plt.plot([0, 20*50],[1.08381, 1.08381], color=next_color(), label='Optimal') # acc=0.85
    graph_good('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_no_opt_ro_rand100', 'thresh 9', 9)
    graph_good('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_no_opt_ro_rand100', 'thresh 7', 7)
    graph_rewards('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_no_opt_ro_rand100_rewards', 100, 1000, 'orig', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_beb_keep_s100u100e1000d4.mat', 'beb d4', cm=False)
    plt.title('no opt ro')
    plt.ylim([-10, 1.5])
    plt.xlim(xmin=000)
    plt.legend(loc='lower right')
    plt.ylabel('Immediate Reward')
    
    plt.subplot(3,1,2)
    plt.plot([0, 20*50],[1.08381, 1.08381], color=next_color(), label='Optimal') # acc=0.85
    graph_good('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_rand100', 'thresh 9', 9)
    graph_good('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_rand100', 'thresh 7', 7)
    graph_rewards('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_rand100_rewards', 100, 1000, 'orig', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_beb_keep_s100u100e1000d4.mat', 'beb d4', cm=False)
    plt.title('reg')
    plt.ylim([-10, 1.5])
    plt.xlim(xmin=000)
    plt.legend(loc='lower right')
    plt.ylabel('Immediate Reward')
    
    plt.subplot(3,1,3)
    plt.plot([0, 20*50],[1.08381, 1.08381], color=next_color(), label='Optimal') # acc=0.85
    graph_good('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_spl_opt_ro_rand100', 'thresh 9', 9)
    graph_good('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_spl_opt_ro_rand100', 'thresh 7', 7)
    graph_rewards('emkeep_s100u100e1000/tiger_ci0.30_r_ci1.00_sampling_spl_opt_ro_rand100_rewards', 100, 1000, 'orig', cm=False)
    graph_mat('iPOMDP/pomdp-util/simulation_train_tiger_ffbs_beb_keep_s100u100e1000d4.mat', 'beb d4', cm=False)
    plt.title('spl opt ro')
    plt.ylim([-10, 1.5])
    plt.xlim(xmin=000)
    plt.legend(loc='lower right')
    plt.ylabel('Immediate Reward')

if __name__ == '__main__':
    show_tworoom_rewards()
#    show_tworoom_actions_dist()
    
#    show_tiger_rewards()
#    show_tiger_actions_dist()
#    show_rsa()
#    tiger_keep_optimistic()

#    show_tworoom_rewards()
    pass

# ================================ old stuffs =================================
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
