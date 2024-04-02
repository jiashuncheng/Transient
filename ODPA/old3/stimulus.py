import numpy as np
import matplotlib.pyplot as plt
from parameters import *
import pickle

#   # 固定每次的气味编码


class Stimulus:

    def __init__(self):

        # generate tuning functions
        self.motion_tuning, self.fix_tuning, self.rule_tuning = self.create_tuning_functions()

        # with open('./savedir/sample.pkl', 'wb') as handle:
        #     pickle.dump(self.motion_tuning, handle)
        # with open('./savedir/sample.pkl', 'rb') as handle:
        #     self.motion_tuning = pickle.load(handle)


    def generate_trial(self, test_mode = False, set_rule = None):
        trial_info = self.generate_basic_trial(test_mode, set_rule)
        trial_info['neural_input'] = np.maximum(0., trial_info['neural_input'])

        return trial_info
        


    def generate_basic_trial(self, test_mode, set_rule = None):
    # def generate_basic_trial(self, test_mode, set_rule = None):
        

        """
        Generate a delayed matching task
        Goal is to determine whether the sample stimulus, possibly manipulated by a rule, is
        identicical to a test stimulus
        Sample and test stimuli are separated by a delay
        """
        
        # range of variable delay, in time steps
        var_delay_max = par['variable_delay_max']//par['dt'] # 300/10=30

        # duration of mask after test onset
        mask_duration = par['mask_duration']//par['dt']  # 50/10=5
        max_delay = args.max_delay
        num_time_steps = (par['fix_time']+par['sample_time']+par['test_time']+max_delay)//par['dt']
        trial_info = {'desired_output'  :  np.zeros((num_time_steps, par['batch_size'], par['n_output']),dtype=np.float32), # [250, 1024, 3]
                      'train_mask'      :  np.ones((num_time_steps, par['batch_size']),dtype=np.float32),  # [250, 1024]
                      'sample'          :  np.zeros((par['batch_size']),dtype=np.int8), # [1024]
                      'test'            :  np.zeros((par['batch_size']),dtype=np.int8),
                      'rule'            :  np.zeros((par['batch_size']),dtype=np.int8),
                      'match'           :  np.zeros((par['batch_size']),dtype=np.int8),
                      'catch'           :  np.zeros((par['batch_size']),dtype=np.int8),
                      'probe'           :  np.zeros((par['batch_size']),dtype=np.int8),
                      'delay'           :  np.zeros((par['batch_size']),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(num_time_steps, par['batch_size'], par['n_input']))}


        # set to mask equal to zero during the dead time
        trial_info['train_mask'][par['dead_time_rng'], :] = 0

        for t in range(par['batch_size']):

            """
            Generate trial paramaters
            """
            if args.mode == 'train':
                sample_dir = np.random.randint(par['num_motion_dirs']) # Odors中选一个
                test_RF = np.random.choice([1,2]) if par['trial_type'] == 'location_DMS' else 0  # 在DMS任务中没有用
            else:
                sample_dir = 1
                test_RF = 0
            
            rule = np.random.randint(par['num_rules']) if set_rule is None else set_rule  # 默认是0

            if par['trial_type'] == 'DMC' or (par['trial_type'] == 'DMS+DMC' and rule == 1) or (par['trial_type'] == 'DMS+DMRS+DMC' and rule == 2):
                # for DMS+DMC trial type, rule 0 will be DMS, and rule 1 will be DMC
                current_trial_DMC = True
            else:
                #this way
                current_trial_DMC = False
            if args.mode == 'train':
                match = np.random.randint(2)   # match 任意选择0/1
            else:
                match = 1
            
            catch = np.random.rand() < par['catch_trial_pct']

            """
            Generate trial paramaters, which can vary given the rule
            """
            if par['num_rules'] == 1:
                match_rotation = int(par['num_motion_dirs']*par['rotation_match']/360)  # 8*0/360 = 0
            else:
                match_rotation = int(par['num_motion_dirs']*par['rotation_match'][rule]/360)

            """
            Determine the delay time for this trial
            The total trial length is kept constant, so a shorter delay implies a longer test stimulus
            """
            if par['random']:
                par['delay_time'] = np.random.choice([3000,4000,5000,6000])
            sample_oneset = (par['dead_time']+par['fix_time'])//par['dt'] # lhx
            delay_oneset = (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt'] # lhx
            test_onset = (par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'])//par['dt']  # 得到test的开始时间
            end_onset = (par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time'])//par['dt']  # 得到test的开始时间
                
            
            fix_time_rng =  range(sample_oneset) # lhx
            sample_time_rng = range(sample_oneset, delay_oneset) # lhx
            del_time_rng =  range(delay_oneset, test_onset)
            test_time_rng =  range(test_onset, end_onset) 
            pad_time_rng = range(end_onset, num_time_steps) 

            # trial_info['train_mask'][test_onset:test_onset+mask_duration, t] = 0

            """
            Generate the sample and test stimuli based on the rule
            """
            # DMC
            if not test_mode:
                if current_trial_DMC: # categorize between two equal size, contiguous zones
                    sample_cat = np.floor(sample_dir/(par['num_motion_dirs']/2))
                    if match == 1: # match trial
                        # do not use sample_dir as a match test stimulus
                        dir0 = int(sample_cat*par['num_motion_dirs']//2)
                        dir1 = int(par['num_motion_dirs']//2 + sample_cat*par['num_motion_dirs']//2)
                        possible_dirs = list(range(dir0, dir1))
                        test_dir = possible_dirs[np.random.randint(len(possible_dirs))]
                    else:
                        test_dir = sample_cat*(par['num_motion_dirs']//2) + np.random.randint(par['num_motion_dirs']//2)
                        test_dir = np.int_((test_dir+par['num_motion_dirs']//2)%par['num_motion_dirs'])
                # DMS or DMRS
                else:
                #这里
                    matching_dir = (sample_dir + match_rotation)%par['num_motion_dirs']
                    if match == 1: # match trial
                        test_dir = matching_dir
                    else:
                        # np.random.seed(0) #2个气味下 随机种子固定不固定没区
                        possible_dirs = np.setdiff1d(list(range(par['num_motion_dirs'])), matching_dir) # 找到2个数组中集合元素的差异
                        test_dir = possible_dirs[np.random.randint(len(possible_dirs))]   #如果不匹配则随机选择一个作为test
            # else:
            #     test_dir = np.random.randint(par['num_motion_dirs'])
            #     # this next part only working for DMS, DMRS tasks
            #     matching_dir = (sample_dir + match_rotation)%par['num_motion_dirs']
            #     match = 1 if test_dir == matching_dir else 0

            """
            Calculate neural input based on sample, tests, fixation, rule, and probe
            """
            # SAMPLE stimulus
            # with open('./savedir/sample_input.pkl', 'wb') as handle:
            #     pickle.dump(trial_info['neural_input'], handle)
            # 用于相同的初始化
            # with open('./savedir/sample_input.pkl', 'rb') as handle:
            #     trial_info['neural_input'][fix_time_rng, t, :] = pickle.load(handle)[fix_time_rng, t, :]
            trial_info['neural_input'][sample_time_rng, t, :] += np.reshape(self.motion_tuning[:, 0, sample_dir],(1,-1))

            # TEST stimulus
            if not catch:
                trial_info['neural_input'][test_time_rng, t, :] += np.reshape(self.motion_tuning[:, test_RF, test_dir],(1,-1))
                # trial_info['neural_input'][label_time_rng, t, :] = 0.
                trial_info['neural_input'][pad_time_rng, t, :] = 0.
                #修改用来测试对sample期间气味的记忆
            # trial_info['neural_input'][test_time_rng, t, :] += np.reshape(self.motion_tuning[:, 0, sample_dir],(1,-1))

            # FIXATION cue
            # if par['num_fix_tuned'] > 0:
            #     trial_info['neural_input'][fix_time_rng, t] += np.reshape(self.fix_tuning[:,0],(-1,1))

            # # RULE CUE
            # if par['num_rules']> 1 and par['num_rule_tuned'] > 0:
            #     trial_info['neural_input'][par['rule_time_rng'][0], t, :] += np.reshape(self.rule_tuning[:,rule],(1,-1))

            """
            Determine the desired network output response
            """
            trial_info['desired_output'][fix_time_rng, t, 0] = 1.
            # trial_info['desired_output'][sample_time_rng, t, sample_dir+4] = 1. # lhx
            # trial_info['desired_output'][del_time_rng, t, 3] = 1. # lhx
            trial_info['desired_output'][sample_time_rng, t, 0] = 1. 
            trial_info['desired_output'][del_time_rng, t, 0] = 1. 
            # trial_info['desired_output'][test_time_rng, t, 0] = 1. 
            if not catch:
                trial_info['train_mask'][test_time_rng, t] *= par['test_cost_multiplier'] # can use a greater weight for test period if needed
                if match == 0:
                    trial_info['desired_output'][test_time_rng, t, 1] = 1.
                else:
                    trial_info['desired_output'][test_time_rng, t, 2] = 1.
            else:
                trial_info['desired_output'][test_time_rng, t, 0] = 1.
            
            # padding 期间 mask = 0
            trial_info['train_mask'][range(end_onset, num_time_steps), t] = 0.
            #测试sample期间气味记忆时长
            # trial_info['desired_output'][test_time_rng, t, sample_dir+4] = 1.

            """
            Append trial info
            """
            trial_info['sample'][t] = sample_dir
            trial_info['test'][t] = test_dir
            trial_info['rule'][t] = rule
            trial_info['catch'][t] = catch
            trial_info['match'][t] = match
            trial_info['delay'][t] = par['delay_time']//par['dt']

        return trial_info


    def create_tuning_functions(self):

        """
        Generate tuning functions for the Postle task
        """

        motion_tuning = np.zeros((par['n_input'], par['num_receptive_fields'], par['num_motion_dirs'])) # [24,1,8]
        fix_tuning = np.zeros((par['n_input'], 1)) #[24,1]
        rule_tuning = np.zeros((par['n_input'], par['num_rules'])) #[24, 1]


        # generate list of prefered directions
        # dividing neurons by 2 since two equal groups representing two modalities
        pref_dirs = np.float32(np.arange(0,360,360/(par['num_motion_tuned']//par['num_receptive_fields'])))  #(0,360,360/24)

        # generate list of possible stimulus directions
        # stim_dirs = np.float32(np.arange(0,360,360/par['num_motion_dirs'])) # (0,360,360/8)
        # theta = np.random.choice([0,30,60,90,120,150])
        theta = 0.
        stim_dirs = np.float32(np.array([0., 180.]))
        # stim_dirs = np.float32(np.array([180., 0.]))

        for n in range(par['num_motion_tuned']//par['num_receptive_fields']):  # 24
            for i in range(par['num_motion_dirs']):  # 8
                for r in range(par['num_receptive_fields']): # 1
                    if par['trial_type'] == 'distractor':
                        if n%par['num_motion_dirs'] == i:
                            motion_tuning[n,0,i] = par['tuning_height']
                    else:
                        d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
                        n_ind = n+r*par['num_motion_tuned']//par['num_receptive_fields']
                        motion_tuning[n_ind,r,i] = par['tuning_height']*np.exp(par['kappa']*d)/np.exp(par['kappa'])  # von Mises distriubution

        for n in range(par['num_fix_tuned']):
            fix_tuning[par['num_motion_tuned']+n,0] = par['tuning_height']

        for n in range(par['num_rule_tuned']):
            for i in range(par['num_rules']):
                if n%par['num_rules'] == i:
                    rule_tuning[par['num_motion_tuned']+par['num_fix_tuned']+n,i] = par['tuning_height']*par['rule_cue_multiplier']


        return motion_tuning, fix_tuning, rule_tuning


    def plot_neural_input(self, trial_info):

        # print(trial_info['desired_output'][ :, 0, :].T)
        f = plt.figure(figsize=(8,4))
        ax = f.add_subplot(1, 1, 1)
        t = np.arange(0,1000+1000+6000+1000, par['dt'])
        t -= 1000
        t0,t1,t2,t3 = np.where(t==-1000), np.where(t==0),np.where(t==1000),np.where(t==7000)
        #im = ax.imshow(trial_info['neural_input'][:,0,:].T, aspect='auto', interpolation='none')
        im = ax.imshow(trial_info['neural_input'][:,:,0], aspect='auto', interpolation='none')
        #plt.imshow(trial_info['desired_output'][:, :, 0], aspect='auto')
        ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
        ax.set_xticklabels([-1000,0,1000,7000])
        ax.set_yticks([0, 9, 18, 27])
        ax.set_yticklabels([0,90,180,270])
        f.colorbar(im,orientation='vertical')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Motion direction')
        ax.set_xlabel('Time relative to sample onset (ms)')
        ax.set_title('Motion input')
        plt.show()
        plt.savefig('stimulus.pdf', format='pdf')
