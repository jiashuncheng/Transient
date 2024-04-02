import sys
import time, datetime
from collections import deque
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.huggingface.huggingface_utils import generate_model_card, generate_replay_video, push_to_hf
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log

import os
import shutil
from glob import glob
from scipy.stats import entropy

def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))

def normalization1(data):
#     data = data - np.mean(data, axis=1)[:,np.newaxis]
    print(data.max(), data.min())
    data = data - np.min(data)#, axis=1)[:,np.newaxis]
    data = data/(np.max(np.abs(data), axis=1)[:,np.newaxis]+0.00001)
    return data

def normalization2(data):
    data = data - np.mean(data, axis=1)[:,np.newaxis]
    data = data/(np.max(np.abs(data), axis=1)[:,np.newaxis]+0.00001)
    return data


def mean_var(A):
    mean = np.mean(A)
    var = np.var(A)
    std = np.std(A)
    return mean, var, std

def c_mass(data):
    interval = 5
    mass_max = 0
    mass_loc = 0
    for i in range(len(data)-interval):
        if np.sum(data[i:i+interval]) > mass_max:
            mass_max = np.sum(data[i:i+interval])
            mass_loc = i + interval/2
    return (mass_loc, mass_max)

def sorting(base):
    loc_all_0 = []
    for i in base[:, :]:
        loc, mass = c_mass(i)
        loc_all_0.append(loc) 
        # mass_all.append(mass)
    B = np.argsort(loc_all_0)  

    return B

def relu(inX):
    return np.maximum(0,inX)

def cal_TI(h, start=20, end=80):
    hidden_act = h
    hidden_act = hidden_act.T #600,90
    data = relu(normalization2(hidden_act).T) #90,600
    # data = relu(hidden_act.T[20:80,])
    ts = data.shape[0]  # number of time points
    entrpy_bins = end-start
    window_size = 1
    r_threshold = 0

    # selected_indx = np.nonzero(np.mean(data, axis=0) > r_threshold)[0]
    selected_indx = np.where(np.max(data, axis=0) > r_threshold)[0] # 大于平均发放率的神经元下标
    # selected_indx = np.array(selected_indx).squeeze()
    data = data[:, selected_indx]

    peak_times = np.argmax(data, axis=0) # 峰值时刻
    delay_peak_times = np.argmax(data[start:end,:], axis=0) #延迟时间 最大的发放率的下标
    index1 = np.where(data[start:end,:]>0.4) # 大于0.4的下标
    end_times = np.clip(delay_peak_times + window_size + 1, 0, ts)
    start_times = np.clip(delay_peak_times - window_size, 0, ts)
    # entrpy = entropy(np.histogram(delay_peak_times, entrpy_bins)[0]) 
    entrpy = entropy(np.histogram(delay_peak_times, entrpy_bins)[0]) # 延迟时间的能量
    entrpy_ori = entropy(np.histogram(delay_peak_times, entrpy_bins)[0], base=2)
    entrpy_max = entropy(np.ones(entrpy_bins)*data.shape[1]/entrpy_bins)
    # entrpy = entropy(np.histogram(peak_times, entrpy_bins)[0] + 0.1 * np.ones(entrpy_bins))
    r2b_ratio = np.zeros(len(selected_indx)) # 大于平均发放率的神经元比例
    trans_index = 0
    for nind in range(len(selected_indx)):
        # mask = np.zeros(ts)
        # mask[int(start_times[nind]):int(end_times[nind])] = 1
        data0 = data[start:end, nind]
        # ridge = np.mean(data0[int(start_times[nind]):int(end_times[nind])])
        ridge = np.sum(data0[start_times[nind]:end_times[nind]])
        # backgr = np.mean(np.ma.MaskedArray(data0, mask))
        backgr = np.sum(data0)
        # r2b_ratio[nind] = np.log(ridge) - np.log(backgr)
        if backgr == 0:
            r2b_ratio[nind] = 0
        else:
            r2b_ratio[nind] = ridge/backgr #* (end-start)/(end_times[nind]-start_times[nind]) # 窗口期间发放率
        trace_sum = ridge/(end_times[nind]-start_times[nind])
        trans_index += trace_sum

    trans_index /= len(selected_indx)
    entrpy = entrpy / entrpy_max
    r2b_ratio = np.nanmean(r2b_ratio)

    index1 = np.where(peak_times>=start)
    index2 = np.where(peak_times<end)
    index_delay = np.intersect1d(index1[0], index2[0]) # 延迟期间的下标
    trans_len = len(index_delay)/len(selected_indx)
    # transient_index = np.sum(data)/(60*500)
    SI_trial_vec = r2b_ratio + entrpy
    Total = SI_trial_vec + trans_len
    return round(Total, 3), entrpy_ori, entrpy, r2b_ratio, round(SI_trial_vec, 3), trans_index, trans_len

def visualize_policy_inputs(normalized_obs: Dict[str, Tensor]) -> None:
    """
    Display actual policy inputs after all wrappers and normalizations using OpenCV imshow.
    """
    import cv2

    if "obs" not in normalized_obs.keys():
        return

    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent
    obs = obs[0]
    if obs.dim() != 3:
        # this function is only for RGB images
        return

    # convert to HWC
    obs = obs.permute(1, 2, 0)
    # convert to numpy
    obs = obs.cpu().numpy()
    # convert to uint8
    obs = cv2.normalize(
        obs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1
    )  # this will be different frame-by-frame but probably good enough to give us an idea?
    scale = 5
    obs = cv2.resize(obs, (obs.shape[1] * scale, obs.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("policy inputs", obs)
    cv2.waitKey(delay=1)


def render_frame(cfg, env, video_frames, num_episodes, last_render_start) -> float:
    render_start = time.time()

    if cfg.save_video:
        need_video_frame = len(video_frames) < cfg.video_frames or cfg.video_frames < 0 and num_episodes == 0
        if need_video_frame:
            frame = env.render()
            if frame is not None:
                video_frames.append(frame.copy())
    else:
        if not cfg.no_render:
            target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
            current_delay = render_start - last_render_start
            time_wait = target_delay - current_delay

            if time_wait > 0:
                # log.info("Wait time %.3f", time_wait)
                time.sleep(time_wait)

            try:
                env.render()
            except (gym.error.Error, TypeError) as ex:
                debug_log_every_n(1000, f"Exception when calling env.render() {str(ex)}")

    return render_start

Time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def saveFile(cfg, words):
    filepath = "/home/jiashuncheng/code/Trasient/RL/train_dir/{}/{}_sf_test_log.txt".format(cfg.experiment, Time)
    file = open(filepath, "a+")
    file.write(words)
    file.write('\n')
    file.close()

def saveObsFile(cfg, obs_list, rew_list):
    Obsfilepath = "/home/jiashuncheng/code/Trasient/RL/train_dir/{}/{}_sf_obs_log.npy".format(cfg.experiment, Time)
    Rewardfilepath = "/home/jiashuncheng/code/Trasient/RL/train_dir/{}/{}_sf_rew_log.npy".format(cfg.experiment, Time)
    with open(Obsfilepath, 'wb') as f:
        np.save(f, obs_list)
    with open(Rewardfilepath, 'wb') as f:
        np.save(f, rew_list)

def saveRewFile(cfg, rew):
    filepath = "/home/jiashuncheng/code/Trasient/RL/train_dir/{}/{}_sf_size{}_log.txt".format(cfg.experiment, Time, cfg.size)
    file = open(filepath, "a+")
    file.write(str(rew))
    file.write('\n')
    file.close()

def saveRewFile2(cfg, rew):
    filepath = "/home/jiashuncheng/code/Trasient/RL/train_dir/{}/{}_sf_delay{}_log.txt".format(cfg.experiment, Time, cfg.delay)
    file = open(filepath, "a+")
    file.write(str(rew))
    file.write('\n')
    file.close()

def saveTIFile(cfg, rew):
    filepath = "/home/jiashuncheng/code/Trasient/RL/train_dir/{}/{}_sf_TI_log.txt".format(cfg.experiment, Time)
    file = open(filepath, "a+")
    file.write(str(rew))
    file.write('\n')
    file.close()
temp=0
def saveSuccessFile(cfg, rew):
    filepath = "/home/jiashuncheng/code/Trasient/RL/train_dir/{}/{}_sf_success_list_log.txt".format(cfg.experiment, Time)
    file = open(filepath, "a+")
    file.write(str(rew))
    file.write('\n')
    file.close()
    global temp
    temp+=1

def cal_TI(h, start=20, end=80):
    hidden_act = h.T #600,90
    assert hidden_act.shape[0]==600 or hidden_act.shape[0]==300 or hidden_act.shape[0]==90 or hidden_act.shape[0]==120 or hidden_act.shape[0]==60 or hidden_act.shape[0]==900
    data = relu(normalization2(hidden_act).T) #90,600
    # data = relu(hidden_act.T[20:80,])
    ts = data.shape[0]  # number of time points
    entrpy_bins = end-start
    window_size = 4
    r_threshold = 0

    entrpy_max = entropy(np.ones(entrpy_bins)*data.shape[1]/entrpy_bins)
    # selected_indx = np.nonzero(np.mean(data, axis=0) > r_threshold)[0]
    selected_indx = np.where(np.max(data, axis=0) > r_threshold)[0] # 大于平均发放率的神经元下标
    # selected_indx = np.array(selected_indx).squeeze()
    data = data[:, selected_indx]

    peak_times = np.argmax(data, axis=0) # 峰值时刻
    delay_peak_times = np.argmax(data[start:end,:], axis=0) #延迟时间 最大的发放率的下标
    end_times = np.clip(delay_peak_times + window_size + 1, 0, ts)
    start_times = np.clip(delay_peak_times - window_size, 0, ts)
    # entrpy = entropy(np.histogram(delay_peak_times, entrpy_bins)[0]) 
    entrpy = entropy(np.histogram(delay_peak_times, entrpy_bins)[0]) # 延迟时间的能量
    entrpy_ori = entropy(np.histogram(delay_peak_times, entrpy_bins)[0], base=2)
    # entrpy = entropy(np.histogram(peak_times, entrpy_bins)[0] + 0.1 * np.ones(entrpy_bins))
    r2b_ratio = np.zeros(len(selected_indx)) # 大于平均发放率的神经元比例
    trans_index = 0
    for nind in range(len(selected_indx)):
        # mask = np.zeros(ts)
        # mask[int(start_times[nind]):int(end_times[nind])] = 1
        data0 = data[start:end, nind]
        # ridge = np.mean(data0[int(start_times[nind]):int(end_times[nind])])
        ridge = np.sum(data0[start_times[nind]:end_times[nind]])
        # backgr = np.mean(np.ma.MaskedArray(data0, mask))
        backgr = np.sum(data0)
        # r2b_ratio[nind] = np.log(ridge) - np.log(backgr)
        if backgr == 0:
            r2b_ratio[nind] = 0
        else:
            r2b_ratio[nind] = ridge/backgr #* (end-start)/(end_times[nind]-start_times[nind]) # 窗口期间发放率
        trace_sum = ridge/(end_times[nind]-start_times[nind])
        trans_index += trace_sum

    trans_index /= len(selected_indx)
    entrpy = entrpy / entrpy_max
    r2b_ratio = np.nanmean(r2b_ratio)

    index1 = np.where(peak_times>=start)
    index2 = np.where(peak_times<end)
    index_delay = np.intersect1d(index1[0], index2[0]) # 延迟期间的下标
    trans_len = len(index_delay)/len(selected_indx)
    # transient_index = np.sum(data)/(60*500)
    SI_trial_vec = r2b_ratio + entrpy
    Total = SI_trial_vec + trans_len
    return round(Total, 3), entrpy_ori, entrpy, r2b_ratio, round(SI_trial_vec, 3), trans_index, trans_len

def savehiddenFile(cfg, rew):
    filepath = "/home/jiashuncheng/code/Trasient/RL/train_dir/{}/{}_sf_hidden_act_log.npy".format(cfg.experiment, Time)
    with open(filepath, 'wb') as f:
        np.save(f, rew)
    print(f'\033[1;31m{filepath}\033[0m')

def enjoy(cfg: Config) -> Tuple[StatusCode, float]:
    verbose = False

    cfg = load_from_checkpoint(cfg)

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1

    render_mode = "human"
    if cfg.save_video:
        render_mode = "rgb_array"
    elif cfg.no_render:
        render_mode = None
    elif cfg.mm_render:
        render_mode = "debug_rgb_array"

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )
    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    reward_list = []
    rew_list = []
    obs_list = []

    obs, infos = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]

    video_frames = []
    num_episodes = 0

    with torch.no_grad():
        while not max_frames_reached(num_frames):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

            if not cfg.no_render:
                visualize_policy_inputs(normalized_obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            # sample actions from the distribution by default
            actions = policy_outputs["actions"]

            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            for _ in range(render_action_repeat):
                last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

                obs, rew, terminated, truncated, infos = env.step(actions)
                rew_list.append(rew.numpy().item())
                if "dmlab" in cfg.experiment:
                    obs_list.append(obs['DEBUG.POS.TRANS'].numpy()[0])
                if "Watermaze2d" in cfg.experiment:
                    obs_list.append(infos[0]['agent_pos'])
                dones = make_dones(terminated, truncated)
                infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                if episode_reward is None:
                    episode_reward = rew.float().clone()
                else:
                    episode_reward += rew.float()

                num_frames += 1
                if num_frames % 100 == 0:
                    log.debug(f"Num frames {num_frames}...")

                dones = dones.cpu().numpy()
                for agent_i, done_flag in enumerate(dones):
                    if done_flag:
                        if "MortarMayhem" in cfg.experiment:
                            # print(len(rew_list))
                            # rew_list = []
                            # saveSuccessFile(cfg, infos[agent_i]['success_list'])
                            # global temp
                            # if temp==100: sys.exit()
                            trace = torch.stack(actor_critic.core.hidden_output).squeeze(1).squeeze(1)
                            hidden_act = trace.cpu().detach().numpy()
                            savehiddenFile(cfg, hidden_act)                
                            print(hidden_act.shape)
                            ti = cal_TI(hidden_act,4,8)[0] 
                            print('ti:',ti)
                            sys.exit()

                            # TI = cal_TI(hidden_act,2,11)[0]
                            # saveTIFile(cfg, TI)
                            # actor_critic.core.hidden_output = []
                            pass

                        if "Ratlapwater" in cfg.experiment:
                            # saveRewFile(cfg, sum(rew_list))
                            # rew_list = []
                            trace = torch.stack(actor_critic.core.hidden_output).squeeze(1)
                            hidden_act = trace.cpu().detach().numpy()
                            savehiddenFile(cfg, hidden_act)                
                            print(hidden_act.shape)
                            TI = cal_TI(hidden_act,2,11)[0]
                            saveTIFile(cfg, TI)
                            actor_critic.core.hidden_output = []

                        if "Tmaze" in cfg.experiment:
                            # saveRewFile2(cfg, sum(rew_list))
                            # rew_list = []
                            trace = torch.stack(actor_critic.core.hidden_output).squeeze(1)
                            hidden_act = trace.cpu().detach().numpy()
                            # savehiddenFile(cfg, hidden_act)
                            print(hidden_act.shape)
                            TI = cal_TI(hidden_act,3,9)[0]
                            saveTIFile(cfg, TI)
                            actor_critic.core.hidden_output = []

                        # use for dmlab watermaze, begin:
                        if "dmlab" in cfg.experiment or "Watermaze2d" in cfg.experiment:
                            if len(np.where(np.array(rew_list)!=0.0)[0]) == 0:
                                firstR_list = np.array([len(rew_list)])
                                firstR = []
                                endR = []
                            else:
                                firstR_list = np.where(np.array(rew_list)!=0.0)[0].tolist()
                                value = np.array(rew_list)[np.where(np.array(rew_list)!=0.0)[0]].tolist()
                                firstR = [firstR_list[0]]
                                endR = [0]
                                sum_ = 0
                                for i in range(len(firstR_list)):
                                    if (sum_ > 3.0 and sum_ <= 4.0) or (sum_ <= 3.0 and (sum_ + value[i] > 4.0)):
                                        endR.append(firstR_list[i])
                                    if sum_ > 4.0:
                                        firstR.append(firstR_list[i])
                                        sum_ = 0
                                    sum_ = sum_ + value[i]
                            obs_list = np.array(obs_list)
                            rew_list = np.array(rew_list)
                            saveObsFile(cfg, obs_list, rew_list)
                            saveFile(cfg, "FirstR_list:" + str(firstR_list))
                            # saveFile(cfg, "Value_list:" + str(value))
                            saveFile(cfg, "FirstR:" + str(firstR))
                            saveFile(cfg, "EndR:" + str(endR))
                            DeltaFirstR = []
                            if len(np.where(np.array(rew_list)!=0.0)[0]) != 0:
                                for i in range(0, len(firstR)):
                                    DeltaFirstR.append(firstR[i]-endR[i])
                            saveFile(cfg, "DeltaFirstR:" + str(DeltaFirstR))
                            saveFile(cfg, "\n")
                            rew_list = []
                            obs_list = []
                            # trace = torch.stack(actor_critic.core.hidden_output).squeeze(1)
                            # hidden_act = trace.cpu().detach().numpy()
                            # savehiddenFile(cfg, hidden_act)                
                            # print(hidden_act.shape)
                            # TI = cal_TI(hidden_act,2,11)[0]
                            # saveTIFile(cfg, TI)
                            # actor_critic.core.hidden_output = []
                        # use for dmlab watermaze, end.
                        finished_episode[agent_i] = True
                        rew = episode_reward[agent_i].item()
                        episode_rewards[agent_i].append(rew)

                        true_objective = rew
                        if isinstance(infos, (list, tuple)):
                            true_objective = infos[agent_i].get("true_objective", rew)
                        true_objectives[agent_i].append(true_objective)

                        if verbose:
                            log.info(
                                "Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
                                agent_i,
                                num_frames,
                                episode_reward[agent_i],
                                true_objectives[agent_i][-1],
                            )
                        rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                        episode_reward[agent_i] = 0

                        if cfg.use_record_episode_statistics:
                            # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                            if "episode" in infos[agent_i].keys():
                                num_episodes += 1
                                reward_list.append(infos[agent_i]["episode"]["r"])
                        else:
                            num_episodes += 1
                            reward_list.append(true_objective)

                # if episode terminated synchronously for all agents, pause a bit before starting a new one
                if all(dones):
                    render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                    time.sleep(0.05)

                if all(finished_episode):
                    finished_episode = [False] * env.num_agents
                    avg_episode_rewards_str, avg_true_objective_str = "", ""
                    for agent_i in range(env.num_agents):
                        avg_rew = np.mean(episode_rewards[agent_i])
                        avg_true_obj = np.mean(true_objectives[agent_i])

                        if not np.isnan(avg_rew):
                            if avg_episode_rewards_str:
                                avg_episode_rewards_str += ", "
                            avg_episode_rewards_str += f"#{agent_i}: {avg_rew:.3f}"
                        if not np.isnan(avg_true_obj):
                            if avg_true_objective_str:
                                avg_true_objective_str += ", "
                            avg_true_objective_str += f"#{agent_i}: {avg_true_obj:.3f}"

                    log.info(
                        "Avg episode rewards: %s, true rewards: %s", avg_episode_rewards_str, avg_true_objective_str
                    )
                    log.info(
                        "Avg episode reward: %.3f, avg true_objective: %.3f",
                        np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]),
                        np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]),
                    )

                # VizDoom multiplayer stuff
                # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                #     key = f'PLAYER{player}_FRAGCOUNT'
                #     if key in infos[0]:
                #         log.debug('Score for player %d: %r', player, infos[0][key])

            if num_episodes >= cfg.max_num_episodes:
                break

    env.close()

    if cfg.save_video:
        if cfg.fps > 0:
            fps = cfg.fps
        else:
            fps = 30
        generate_replay_video(experiment_dir(cfg=cfg), video_frames, fps, cfg)

    if cfg.push_to_hub:
        generate_model_card(
            experiment_dir(cfg=cfg),
            cfg.algo,
            cfg.env,
            cfg.hf_repository,
            reward_list,
            cfg.enjoy_script,
            cfg.train_script,
        )
        push_to_hf(experiment_dir(cfg=cfg), cfg.hf_repository)

    return ExperimentStatus.SUCCESS, sum([sum(episode_rewards[i]) for i in range(env.num_agents)]) / sum(
        [len(episode_rewards[i]) for i in range(env.num_agents)]
    )
