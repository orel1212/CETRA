from datetime import datetime
from detectors import XGBoostDetector, ClassificationUsingRNNDetector, FFNNDetector, CNNDetector, PDRCNNDetector
from chainerrl import misc

import envs
from model import CetraAgent

from utils import load_dataset, dataset_split, reward_functions_for_agent, calc_T_N_rates, calc_accurary, calc_auc, \
    calc_precision_recall, calc_F1_score, cast_by_threshold

import matplotlib.pyplot as plt

import os
import sys
import csv
import numpy as np
import pandas as pd
import random

import time
import gym
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from config import Config



config_obj = Config()

flag_times = config_obj.detectors['flag_times']

DATASET_PATH = Path(config_obj.data['default_dataset_path'])

DATASET = config_obj.data['default_dataset']

if len(sys.argv) > 1:
    DATASET = sys.argv[1]

experiment_number = config_obj.data['experiment_number']

if len(sys.argv) > 2:
    experiment_number = int(sys.argv[2])

FILES_PATH = DATASET_PATH

to_load_flag = config_obj.model['load_model_flag']
load_model_path = Path(config_obj.model['model_dir_path'])

file_to_read = DATASET + config_obj.data['dataset_postfix']
path_file = DATASET_PATH / file_to_read

if os.path.isfile(path_file):
    print("Loaded " + DATASET + " dataset...")
else:
    print("Not found " + DATASET + "dataset...")
    exit()
urls_df = pd.read_csv(path_file, index_col=False)

act_deterministically_flag = config_obj.model['act_deterministically_flag']

test_set_split = config_obj.data['test_set_split']
valid_set_split = config_obj.data['valid_set_split']

num_of_epochs = config_obj.model['num_of_epochs']

mode = config_obj.model['mode']

random_seed = config_obj.model['random_seed']

predictions_train, times_train, predictions_valid, times_valid, predictions_test, times_test = [], [], [], [], [], []

output_file_suffix = str(DATASET + "_exp" + str(experiment_number) + "_" + config_obj.model['save_output_filename'])

output_file = Path(output_file_suffix)

# metric based hyperparams

batch_metrics_size = config_obj.data['batch_metrics_size']

user_defined_metrics_dict = config_obj.model['user_defined_metrics_dict']

curr_batch_counter = 1

curr_batch_rewards = np.array([], dtype=np.float32)

curr_metrics_dict = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

METRIC_BONUS_FUNCTION = config_obj.model['metrics_bonus_function']


def update_batch_counter():
    global curr_batch_counter
    curr_batch_counter += 1


def reset_current_metrics():
    curr_metrics_keys = curr_metrics_dict.keys()
    for key in curr_metrics_keys:
        curr_metrics_dict[key] = 0
    global curr_batch_counter
    curr_batch_counter = 1
    global curr_batch_rewards
    curr_batch_rewards = np.array([], dtype=np.float32)


def reward_for_agent_after_metrics_evaluation():
    total_reward = 0
    base_total_reward = curr_batch_rewards.sum()  # sum over batch rewards
    metrics = user_defined_metrics_dict.keys()
    rates_dict = calc_T_N_rates(curr_metrics_dict['TP'], curr_metrics_dict['TN'], curr_metrics_dict['FP'],
                                curr_metrics_dict['FN'])                  
    for key in metrics:
        if user_defined_metrics_dict[key]['min'] == -1:  # need to skip metric, user does not want it
            continue
        else:
            if key[0] == 'F':  # to compensate for difference in measures, TPR is higher value (0.9), FPR is not(0.05). FPR, FNR starts with F
                curr_metric_val = 1 - rates_dict[key]
                curr_user_defined_metric_val = 1 - user_defined_metrics_dict[key]['min']
                curr_user_defined_extra_metric_val = 1 - user_defined_metrics_dict[key]['extra']
            else:
                curr_metric_val = rates_dict[key]
                curr_user_defined_metric_val = user_defined_metrics_dict[key]['min']
                curr_user_defined_extra_metric_val = user_defined_metrics_dict[key]['extra']

            if curr_metric_val < curr_user_defined_metric_val:
                total_reward += METRIC_BONUS_FUNCTION(
                    -1 * np.absolute(base_total_reward))  # compensate for negative values
            elif curr_metric_val >= curr_user_defined_extra_metric_val:
                total_reward += METRIC_BONUS_FUNCTION(base_total_reward)
            else:
                total_reward += base_total_reward
    return total_reward


def add_reward_to_batch_rewards(reward):
    global curr_batch_rewards
    curr_batch_rewards = np.append(curr_batch_rewards, [reward], axis=0)


def update_current_metrics_and_rewards(episode_prediction, episode_reward):
    if episode_prediction == 'DUP' or episode_prediction == 'DIR':
        return
    flag_found_prediction_in_metrics = False
    curr_metrics_keys = curr_metrics_dict.keys()
    for key in curr_metrics_keys:
        if episode_prediction == key:
            curr_metrics_dict[key] = curr_metrics_dict[key] + 1
            add_reward_to_batch_rewards(episode_reward)
            flag_found_prediction_in_metrics = True

    if flag_found_prediction_in_metrics == False:
        print("update_current_metrics_and_rewards: Metric: " + episode_prediction + " is unknown!!")
    else:
        update_batch_counter()


def process_url(url, train=True):
    reward, done = 0, False
    state = env.reset(url=url)
    # print("start state:"+str(state))
    while not done:
        if train:
            action = agent.act_and_train(state, reward=reward)
        else:
            action = agent.act(state)

        state, reward, done, _info = env.step(action)
        # print("state:" + str(state) +"_reward:"+str(reward))

    episode_pred, final_costs = env.get_pred_and_costs()
    episode_costs = final_costs.sum()

    # End episode
    if train:
        update_current_metrics_and_rewards(episode_pred, episode_costs)
        if curr_batch_counter == batch_metrics_size:
            print("reward before: " + str(curr_batch_rewards.sum()))
            total_batch_reward = reward_for_agent_after_metrics_evaluation()
            print("reward after: " + str(total_batch_reward))
            reset_current_metrics()
            reward = reward + total_batch_reward  # update reward with batch reward after a batch is done.
        agent.stop_episode_and_train(state, reward, done)
    else:
        agent.stop_episode()

    return episode_pred, episode_costs


def proccess_all_urls(url):
    if mode == 'train':
        p, t = process_url(url, train=True)
        if p == 'DUP' or p == 'DIR':  # for DIRECT or DUP ACTION
            if config_obj.general['debug_level_logs']:
                print("not added to times, " + p + " action!")
        else:
            predictions_train.append(p)
            times_train.append(t)

    elif mode == 'valid':
        p, t = process_url(url, train=False)
        if p == 'DUP' or p == 'DIR':  # for DIRECT or DUP ACTION
            if config_obj.general['debug_level_logs']:
                print("not added to times, " + p + " action!")
        else:
            predictions_valid.append(p)
            times_valid.append(t)
    else:
        p, t = process_url(url, train=False)
        if p == 'DUP' or p == 'DIR':  # for DIRECT or DUP ACTION
            if config_obj.general['debug_level_logs']:
                print("not added to times, " + p + " action!")
        else:
            predictions_test.append(p)
            times_test.append(t)


def metrics(original_data, predictions, times, mode='train', epoch_number=0, duration=0):
    detectors_cols = ["CnnModel", "PDRCNN", "XGBoost", "ClassificationUsingRNN", "FFNN"]
    detectors_metrics_cols = []
    for detector in detectors_cols:
        detectors_metrics_cols.append(detector + "_Accuracy")
        detectors_metrics_cols.append(detector + "_Precision")
        detectors_metrics_cols.append(detector + "_Recall")
        detectors_metrics_cols.append(detector + "_F1")
        detectors_metrics_cols.append(detector + "_AUC")
        detectors_metrics_cols.append(detector + "_AvgTime")
    detectors_metrics_data = []
    for detector in detectors_cols:
        labels = original_data['label']
        predicts_data = original_data[detector].apply(cast_by_threshold)
        # accuracy: (tp + tn) / (p + n)
        accuracy_d = accuracy_score(labels, predicts_data)
        detectors_metrics_data.append(accuracy_d)

        # precision tp / (tp + fp)
        precision_d = precision_score(labels, predicts_data)
        detectors_metrics_data.append(precision_d)

        # recall: tp / (tp + fn)
        recall_d = recall_score(labels, predicts_data)
        detectors_metrics_data.append(recall_d)

        # f1: 2 tp / (2 tp + fp + fn)
        f1_d = f1_score(labels, predicts_data)
        detectors_metrics_data.append(f1_d)

        auc_d = roc_auc_score(labels, predicts_data[:])
        detectors_metrics_data.append(auc_d)

        avg_time_d = original_data[detector + "Time"].to_numpy().mean()
        detectors_metrics_data.append(avg_time_d)

    accuracy = calc_accurary(predictions)
    auc = calc_auc(predictions)
    precision, recall = calc_precision_recall(predictions)
    F1_score = calc_F1_score(precision, recall)

    true_positives = predictions.count('TP')
    true_negatives = predictions.count('TN')
    false_positives = predictions.count('FP')
    false_negatives = predictions.count('FN')
    rates_dict = calc_T_N_rates(true_positives, true_negatives, false_positives, false_negatives)

    if len(times) == 0:
        average_time = 0
        if config_obj.general['debug_level_logs']:
            print("DIR only")
    else:
        average_time = sum(times) / len(times)
    metrics = [mode, epoch_number, accuracy, auc, rates_dict['TPR'], rates_dict['FPR'], rates_dict['TNR'],
               rates_dict['FNR'], precision, recall, F1_score, average_time, duration] + detectors_metrics_data
    if not output_file.exists():
        columns_names = ['mode', 'epoch_number', 'accuracy', 'auc', 'TPR', 'FPR', 'TNR', 'FNR', 'precision', 'recall',
                         'f1', 'average_time', 'duration'] + detectors_metrics_cols
        model_df = pd.DataFrame(None, columns=columns_names)
        model_df.to_csv(str(output_file), index=False)
    with open(str(output_file), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics)

    f.close()
    print("saved metrics file!")


def update_detectors_flag_times(detectors, flag_times='REAL'):
    for detector_object in detectors.values():
        detector_object.set_flag_times(flag_times)



def train_helper(train_data, detectors, epoch_num):
    update_detectors_flag_times(detectors, flag_times)
    agent.set_act_deterministically_flag(act_deterministically_flag)
    predictions_train.clear()
    times_train.clear()
    print("started" + ' epoch_' + str(epoch_num) + " train")
    start = datetime.now()
    train_data.apply(proccess_all_urls, axis=1)
    end = datetime.now()
    duration = (end - start).total_seconds()
    metrics(train_data, predictions_train, times_train, mode, epoch_num, duration)  # metrics of Train
    time_save = datetime.now()
    # Dump model on train
    save_path = str(DATASET + '_exp_' + str(experiment_number) + '_epoch_' + str(epoch_num) + "_" + str(
        time_save.strftime("%d-%m-%y")) + "_" + str(time_save.hour) + "-" + str(time_save.minute) + "-" + str(
        time_save.second))
    agent.save(save_path)
    print("Saved agent model on epoch:" + str(epoch_num))
    
    
def valid_evaluation_helper(valid_data, detectors, epoch_num):

    agent.set_act_deterministically_flag(True)
    flag_times = "REAL"
    update_detectors_flag_times(detectors, flag_times)

    print("started" + ' epoch_' + str(epoch_num) + " validation")
    mode = 'valid'
    predictions_valid.clear()
    times_valid.clear()
    valid_data.apply(proccess_all_urls, axis=1)
    metrics(valid_data, predictions_valid, times_valid, mode, epoch_num)  # metrics of Valid
    
    
def test_evaluation_helper(test_data, detectors, epoch_num):

    agent.set_act_deterministically_flag(True)
    flag_times = "REAL"
    update_detectors_flag_times(detectors, flag_times)

    print("started" + ' epoch_' + str(epoch_num) + " test")
    mode = 'test'
    predictions_test.clear()
    times_test.clear()
    test_data.apply(proccess_all_urls, axis=1)
    metrics(test_data, predictions_test, times_test, mode)  # metrics of Test

if __name__ == '__main__':

    ffnn_detector = FFNNDetector(flag_times)
    pdrcnn_detector = PDRCNNDetector(flag_times)
    cnn_detector = CNNDetector(flag_times)
    xgb_detector = XGBoostDetector(flag_times)
    classusingrnn_detector = ClassificationUsingRNNDetector(flag_times)

    misc.set_random_seed(random_seed)
    detectors = {
        'ffnn': ffnn_detector,
        'pdrcnn': pdrcnn_detector,
        'cnn': cnn_detector,
        'xgb': xgb_detector,
        'classusingrnn': classusingrnn_detector
    }

    reward_func, t_func, fp_func, fn_func = reward_functions_for_agent(DATASET, experiment_number)
    env = gym.make(
        'Cetra-v1',
        detectors=detectors,  # Action labels
        cost_func=reward_func,  # Cost function
        t_func=t_func,  # Reward for TP or TN
        fp_func=fp_func,  # Reward for FP
        fn_func=fn_func,  # Reward for FN
        illegal_reward=config_obj.detectors['illegal_reward']  # Cost for illegal action
    )

    agent = CetraAgent(train_mode=True, load_model=to_load_flag, path_to_load=load_model_path,
                       input_size=env.observation_space.shape[0], output_size=env.action_space.n)

    urls, labels = load_dataset(urls_df)

    train_data, valid_data, test_data = dataset_split(urls, labels, test_set_split, valid_set_split)

    print(DATASET + '_exp_' + str(experiment_number))

    for epoch_num in range(1, num_of_epochs + 1):
        if mode == 'train':
            train_helper(train_data, detectors, epoch_num)
            valid_evaluation_helper(valid_data, detectors, epoch_num)
        test_evaluation_helper(test_data, detectors, epoch_num)




