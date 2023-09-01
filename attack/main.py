from datetime import datetime
from detectors import XGBoostDetector, ClassificationUsingRNNDetector, FFNNDetector, CNNDetector, PDRCNNDetector
from model import CetraAgent
from chainerrl import misc

from attacks import L_Norm_Attacks
from utils import load_dataset, dataset_split, reward_functions_for_agent

from config import Config

import os
import sys
import csv
import numpy as np
import pandas as pd
import random
import envs
import time
import gym
from pathlib import Path


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

stochastic_resource_internal_epochs = config_obj.model['stochastic_resource_internal_epochs']

mode = config_obj.model['mode']

random_seed = config_obj.model['random_seed']

adversarial_sampling_epochs = config_obj.model['adversarial_sampling_epochs']

batch_attack_size = config_obj.data['batch_attack_size']

epsilon_to_attack = config_obj.model['epsilon_to_attack']

misclassify_attack = config_obj.model['misclassify_attack']

resource_attack = config_obj.model['resource_attack']

resource_attack_mode = config_obj.model['resource_attack_mode']

attack_method_gradient = config_obj.model['attack_method_gradient']

split_benign_percentages = config_obj.data['benign_percentages']

split_phish_percentages = config_obj.data['phish_percentages']

req_dim = config_obj.data['req_dim']

to_create_plotting_flag = config_obj.general['create_plotting_flag']

to_histogram = config_obj.general['histogram']

attack_norm = config_obj.model['attack_norm']

loss_f = config_obj.model['loss_f']

attack = config_obj.model['attack_type']

step_eps = config_obj.model['step_eps']

print(f'Attack configurations:\n {attack=}\n {step_eps=} \n {epsilon_to_attack=} \n {batch_attack_size=}')

BENIGN = config_obj.data['dataset_postfix']
PHISH = config_obj.data['dataset_postfix']

attack_valid_data = config_obj.data['attack_valid_data']

attack_valid_benign_to_phish, attack_valid_phish_to_benign, attack_test_benign_to_phish, attack_test_phish_to_benign = [], [], [], []

attack_valid_benign_benign, attack_valid_phish_phish, attack_test_benign_benign, attack_test_phish_phish = [], [], [], []

valid_bad_samples, test_bad_samples = 0, 0

valid_num_detector_changes_lst, valid_magnitude_changes_lst, test_num_detector_changes_lst, test_magnitude_changes_lst = [], [], [], []

dim_min_eps_benign_to_phish_attacks, dim_min_eps_phish_to_benign_attacks = [], []

attack_output_file_suffix = str(DATASET + "_exp" + str(experiment_number) + "_" + config_obj.model['save_output_filename'])

attack_output_file = Path(attack_output_file_suffix)


def process_url_misclassify_attack(url):
    reward, done = 0, False
    state = env.reset(url=url)
    action = -1
    # print("start state:"+str(state))
    while not done:
        action = agent.act(state)
        state, reward, done, _info = env.step(action)
    agent.stop_episode()

    real_label = url['label']
    chosen_label = action
    true_pred_flag = False

    if len(_info) > 0:  # check if dup or dir action
        return False, BENIGN, real_label, False, 0, [], _info

    agent.set_act_deterministically_flag(act_deterministically_flag)

    if real_label == 1:  # avoid fp/fn by drl.
        if chosen_label == BENIGN:
            return False, BENIGN, real_label, true_pred_flag, 0, [], {}
    elif real_label == 0:  # avoid fp/fn by drl.
        if chosen_label == PHISH:
            return False, PHISH, real_label, true_pred_flag, 0, [], {}
    else:
        true_pred_flag = True

    if chosen_label == PHISH:  # Try to make it phish
        target_label = BENIGN
    else:
        target_label = PHISH

    if to_create_plotting_flag is True:

        min_eps_per_dim_lst = L_Norm_Attacks.misclassify_attack_plotting_vectorized(url, env, agent, target_label,
                                                                                    dim=req_dim,
                                                                                    to_plot=False)
        if target_label == PHISH:
            dim_min_eps_benign_to_phish_attacks.append(min_eps_per_dim_lst)
        else:
            dim_min_eps_phish_to_benign_attacks.append(min_eps_per_dim_lst)
        return False, PHISH, real_label, true_pred_flag, 0, [], {}

    else:
        if attack_method_gradient is False:
            success_attack, perturbed_states, num_detector_changes, magnitude_changes = L_Norm_Attacks.misclassify_determenistic_exhaustive_targeted_non_gradient(
                url, env, agent, target_label, epsilon=epsilon_to_attack, attack=attack_norm)

        else:
            success_attack, perturbed_states, num_detector_changes, magnitude_changes = L_Norm_Attacks.misclassify_determenistic_exhaustive_targeted_linf_gradient(
                url, env, agent, target=target_label, epsilon=epsilon_to_attack, loss_f=loss_f, attack=attack,
                step_eps=step_eps)
        # episode_pred, final_costs = env.get_pred_and_costs()
        # episode_costs = final_costs.sum()

    return success_attack, chosen_label, target_label, true_pred_flag, num_detector_changes, magnitude_changes, {}


def process_all_urls_misclassify_attack(url):
    success_attack, chosen_label, target_label, true_pred_flag, num_detector_changes, magnitude_changes, info = process_url_misclassify_attack(
        url)
    if mode == 'valid':
        if success_attack == True:
            if chosen_label == PHISH:
                attack_valid_phish_to_benign.append(1)
            elif chosen_label == BENIGN:
                attack_valid_benign_to_phish.append(1)

            valid_magnitude_changes_lst.append(magnitude_changes)
            valid_num_detector_changes_lst.append(num_detector_changes)

        elif len(info) > 0:  # dir or dup action chosen
            global valid_bad_samples
            valid_bad_samples = valid_bad_samples + 1

    elif mode == 'test':
        if success_attack == True:
            if chosen_label == PHISH:
                attack_test_phish_to_benign.append(1)
            elif chosen_label == BENIGN:
                attack_test_benign_to_phish.append(1)
            test_magnitude_changes_lst.append(magnitude_changes)
            test_num_detector_changes_lst.append(num_detector_changes)
        elif len(info) > 0:  # dir or dup action chosen
            global test_bad_samples
            test_bad_samples = test_bad_samples + 1


def process_url_resources_attack(url):
    reward, done = 0, False
    state = env.reset(url=url)
    action = -1
    # print("start state:"+str(state))
    while not done:
        action = agent.act(state)
        state, reward, done, _info = env.step(action)
    agent.stop_episode()

    real_label = url['label']
    chosen_label = action

    if len(_info) > 0:  # check if dup or dir action
        return False, BENIGN, real_label, False, 0, [], _info, 0

    agent.set_act_deterministically_flag(act_deterministically_flag)

    true_pred_flag = False

    if real_label == 1:  # avoid fp/fn by drl.
        if chosen_label == BENIGN:
            return False, BENIGN, real_label, true_pred_flag, 0, [], {}, 0
    elif real_label == 0:  # avoid fp/fn by drl.
        if chosen_label == PHISH:
            return False, PHISH, real_label, true_pred_flag, 0, [], {}, 0
    else:
        true_pred_flag = True

    target_label = chosen_label

    if attack_method_gradient is False:
        print("NOT IMPLEMENTED")
        exit()
        success_attack, perturbed_states, num_detector_changes, magnitude_changes = L_Norm_Attacks.resources_determenistic_exhaustive_targeted_non_gradient(
            url, env, agent, target_label, epsilon=epsilon_to_attack, attack=attack_norm)

    else:
        if act_deterministically_flag is True:
            success_attack, perturbed_states, num_detector_changes, magnitude_changes = L_Norm_Attacks.resources_determenistic_exhaustive_targeted_linf_gradient(
                url, env, agent, target=target_label, epsilon=epsilon_to_attack, loss_f=loss_f, attack=attack,
                step_eps=step_eps, mode=resource_attack_mode)
        else:  # stochastic
            success_attack_prob, avg_num_detector_changes, avg_magnitude_changes = 0, 0, np.full(shape=state.shape,
                                                                                                 fill_value=0,
                                                                                                 dtype=np.float32)
            for i in range(stochastic_resource_internal_epochs):
                success_attack, perturbed_states, num_detector_changes, magnitude_changes = L_Norm_Attacks.resources_determenistic_exhaustive_targeted_linf_gradient(
                    url, env, agent, target=target_label, epsilon=epsilon_to_attack, loss_f=loss_f, attack=attack,
                    step_eps=step_eps, mode=resource_attack_mode)
                if success_attack is True:
                    success_attack_prob += 1
                    avg_num_detector_changes += num_detector_changes
                    for i in range(len(avg_magnitude_changes)):
                        avg_magnitude_changes[i] += magnitude_changes[i]

            if success_attack_prob > 0:
                avg_num_detector_changes /= success_attack_prob  # avg over success attacks
                for i in range(len(avg_magnitude_changes)):
                    avg_magnitude_changes[i] /= success_attack_prob  # avg over success attacks
            success_attack_prob /= stochastic_resource_internal_epochs

            return True, chosen_label, target_label, true_pred_flag, avg_num_detector_changes, avg_magnitude_changes, {}, success_attack_prob

    return success_attack, chosen_label, target_label, true_pred_flag, num_detector_changes, magnitude_changes, {}, 1


def process_all_urls_resources_attack(url):
    success_attack, chosen_label, target_label, true_pred_flag, num_detector_changes, magnitude_changes, info, success_attack_prob = process_url_resources_attack(
        url)
    if mode == 'valid':
        if success_attack == True:
            if chosen_label == PHISH:
                attack_valid_phish_phish.append(success_attack_prob)
            elif chosen_label == BENIGN:
                attack_valid_benign_benign.append(success_attack_prob)
            valid_magnitude_changes_lst.append(magnitude_changes)
            valid_num_detector_changes_lst.append(num_detector_changes)
        elif len(info) > 0:  # dir or dup action chosen
            global valid_bad_samples
            valid_bad_samples = valid_bad_samples + 1


    elif mode == 'test':
        if success_attack == True:
            if chosen_label == PHISH:
                attack_test_phish_phish.append(success_attack_prob)
            elif chosen_label == BENIGN:
                attack_test_benign_benign.append(success_attack_prob)
            test_magnitude_changes_lst.append(magnitude_changes)
            test_num_detector_changes_lst.append(num_detector_changes)
        elif len(info) > 0:  # dir or dup action chosen
            global test_bad_samples
            test_bad_samples = test_bad_samples + 1


def sample_benign_phish(dataframe_data, attack_size, split_benign_percentages, split_phish_percentages, replace=False):
    if 'label' in dataframe_data.columns:
        total_labels = dataframe_data['label'].values.tolist()
        count_lab_1 = total_labels.count(1)
        count_lab_0 = total_labels.count(0)
        benign_attack_size = min(count_lab_0, int(attack_size * split_benign_percentages))
        phish_attack_size = min(count_lab_1, int(attack_size * split_phish_percentages))
        attack_data_benign = dataframe_data.loc[dataframe_data['label'] == 0].sample(
            benign_attack_size, replace=replace, axis=0)
        attack_data_phish = dataframe_data.loc[dataframe_data['label'] == 1].sample(
            phish_attack_size, replace=replace, axis=0)
        attack_sampled_data = pd.concat([attack_data_phish, attack_data_benign], axis=0)
        return attack_sampled_data
    else:
        print("Failed to sample phish_benign")
        exit()


def plot_histogram(dim_min_eps_benign_to_phish_attacks, dim_min_eps_phish_to_benign_attacks, req_dim):
    fig, ax = plt.subplots(1, req_dim, figsize=(8, 6))
    n_bins = list(np.linspace(start=0, stop=1, num=10))
    for i in range(req_dim):
        if len(dim_min_eps_benign_to_phish_attacks) > 0:
            ax[i].hist(np.array(dim_min_eps_benign_to_phish_attacks, dtype=np.float32)[:, i], bins=n_bins,
                       color='b', label='benign to phishing')
        if len(dim_min_eps_phish_to_benign_attacks) > 0:
            ax[i].hist(np.array(dim_min_eps_phish_to_benign_attacks, dtype=np.float32)[:, i], bins=n_bins,
                       color='r', label='phishing to benign')

        ax[i].set_title(f'attack state - timestamp {i + 1}')
        ax[i].set_xlabel('epsilon')
        # ax[i].set_xlim(left=0, right=1)
        ax[i].legend()
    plt.show()


def attack_metrics(determenistic_attack, misclassify_attack, attack_mode, num_bad_examples, epsilon, attack_method,
                   step_eps, split_benign_percentages, split_phish_percentages,
                   num_benign_success, num_phish_success, detector_changes_lst, magnitude_changes_lst, mode='test',
                   epoch_number=0, duration=0):
    detectors_cols = ["CnnModel", "PDRCNN", "XGBoost", "ClassificationUsingRNN", "FFNN"]

    detectors_metrics_data = [0, 0, 0, 0, 0]
    # print(magnitude_changes_lst)
    # print(detector_changes_lst)
    avg_magnitude_np = np.mean(np.array(magnitude_changes_lst), axis=0) if len(
        magnitude_changes_lst) > 0 else detectors_metrics_data
    total_avg_changes = np.mean(np.array(detector_changes_lst)) if len(magnitude_changes_lst) > 0 else 0
    metrics = [mode, determenistic_attack, epoch_number, duration, num_bad_examples, epsilon, attack_method, step_eps,
               split_benign_percentages,
               split_phish_percentages, num_benign_success, num_phish_success, attack_mode]
    if num_benign_success + num_phish_success > 0:
        metrics = metrics + [total_avg_changes] + list(avg_magnitude_np)
    else:
        metrics = metrics + [0] + detectors_metrics_data
    if not attack_output_file.exists():
        columns_names = ['mode', 'determenistic_attack', 'epoch_number', 'duration', 'num_bad_examples', 'epsilon',
                         'attack_method', 'step_eps', 'benign_percentages', 'phish_percentages']
        if misclassify_attack is True:
            part_2_columns_names = ['benign_to_phish', 'phish_to_benign', '-----']
        else:
            part_2_columns_names = ['benign_success', 'phish_success', "attack_mode"]
        columns_names = columns_names + part_2_columns_names + ['avg_detector_changes'] + detectors_cols
        model_df = pd.DataFrame(None, columns=columns_names)
        model_df.to_csv(str(attack_output_file), index=False)
    with open(str(attack_output_file), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics)

    f.close()
    print("Saved attack metric file!")


def update_detectors_flag_times(detectors, flag_times='REAL'):
    for detector_object in detectors.values():
        detector_object.set_flag_times(flag_times)



def validation_set_attacks(valid_data):
    valid_attack_size = int(batch_attack_size * (valid_set_split / test_set_split))
    misc.set_random_seed(None)  # avoid seed random_seed to true sampling
    # valid_attack_data = valid_data.sample(valid_attack_size, replace=False, axis=0)
    valid_attack_data = sample_benign_phish(valid_data, valid_attack_size, split_benign_percentages,
                                            split_phish_percentages)
    misc.set_random_seed(random_seed)
    valid_num_detector_changes_lst.clear()
    valid_magnitude_changes_lst.clear()
    dim_min_eps_benign_to_phish_attacks.clear()
    dim_min_eps_phish_to_benign_attacks.clear()

    start = datetime.now()
    print("started attacking " + ' epoch_' + str(epoch) + " validation")
    mode = 'valid'
    valid_bad_samples = 0

    if misclassify_attack is True:
        attack_valid_benign_to_phish.clear()
        attack_valid_phish_to_benign.clear()
        valid_attack_data.apply(process_all_urls_misclassify_attack, axis=1)
        end = datetime.now()
        duration = (end - start).total_seconds()
        if act_deterministically_flag is True:
            total_attack_valid_benign_to_phish = np.sum(attack_valid_benign_to_phish)
            total_attack_valid_phish_to_benign = np.sum(attack_valid_phish_to_benign)
        else:
            total_attack_valid_benign_to_phish = np.mean(attack_valid_benign_to_phish) if len(
                attack_valid_benign_to_phish) > 0 else 0
            total_attack_valid_phish_to_benign = np.mean(attack_valid_phish_to_benign) if len(
                attack_valid_phish_to_benign) > 0 else 0
        print("num misclassify benign valid to phish:" + str(total_attack_valid_benign_to_phish))
        print("num misclassify phish valid to benign:" + str(total_attack_valid_phish_to_benign))
        attack_metrics(act_deterministically_flag, misclassify_attack, "", valid_bad_samples, epsilon_to_attack,
                       attack, step_eps, split_benign_percentages,
                       split_phish_percentages, total_attack_valid_benign_to_phish,
                       total_attack_valid_phish_to_benign,
                       valid_num_detector_changes_lst, valid_magnitude_changes_lst, mode, epoch, duration)

    elif resource_attack is True:
        attack_valid_benign_benign.clear()
        attack_valid_phish_phish.clear()
        valid_attack_data.apply(process_all_urls_resources_attack, axis=1)
        end = datetime.now()
        duration = (end - start).total_seconds()
        if act_deterministically_flag is True:
            total_attack_valid_benign_benign = np.sum(attack_valid_benign_benign)
            total_attack_valid_phish_phish = np.sum(attack_valid_phish_phish)
        else:
            total_attack_valid_benign_benign = np.mean(attack_valid_benign_benign) if len(
                attack_valid_benign_benign) > 0 else 0
            total_attack_valid_phish_phish = np.mean(attack_valid_phish_phish) if len(
                attack_valid_phish_phish) > 0 else 0
        print("num successful benign valid resources attack:" + str(total_attack_valid_benign_benign))
        print("num successful phish valid resources attack:" + str(total_attack_valid_phish_phish))
        attack_metrics(act_deterministically_flag, misclassify_attack, resource_attack_mode, valid_bad_samples,
                       epsilon_to_attack, attack, step_eps, split_benign_percentages,
                       split_phish_percentages, total_attack_valid_benign_benign,
                       total_attack_valid_phish_phish,
                       valid_num_detector_changes_lst, valid_magnitude_changes_lst, mode, epoch, duration)

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    if to_histogram == True:
        plot_histogram(dim_min_eps_benign_to_phish_attacks, dim_min_eps_phish_to_benign_attacks, req_dim)
        
        
def test_set_attacks(test_data):

    start = datetime.now()
    misc.set_random_seed(None)  # avoid seed random_seed to true sampling
    # test_attack_data = test_data.sample(batch_attack_size, replace=True, axis=0)
    test_attack_data = sample_benign_phish(test_data, batch_attack_size, split_benign_percentages,
                                           split_phish_percentages)

    misc.set_random_seed(random_seed)

    test_num_detector_changes_lst.clear()
    test_magnitude_changes_lst.clear()
    dim_min_eps_benign_to_phish_attacks.clear()
    dim_min_eps_phish_to_benign_attacks.clear()
    print("started attacking " + ' epoch_' + str(epoch) + " test")
    mode = 'test'
    test_bad_samples = 0

    if misclassify_attack is True:
        attack_test_benign_to_phish.clear()
        attack_test_phish_to_benign.clear()
        test_attack_data.apply(process_all_urls_misclassify_attack, axis=1)
        end = datetime.now()
        duration = (end - start).total_seconds()
        if act_deterministically_flag is True:
            total_attack_test_benign_to_phish = np.sum(attack_test_benign_to_phish)
            total_attack_test_phish_to_benign = np.sum(attack_test_phish_to_benign)
        else:
            total_attack_test_benign_to_phish = np.mean(attack_test_benign_to_phish) if len(
                attack_test_benign_to_phish) > 0 else 0
            total_attack_test_phish_to_benign = np.mean(attack_test_phish_to_benign) if len(
                attack_test_phish_to_benign) > 0 else 0
        print("num misclassify benign test to phish:" + str(total_attack_test_benign_to_phish))
        print("num misclassify phish test to benign:" + str(total_attack_test_phish_to_benign))
        attack_metrics(act_deterministically_flag, misclassify_attack, "", test_bad_samples, epsilon_to_attack,
                       attack, step_eps, split_benign_percentages, split_phish_percentages,
                       total_attack_test_benign_to_phish, total_attack_test_phish_to_benign,
                       test_num_detector_changes_lst, test_magnitude_changes_lst, mode, epoch, duration)
    else:
        attack_test_benign_benign.clear()
        attack_test_phish_phish.clear()
        test_attack_data.apply(process_all_urls_resources_attack, axis=1)
        end = datetime.now()
        duration = (end - start).total_seconds()
        if act_deterministically_flag is True:
            total_attack_test_benign_benign = np.sum(attack_test_benign_benign)
            total_attack_test_phish_phish = np.sum(attack_test_phish_phish)
        else:
            total_attack_test_benign_benign = np.mean(attack_test_benign_benign) if len(
                attack_test_benign_benign) > 0 else 0
            total_attack_test_phish_phish = np.mean(attack_test_phish_phish) if len(
                attack_test_phish_phish) > 0 else 0
        print("num successful benign test resources attack:" + str(total_attack_test_benign_benign))
        print("num successful phish test resources attack:" + str(total_attack_test_phish_phish))
        attack_metrics(act_deterministically_flag, misclassify_attack, resource_attack_mode, test_bad_samples,
                       epsilon_to_attack, attack, step_eps, split_benign_percentages, split_phish_percentages,
                       total_attack_test_benign_benign, total_attack_test_phish_phish,
                       test_num_detector_changes_lst, test_magnitude_changes_lst, mode, epoch, duration)

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    if to_histogram == True:
        plot_histogram(dim_min_eps_benign_to_phish_attacks, dim_min_eps_phish_to_benign_attacks, req_dim)

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

    BENIGN, PHISH = env.get_benign_phish_labels_indices()

    agent = CetraAgent(train_mode=True, load_model=to_load_flag, path_to_load=load_model_path,
                       input_size=env.observation_space.shape[0], output_size=env.action_space.n)

    urls, labels = load_dataset(urls_df)

    train_data, valid_data, test_data = dataset_split(urls, labels, test_set_split, valid_set_split)

    print(DATASET + '_exp_' + str(experiment_number))

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    agent.set_act_deterministically_flag(True)
    flag_times = "REAL"

    for epoch in range(1, adversarial_sampling_epochs + 1):

        if attack_valid_data is True:
            validation_set_attacks(valid_data)


        test_set_attacks(test_data)
        
