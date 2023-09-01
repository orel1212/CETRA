from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def calc_T_N_rates(true_positives, true_negatives, false_positives, false_negatives):
    '''
    function that calcs rates:
    Sensitivity - TPR = true_positives/true_positives+false_negatives
    FPR = false_positives / (false_positives + true_negatives)
    Specificity - TNR = true_negatives / true_negatives + false_positives
    FNR = false_negatives / false_negatives + true_positives
    :param true_positives:
    :param true_negatives:
    :param false_positives:
    :param false_negatives:
    :return: TPR,FPR,TNR,FNR
    '''
    if true_positives == 0:
        TPR = 0
    else:
        TPR = true_positives / (true_positives + false_negatives)
    if false_positives == 0:
        FPR = 0
    else:
        FPR = false_positives / (false_positives + true_negatives)
    if true_negatives == 0:
        TNR = 0
    else:
        TNR = true_negatives / (true_negatives + false_positives)
    if false_negatives == 0:
        FNR = 0
    else:
        FNR = false_negatives / (false_negatives + true_positives)

    return {'TPR': TPR, 'FPR': FPR, 'TNR': TNR, 'FNR': FNR}


def calc_accurary(preds):
    true_rate = preds.count('TP') + preds.count('TN')
    ratio = true_rate / len(preds) if len(preds) > 0 else 0
    accuracy = 100 * ratio
    return accuracy


def calc_precision_recall(preds):
    true_positives = preds.count('TP')
    false_positives = preds.count('FP')
    false_negatives = preds.count('FN')
    if true_positives == 0:
        precision, recall = 0, 0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
    return (precision, recall)


def calc_F1_score(precision, recall):
    if precision + recall == 0:
        return 0
    else:
        F1_score = (2 * precision * recall) / (precision + recall)
        return F1_score


def calc_auc(preds):
    '''
    calculate auc based on FPR and TPR
    divide the space into 2 parts: a triangle and a trapezium:
        The triangle will have area TPR*FRP/2.
        The trapezium (1-FPR)*(1+TPR)/2 = 1/2 - FPR/2 + TPR/2 - TPR*FPR/2.
    Total Area = AUC = 1/2 - FPR/2 + TPR/2
    '''

    true_positives = preds.count('TP')
    true_negatives = preds.count('TN')
    false_positives = preds.count('FP')
    false_negatives = preds.count('FN')
    rates_dict = calc_T_N_rates(true_positives, true_negatives, false_positives, false_negatives)

    AUC = 1 / 2 - rates_dict['FPR'] / 2 + rates_dict['TPR'] / 2
    return AUC


def cast_by_threshold(val, threshold=0.5):
    result = 1
    if val < threshold:
        result = 0
    return result


def load_dataset(urls_df):
    urls_columns = urls_df.columns.values.tolist()
    try:
        index_url_column = urls_columns.index('url')
        index_xgb_column = urls_columns.index('XGBoost')
        index_pdrcnn_column = urls_columns.index('PDRCNN')
        index_cnn_column = urls_columns.index('CnnModel')
        index_classurnn_column = urls_columns.index('ClassificationUsingRNN')
        index_ffnn_column = urls_columns.index('FFNN')
        index_xgb_time_column = urls_columns.index('XGBoostTime')
        index_pdrcnn_time_column = urls_columns.index('PDRCNNTime')
        index_cnn_time_column = urls_columns.index('CnnModelTime')
        index_classurnn_time_column = urls_columns.index('ClassificationUsingRNNTime')
        index_ffnn_time_column = urls_columns.index('FFNNTime')
        index_label_column = urls_columns.index('label')
    except:
        print("Not found desired columns in dataframe. Exiting...")
    total_urls = urls_df.drop(columns=['label'])
    total_labels = urls_df['label']
    lab_1 = urls_df['label'].values.tolist().count(1)
    lab_0 = urls_df['label'].values.tolist().count(0)
    print("Total Urls Phishing: " + str(lab_1))
    print("Total Urls Benign: " + str(lab_0))

    print("Total Urls: " + str(len(total_urls)))

    return total_urls, total_labels


def dataset_split(urls, labels, test_set_split, valid_set_split, seed=41):
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(urls, labels, test_size=test_set_split,
                                                                    random_state=seed, stratify=labels)

    print("Train/Test split size: " + str(test_set_split))

    x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=valid_set_split,
                                                          random_state=seed, stratify=y_train_valid)

    print("Train/Valid split size: " + str(valid_set_split))

    train_data = pd.concat([x_train, y_train], axis=1)
    valid_data = pd.concat([x_valid, y_valid], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)

    print("Total Urls to train: " + str(len(train_data)))
    print("Total Urls to validation: " + str(len(valid_data)))
    print("Total Urls to test: " + str(len(test_data)))

    return train_data, valid_data, test_data


def reward_functions_for_agent(DATASET, experiment_number):
    h = 1
    p = 6
    if DATASET == 'Wang':
        w = 0.8925
    elif DATASET == 'Bahnsen':
        w = 0.5301

    if experiment_number == 0:  # for test - original
        reward_func = lambda r: r if r < 1 else min(1 + np.log2(r), 6)  # reward_func
        t_func = lambda r: r  # Reward for TP or TN
        fp_func = lambda r: -r  # Reward for FP
        fn_func = lambda r: -r  # Reward for FN

    elif experiment_number == 1:
        reward_func = lambda r: h * r / w if r / w < 1 else min(h * (1 + np.log2(r / w)), p)  # reward_func
        t_func = lambda r: r  # Reward for TP or TN
        fp_func = lambda r: -1 * r  # Reward for FP
        fn_func = lambda r: -1 * r  # Reward for FN

    elif experiment_number == 2:
        reward_func = lambda r: h * r / w if r / w < 1 else min(h * (1 + np.log2(r / w)), p)  # reward_func
        t_func = lambda r: r  # Reward for TP or TN
        fp_func = lambda r: -10 * r  # Reward for FP
        fn_func = lambda r: -10 * r  # Reward for FN

    elif experiment_number == 3:
        reward_func = lambda r: h * r / w if r / w < 1 else min(h * (1 + np.log2(r / w)), p)  # reward_func
        t_func = lambda r: 1  # Reward for TP or TN
        fp_func = lambda r: -1 * r  # Reward for FP
        fn_func = lambda r: -1 * r  # Reward for FN

    elif experiment_number == 4:
        reward_func = lambda r: h * r / w if r / w < 1 else min(h * (1 + np.log2(r / w)), p)  # reward_func
        t_func = lambda r: 10  # Reward for TP or TN
        fp_func = lambda r: -1 * r  # Reward for FP
        fn_func = lambda r: -1 * r  # Reward for FN

    elif experiment_number == 5:
        reward_func = lambda r: h * r / w if r / w < 1 else min(h * (1 + np.log2(r / w)), p)  # reward_func
        t_func = lambda r: 100  # Reward for TP or TN
        fp_func = lambda r: -1 * r  # Reward for FP
        fn_func = lambda r: -1 * r  # Reward for FN

    return reward_func, t_func, fp_func, fn_func
