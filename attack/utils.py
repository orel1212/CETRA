from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


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
