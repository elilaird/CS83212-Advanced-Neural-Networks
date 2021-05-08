import pickle
import pandas as pd


def save_history(path, data):
    # Save histories
    with open(path, 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)


def read_history(path):
    # Read histories
    with open(path, 'rb') as input:
        data = pickle.load(input)

    return data

# Gets the worst performing class per epoch according to a metric name


def get_max_class_per_epoch(class_val_history, metric='loss'):
    # Idea is for loss here since the worst performing class w.r.t. loss is the
    # class with the max loss at that specific epoch.
    all_histories = [metric_tracker.history[metric]
                     for _, metric_tracker in class_val_history.class_histories.items()]

    all_histories = np.vstack(all_histories)

    return np.max(all_histories, axis=0)


def get_min_class_per_epoch(class_val_history, metric='acc'):
    # Idea is for acc here since the worst performing class w.r.t. acc is the
    # class with the min acc at that specific epoch.
    all_histories = [metric_tracker.history[metric]
                     for _, metric_tracker in class_val_history.class_histories.items()]

    all_histories = np.vstack(all_histories)

    return np.min(all_histories, axis=0)


def get_by_class_metrics(class_val_history, metric='loss'):
    # Gets the metric for each class and puts it in a dataframe for plotting
    df = pd.DataFrame({class_name: metric_tracker.history[metric]
                       for class_name, metric_tracker in class_val_history.class_histories.items()})

    return df
