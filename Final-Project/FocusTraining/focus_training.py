import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

import numpy as np
from tqdm.auto import tqdm


class EpochMetricTracker:
    """EpochMetricTracker keeps track of metrics w.r.t. a single epoch.

    Parameters
    ----------
    metrics : list
        List of the metrics that we want to track.  Loss is automatically
        included.

    Attributes
    ----------
    total_loss : float
        Keeps track of the complete loss for the entire epoch.
    total_y_true : tensor
        Integer tensor holding all of the true labels for the epoch.
    total_y_pred : tensor
        Tensor holding all of the predictions for an epoch.
    metrics

    """

    def __init__(self, metrics):
        self.metrics = metrics
        self.total_loss = 0
        self.total_y_true = []
        self.total_y_pred = []

    def update(self, loss, y_true, y_pred):
        self.total_loss += loss
        y_pred_logit = np.argmax(y_pred, axis=1)

        self.total_y_true = np.hstack((self.total_y_true, y_true))
        self.total_y_pred = np.hstack((self.total_y_pred, y_pred_logit))
        self.total_observations = len(self.total_y_true)

    def get_display(self, prefix, n_batches_elapsed=0, n_observations=0,
                    n_decimals=3):
        # Prefix should be '' for training and 'val_' for validation
        avg_loss = self.total_loss / self.total_observations

        display = {
            prefix + 'loss': str(self._round_value(avg_loss, n_decimals))}
        for metric in self.metrics:
            val = metric.compute_display_value(
                self.total_y_true, self.total_y_pred, n_decimals)
            display[str(prefix) + str(metric.name)] = val

        return display

    def get_values_with_metric(self, monitor_metric):
        if monitor_metric == 'loss':
            return self.total_loss / self.total_observations

        for metric in self.metrics:
            if metric.name == monitor_metric:
                val = metric.compute(self.total_y_true, self.total_y_pred)
                return val
            else:
                continue
        raise ValueError(f'Metric {monitor_metric} is not watched.')

    def _round_value(self, val, n_decimals):
        return np.round(val, n_decimals)


class ModelMetricTracker:
    """ModelMetricTracker tracks metrics over all epochs.

    Parameters
    ----------
    metrics : list
        List of the metrics that we want to track.  Loss is automatically
        included.
    prefix : str
        To keep track if the metrics are for training or validation.  '' for
        training and 'val_' for validation.

    Attributes
    ----------
    history : dict
        For each metric name (key) there is a corresponding list of the metric
        being tracked over all epochs.
    """

    def __init__(self, metrics, prefix):
        self.metrics = metrics
        self.prefix = prefix  # '' for train, 'val_' for validation
        self.history = {m.name: [] for m in self.metrics}
        self.history['loss'] = []

    def update(self, epoch_metric_tracker):
        avg_loss = epoch_metric_tracker.total_loss / \
            epoch_metric_tracker.total_observations
        self.history['loss'].append(avg_loss)

        for metric in self.metrics:
            val = metric.compute(
                epoch_metric_tracker.total_y_true, epoch_metric_tracker.total_y_pred)
            self.history[metric.name].append(val)


class ByClassHistory:
    """ByClassHistory keeps track of histories for each class.

    Parameters
    ----------
    classes : list
        List of the class names in the dataset.
    metrics : list
        List of the metrics that we want to track.  Loss is automatically
        included.
    prefix : str
        To keep track if the metrics are for training or validation.  '' for
        training and 'val_' for validation.

    Attributes
    ----------
    class_histories : dict
        Dictionary with each class being the key and each value is a
        ModelMetricTracker to track the metrics for the class over all epochs.

    """

    def __init__(self, classes, metrics, prefix='val_'):
        self.classes = classes
        self.metrics = metrics
        self.prefix = prefix
        self.class_histories = {c: ModelMetricTracker(
            self.metrics, self.prefix) for c in self.classes}

    def update(self, metrics_by_class):
        for class_name, metrics in metrics_by_class.items():
            self.class_histories[class_name].update(metrics)


class DatasetStruct:
    """DatasetStruct cleanly wraps a tf dataset to keep track of useful items
    specific to the dataset.

    Parameters
    ----------
    name : str
        Name of the dataset.
    dataset : tf dataset
        Actual dataset.

    Attributes
    ----------
    n_observations : int
        Total number of observations in the dataset (used for steps per epoch).
    """

    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.n_observations = sum(1 for _ in dataset)


class StartToFocus:
    """StartToFocus is essentially a callback for the training loop to know when
    to start running our modified training procedure.  If time_to_focus returns
    true, then the training loop knows to use the modified training procedure
    for that epoch.

    Attributes
    ----------
    focus : bool
        Keeps track of whether the training loop is currently focusing.

    """

    def __init__(self):
        self.focus = False

    def time_to_focus(self, train_history, total_val_history, class_val_history):
        return True

    def toggle_focus(self):
        self.focus = not self.focus


class FocusLearning:
    """FocusLearning is our modified training procedure where we take gradients
    w.r.t. the worst performing classes in hopes to "focus" on where we are
    doing poorly.

    Parameters
    ----------
    model : Keras model
        The model we want to train (note: not compiled).
    dset_name : str
        tfds string name of the dataset we want to load.
    class_names : list
        List of the names of the classes for better processing on displays and
        filtering datasets by labels.
    metrics : list
        List of metrics that we want to track during training.
    optimizer : keras optimizer
        Optimizer to use during training.
    start_to_focus : StartToFocus
        Instance of StartToFocus that will call the time_to_focus function to
        initiate the modified training procedure.

    Attributes
    ----------
    n_classes : int
        Number of classes in the dataset.
    n_train_observations : int
        Number of training observations for steps per epoch.
    n_test_observations : int
        Number of testing observations for steps per epoch.
    train_dataset : DatasetStruct
        Training dataset.
    test_dataset : DatasetStruct
        Testing dataset.
    split_test_datasets : dict
        Dictionary of split testing sets based on the class label (key) and the
        filtered dataset (value).
    loss_fn : keras loss
        Loss function used during training.

    """

    def __init__(self, model, dset_name, class_names, metrics, optimizer,
                 start_to_focus):
        # Model we want to train
        self.model = model

        # Load in data
        dset, info = tfds.load(dset_name, with_info=True, as_supervised=True)

        # Get the number of classes
        self.n_classes = info.features['label'].num_classes

        # Split into train/test
        self.n_train_observations = info.splits['train'].num_examples
        self.n_test_observations = info.splits['test'].num_examples

        # Create dataset structs for easier processing
        self.train_dataset = DatasetStruct('full_train', dset['train'])
        self.test_dataset = DatasetStruct('full_test', dset['test'])

        # For easier filtering on the focus part
        self.class_names = class_names

        # Filter the test/train dataset by classes
        self.split_test_datasets = self._filter_by_class(self.test_dataset)
        self.split_train_datasets = self._filter_by_class(self.train_dataset)

        # When to kickstart the modified training procedure
        self.start_to_focus = start_to_focus

        # List of metrics that we want to compute (loss is implied)
        self.metrics = metrics

        # Optimizer to use for training
        self.optimizer = optimizer

        # For some reason I kept getting issues passing this in a parameter, but
        # this is what we use the whole time anyways.
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

    def _filter_by_class(self, dataset):
        """_filter_by_class filters a dataset by all the classes in the dataset.

        Parameters
        ----------
        dataset : DatasetStruct
            Dataset that we want to filter.

        Returns
        -------
        dict
            Key: classname, value: DatasetStruct of the filtered dataset for the
            given key.

        """
        # Filter dataset to labels with current class_name
        split_test_datasets = {
            class_name: DatasetStruct(class_name,
                                      dataset.dataset.filter(lambda x, y: y == class_name))
            for class_name in self.class_names}

        return split_test_datasets

    def _evaluate(self, dataset, batch_size, desc='', steps_per_epoch=0,
                  prefix='val_'):
        """_evaluate does an evaluation loop on a single dataset.

        Parameters
        ----------
        dataset : tf dataset
            TF dataset to evaluate the model on.
        batch_size : int
            Batch size to use.
        desc : str
            Description of the progress bar to use.
        steps_per_epoch : int
            Number of steps in an epoch.
        prefix : str
            Keep track of training and validation metrics.

        Returns
        -------
        EpochMetricTracker
            Metrics from the evaluation loop.

        """

        # Track metrics
        eval_metric_tracker = EpochMetricTracker(self.metrics)

        # Batch the testing dataset (don't care about shuffling or anything since just evaluating)
        dataset = dataset.batch(batch_size)

        l_bar = '{desc}: {percentage:.3f}%|'
        r_bar = '| {n_fmt}/{total_fmt} {elapsed}<{remaining}, ' '{rate_fmt}{postfix}'
        bar_format = '{l_bar}{bar}{r_bar}'

        eval_pbar = tqdm(enumerate(dataset), bar_format=bar_format, desc=desc,
                         total=steps_per_epoch, leave=False)

        for batch_idx, (x_batch_test, y_batch_test) in eval_pbar:

            # Run the forward pass.
            # Logits for this minibatch
            logits = self.model(x_batch_test, training=False)

            # Compute the loss value for this minibatch.
            loss_value = self.loss_fn(y_batch_test, logits)

            # User feedback
            eval_metric_tracker.update(
                loss_value.numpy(), y_batch_test, logits)
            postfix_metrics = eval_metric_tracker.get_display(prefix)
            eval_pbar.set_postfix(postfix_metrics)

        return eval_metric_tracker

    def _evaluate_by_class(self, batch_size):
        """_evaluate_by_class evaluates the performance of each test class based
        on the current model.

        Parameters
        ----------
        batch_size : int
            Batch size to use.

        Returns
        -------
        dict
            Dictionary of key: class name, value: EpochMetricTracker of the
            metrics during evaluation.

        """
        metrics_by_class = {}

        for _, dataset_struct in self.split_test_datasets.items():
            # Get the dataset name and values
            dataset_name, dataset = dataset_struct.name, dataset_struct.dataset
            steps_per_epoch = self._compute_steps_per_epoch(
                dataset_struct.n_observations, batch_size)
            desc = 'Evaluating Class ' + str(dataset_name)

            # Evaluate performance on the dataset
            eval_metrics = self._evaluate(
                dataset, batch_size, desc=desc, steps_per_epoch=steps_per_epoch)

            # Add to the by class metrics
            metrics_by_class[dataset_name] = eval_metrics

        return metrics_by_class

    def _train(self, dataset, batch_size, epoch_tqdm_desc, steps_per_epoch,
               grad_penalty=1, **kwargs):
        """_train trains the model for 1 epoch based on the specified
        parameters.

        Parameters
        ----------
        dataset : DatasetStruct
            Dataset we want to train on (can be full train/filtered train).
        batch_size : int
            Batch size to use.
        epoch_tqdm_desc : str
            Description of the progress bar.
        steps_per_epoch : int
            Number of steps in an epoch.
        grad_penalty : float
            Penalty to the gradients we wan to apply in this training cycle.
        **kwargs
            Low level arguments.  For example prefix for generating displays.

        Returns
        -------
        progressbar, metric display, EpochMetricTracker
            Progressbar we will update after evaluation, display we want to
            update, and the metrics for the epoch.

        """

        # Expects dataset to be a DatasetStruct, so get the dataset only
        dataset = dataset.dataset

        # Put into tf format
        grad_penalty = tf.constant(grad_penalty, dtype=tf.float32)

        l_bar = '{desc}: {percentage:.3f}%|'
        r_bar = '| {n_fmt}/{total_fmt} {elapsed}<{remaining}, ' '{rate_fmt}{postfix}'
        bar_format = '{l_bar}{bar}{r_bar}'

        # Batch/shuffle the dataset
        training_dataset = dataset.shuffle(1000).batch(batch_size)

        # Have to define loss here for whatever reason
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Keep track of metrics
        train_metrics_tracker = EpochMetricTracker(self.metrics)

        train_pbar = tqdm(enumerate(training_dataset), bar_format=bar_format,
                          desc=epoch_tqdm_desc, total=steps_per_epoch)

        for batch_idx, (x_batch_train, y_batch_train) in train_pbar:
            with tf.GradientTape() as tape:
                # Run the forward pass.
                logits = self.model(x_batch_train, training=True)

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, self.model.trainable_weights)

            # Add in gradient clipping/penalty if wanted
            grads = [g * grad_penalty for g in grads]

            # Update weights
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_weights))

            # User feedback
            train_metrics_tracker.update(
                loss_value.numpy(), y_batch_train, logits)
            postfix_metrics = train_metrics_tracker.get_display(**kwargs)
            train_pbar.set_postfix(postfix_metrics)

        return (train_pbar, train_metrics_tracker.get_display(**kwargs),
                train_metrics_tracker)

    def _compute_steps_per_epoch(self, total_observations, batch_size):
        """Computes the steps_per_epoch.

        Parameters
        ----------
        total_observations : int
            Number of observations in the dataset.
        batch_size : int
            Batch size being used.

        Returns
        -------
        int
            Number of steps per epoch.

        """
        return total_observations // batch_size + int(total_observations % batch_size > 0)

    def _combine_metrics_by_classes(self, metrics_by_class):
        """Combines the metrics for all classes to get general view of
        performance for a specific epoch.

        Parameters
        ----------
        metrics_by_class : dict
            Dictionary with all the metrics for each class.

        Returns
        -------
        EpochMetricTracker
            Combined metrics (from all classes) for an epoch.

        """
        total_metrics_tracker = EpochMetricTracker(self.metrics)

        total_loss = 0
        total_y_true = []
        total_y_pred = []
        for _, metrics in metrics_by_class.items():
            total_loss += metrics.total_loss
            total_y_true = np.hstack((total_y_true, metrics.total_y_true))
            total_y_pred = np.hstack((total_y_pred, metrics.total_y_pred))

        # Set the values
        total_metrics_tracker.total_loss = total_loss
        total_metrics_tracker.total_y_true = total_y_true
        total_metrics_tracker.total_y_pred = total_y_pred
        total_metrics_tracker.total_observations = len(total_y_true)

        return total_metrics_tracker

    def _get_display_metrics_by_class(self, metrics_by_class, prefix):
        """Formats the display for each class' metrics.

        Parameters
        ----------
        metrics_by_class : dict
            Dictionary with all the metrics for each class.
        prefix : str
            Keep track of train/testing for display.

        Returns
        -------
        dict
            Formatted display of the class metrics.

        """
        class_metrics_display = {}

        for class_name, metrics in metrics_by_class.items():
            class_metrics_display.update(metrics.get_display(
                str(prefix) + str(class_name) + '_'))

        return class_metrics_display

    def _create_worst_k_dataset(self, worst_k_classes):
        """Creates a dataset with just the worst k classes (from training set).

        Parameters
        ----------
        worst_k_classes : list
            List of (class_name, monitor_metric_score)

        Returns
        -------
        DatasetStruct
            Dataset with only worst k classes from the training set.

        """
        worst_k_dataset = self.split_train_datasets[worst_k_classes[0][0]].dataset

        # For shuffling
        total_in_worst_k = self.split_train_datasets[worst_k_classes[0]
                                                     [0]].n_observations

        for dataset_name, _ in worst_k_classes[1:]:
            worst_k_dataset = worst_k_dataset.concatenate(
                self.split_train_datasets[dataset_name].dataset)
            # Shuffle better
            current_total = self.split_train_datasets[dataset_name].n_observations
            total_in_worst_k += current_total
            worst_k_dataset.shuffle(total_in_worst_k)
            total_in_worst_k -= current_total

        dataset_name = 'worst_' + \
            '_'.join([str(val) for val in list(zip(*worst_k_classes))[0]])

        return DatasetStruct(dataset_name, worst_k_dataset)

    def _get_worst_k(self, metrics_by_class, monitor_metric, k):
        """Gets the worst k classes.

        Parameters
        ----------
        metrics_by_class : dict
            Dictionary holding the metrics for each class.
        monitor_metric : str
            Metric that quantifies "worst".
        k : int
            Number of classes that we want to focus on.

        Returns
        -------
        list
            List of k worst classes in form (class_name, monitor_metric) in
            descending order.

        """
        monitor_metric_list = []

        for class_name, metrics in metrics_by_class.items():
            val = metrics.get_values_with_metric(monitor_metric)
            monitor_metric_list.append((class_name, val))

        # Loss worst is highest loss
        if monitor_metric == 'loss':
            sorted_metrics = sorted(monitor_metric_list,
                                    key=lambda x: x[1], reverse=True)[:k]
        # Other metrics we are using worst is lower value
        else:
            sorted_metrics = sorted(monitor_metric_list,
                                    key=lambda x: x[1])[:k]

        return sorted_metrics

    def _format_worst_k(self, worst_k_classes, metric):
        """Puts worst k in pretty printing format.

        Parameters
        ----------
        worst_k_classes : list
            List of k worst classes in form (class_name, monitor_metric) in
            descending order.
        metric : str
            Metric that quantifies "worst".

        Returns
        -------
        dict
            Formatted way to print.

        """
        return {'Class ' + str(class_name): 'val_' + str(metric) + ': ' + str(np.round(val, 3))
                for class_name, val in worst_k_classes}

    def train(self, n_epochs, batch_size=16, normal_grad_penalty=1,
              val_monitor_metric='loss', worst_k=2, focus_penalty=1e-2):
        """Trains the model for a certain number of epochs.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train the model.
        batch_size : int
            Batch size to use.
        normal_grad_penalty : float
            Penalty to apply to gradients from the tradional training loop.
        val_monitor_metric : str, default: 'loss'
            Valiation metric to monitor for getting the "worst" classes.
        worst_k : int
            How many classes we want to focus on with the modified procedure.
        focus_penalty : float
            Penalty to apply to gradients from the focus training loop.

        Returns
        -------
        train_history, total_val_history, class_val_history
            Histories for training, total valiation, and by class validation.

        """

        # Store histories
        class_val_history = ByClassHistory(
            self.class_names, self.metrics, prefix='val_')
        total_val_history = ModelMetricTracker(
            metrics=self.metrics, prefix='val_')
        train_history = ModelMetricTracker(metrics=self.metrics, prefix='')

        # Run for each epoch
        for epoch in range(n_epochs):

            # For user feedback
            epoch_tqdm_desc = 'Epoch ' + str(epoch + 1)

            # Train the model on the training data
            train_steps_per_epoch = self._compute_steps_per_epoch(
                self.n_train_observations, batch_size)
            train_pbar, train_metrics_display, train_metrics_tracker = self._train(self.train_dataset,
                                                                                   batch_size, epoch_tqdm_desc,
                                                                                   train_steps_per_epoch,
                                                                                   grad_penalty=normal_grad_penalty,
                                                                                   prefix='')

            # Store history
            train_history.update(train_metrics_tracker)

            # Evaluate model (by class and get total value)
            metrics_by_class = self._evaluate_by_class(batch_size)

            # Display evaluation metrics
            class_metrics_display = self._get_display_metrics_by_class(
                metrics_by_class, 'val_')
            val_total_metrics_tracker = self._combine_metrics_by_classes(
                metrics_by_class)
            val_metrics_display = val_total_metrics_tracker.get_display('val_')
            val_metrics_display.update(class_metrics_display)
            combined = {**train_metrics_display, **val_metrics_display}
            train_pbar.set_postfix(combined)
            train_pbar.update()
            train_pbar.close()

            # If modified procedure kicks in, get grads w.r.t. worst worst_k classes
            if self.start_to_focus.time_to_focus(train_history, total_val_history, class_val_history):

                # Get the worst performing k classes
                worst_k_classes = self._get_worst_k(
                    metrics_by_class, val_monitor_metric, worst_k)
                print(
                    f'Worst performing classes: {self._format_worst_k(worst_k_classes, val_monitor_metric)}')

                # Create a dataset with just the worst k classes
                worst_k_dataset = self._create_worst_k_dataset(worst_k_classes)

                # For user feedback
                epoch_tqdm_desc = 'Epoch ' + \
                    str(epoch + 1) + ' w.r.t ' + worst_k_dataset.name

                # Train the model w.r.t. the worst classes only
                train_steps_per_epoch = self._compute_steps_per_epoch(
                    worst_k_dataset.n_observations, batch_size)
                train_pbar, train_metrics_display, train_metrics_tracker = self._train(worst_k_dataset, batch_size,
                                                                                       epoch_tqdm_desc,
                                                                                       train_steps_per_epoch,
                                                                                       grad_penalty=focus_penalty,
                                                                                       prefix='')

                # Evaluate model (by class and get total value)--again
                metrics_by_class = self._evaluate_by_class(batch_size)

                # Display evaluation metrics
                class_metrics_display = self._get_display_metrics_by_class(
                    metrics_by_class, 'val_')
                val_total_metrics_tracker = self._combine_metrics_by_classes(
                    metrics_by_class)
                val_metrics_display = val_total_metrics_tracker.get_display(
                    'val_')
                val_metrics_display.update(class_metrics_display)
                combined = {**train_metrics_display, **val_metrics_display}
                train_pbar.set_postfix(combined)
                train_pbar.update()
                train_pbar.close()

            # End of epoch, store validation results
            class_val_history.update(metrics_by_class)
            total_val_history.update(val_total_metrics_tracker)

        return train_history, total_val_history, class_val_history
