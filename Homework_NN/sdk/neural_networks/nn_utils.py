import numpy as np
from keras.callbacks import Callback

def selection_by_corr(dataset, threshold):
    """
    Selects features from dataset by correlation matrix.
    :param dataset: pandas.DataFrame to select features from
    :param threshold: float value of correlation threshold
    :return: names of selected features
    """
    corr_ = (dataset.corr() * -(np.identity(dataset.shape[1]) - 1)).abs()
    while corr_.max().max() > threshold:
        args = np.unravel_index(corr_.to_numpy().argmax(), corr_.shape)
        if corr_.iloc[args[0], :].mean() > corr_.iloc[:, args[1]].mean():
            name_drop = corr_.iloc[args[0], :].name
            corr_.drop(name_drop, axis=1, inplace=True)
            corr_.drop(name_drop, axis=0, inplace=True)
        else:
            name_drop = corr_.iloc[:, args[1]].name
            corr_.drop(name_drop, axis=1, inplace=True)
            corr_.drop(name_drop, axis=0, inplace=True)
    return corr_.columns.values


class SaveBestModelCallback(Callback):
    def __init__(self, threshold: float, filepath: str, monitor_metric: str):
        super().__init__()
        self.threshold = threshold
        self.filepath = filepath
        self.monitor_metric = monitor_metric
        self.best_metric = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.monitor_metric)
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            if current_metric < self.threshold:
                path_ = self.filepath
                print('Saving model to {}'.format(path_))
                # Save model weights
                self.model.save_weights(path_)

                # Save model structure
                with open(path_ + '.json', 'w') as f:
                    f.write(self.model.to_json())




# # Create an instance of the custom callback
# save_model_callback = SaveModelCallback(threshold=0.1)

# # Train the model with the custom callback
# model.fit(x_train, y_train, epochs=10, callbacks=[save_model_callback])