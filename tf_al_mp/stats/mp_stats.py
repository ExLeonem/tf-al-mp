import tensorflow as tf
from ..utils.moment_propagation import MP


class MpStats:


    @staticmethod
    def loss(self, predictions, inputs, targets, **kwargs):
        """

        """
        exp, _var = predictions
        exp, _var = MP.Gaussian_Softmax(exp, _var)

        loss_fn = tf.keras.losses.get(self._model.loss)
        loss = loss_fn(targets, exp)
        return {"loss": loss}


    @staticmethod
    def accuracy(self, predictions, inputs, targets, **kwargs):
        """

        """
        exp, _var = predictions
        exp, _var = MP.Gaussian_Softmax(exp, _var)

        extended_pred = self._problem.extend_binary_prediction(exp)
        labels = np.argmax(extended_pred, axis=1)
        acc = np.mean(labels == targets)
        return {"acc": acc}

    
    @staticmethod
    def auc(self, predictions, inputs, targets, **kwargs):
        pass


    @staticmethod
    def f1(self, predictions, inputs, targets, **kwargs):
        # https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score