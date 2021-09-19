import numpy as np
import tensorflow as tf
from scipy.stats import norm

from tf_al.wrapper import Model
from ..utils.moment_propagation import MP



class MomentPropagation(Model):
    """
        Takes a regular MC Dropout model as input, that is used for fitting.
        For evaluation a moment propagation model is created an used 

    """

    def __init__(self, model, config=None, verbose=False, **kwargs):
        super(MomentPropagation, self).__init__(model, config, model_type="moment_propagation", verbose=verbose, **kwargs)

        self.__verbose = verbose
        self.__mp_model = self._create_mp_model(model)
        self.__compile_params = None


    def __call__(self, inputs, **kwargs):
        return self.__mp_model.predict(inputs, **kwargs)

    
    def fit(self, *args, **kwargs):
        history = super().fit(*args, **kwargs)
        self.__mp_model = self._create_mp_model(self._model)
        return history


    def compile(self, **kwargs):

        if kwargs is not None and len(kwargs.keys()) > 0:
            print("Set compile params")
            self.__compile_params = kwargs

        self._model.compile(**self.__compile_params)
        # self.__base_model.compile(**self.__compile_params)


    def evaluate(self, inputs, targets, **kwargs):
        """
            Evaluates the performance of the model.

            Parameters:
                inputs (numpy.ndarray): The inputs of the neural network.
                targets (numpy.ndarray): The true output values.
                batch_size (int): The number of batches to use for the prediction. (default=1)
                

            Returns:
                (list()) Loss and accuracy of the model.
        """

        self.logger.info("Evaluate kwargs: {}".format(kwargs))

        exp, _var = self.__mp_model.predict(inputs, **kwargs)
        exp, _var = MP.Gaussian_Softmax(exp, _var)

        loss, acc = self.__evaluate(exp, targets)
        return {"loss": loss, "accuracy": acc}


    def __evaluate(self, prediction, targets):
        """
            Calculate the accuracy and the loss
            of the prediction.

            Parameters:
                prediction (numpy.ndarray): The predictions made.
                targets (numpy.ndarray): The target values.

            Returns:
                (list()) The accuracy and 
        """

        self.logger.info("Prediction shape: {}".format(prediction.shape))

        loss_fn = tf.keras.losses.get(self._model.loss)
        loss = loss_fn(targets, prediction)
        
        prediction = self._problem.extend_binary_prediction(prediction)
        labels = np.argmax(prediction, axis=1)
        acc = np.mean(labels == targets)
        return [np.mean(loss.numpy()), acc]
        

    def _create_mp_model(self, model, drop_softmax=True):
        """
            Transforms the set base model into an moment propagation model.

            Returns:
                (tf.Model) as a moment propagation model.
        """
        _mp = MP()
        
        # Remove softmax layer for inference with mp model
        if "softmax" in model._layers[-1].name and drop_softmax:
            cloned_model = tf.keras.models.clone_model(model)
            cloned_model.set_weights(model.get_weights())
            cloned_model._layers.pop(-1)
            return _mp.create_MP_Model(model=cloned_model, use_mp=True)

        return _mp.create_MP_Model(model=model, use_mp=True, verbose=self.__verbose)


    def variance(self, predictions):
        expectation, variance = predictions
        variance = self._problem.extend_binary_predictions(variance)
        return self.__cast_tensor_to_numpy(variance)


    def expectation(self, predictions):
        expectation, variance = predictions
        expectation = self._problem.extend_binary_predictions(expectation)
        return self.__cast_tensor_to_numpy(expectation) 



    # -----------
    # Metrics hooks
    # -----------------------

    def _on_evaluate_loss(self, predictions, inputs, targets, **kwargs):
        """

        """
        self.logger.info("Evaluate kwargs: {}".format(kwargs))
        exp, _var = predictions
        exp, _var = MP.Gaussian_Softmax(exp, _var)

        loss_fn = tf.keras.losses.get(self._model.loss)
        loss = loss_fn(targets, exp)
        return {"loss": loss}


        prediction = self._problem.extend_binary_prediction(prediction)
        labels = np.argmax(prediction, axis=1)
        acc = np.mean(labels == targets)
        return [np.mean(loss.numpy()), acc]



    def _on_evaluate_acc(self, **kwargs):
        """

        """
        pass


    # --------
    # Weights loading
    # ------------------

    # def load_weights(self):
    #     path = self._checkpoints.PATH
    #     self.__base_model.load_weights(path)

    # def save_weights(self):

    #     path = self._checkpoints.PATH
    #     self.__base_model.save_weights(path)


    # --------
    # Utilities
    # ---------------
    
    def __cast_tensor_to_numpy(self, values):
        """
            Cast tensor objects of different libraries to
            numpy arrays.
        """

        # Values already of type numpy.ndarray
        if isinstance(values, np.ndarray):
            return values

        # values = tf.make_ndarray(values)
        values = values.numpy()
        return values


    # ----------------
    # Custom acquisition functions
    # ---------------------------

    def get_query_fn(self, name):

        if name == "max_entropy":
            return self.__max_entropy
        
        if name == "bald":
            return self.__bald
        
        if name == "max_var_ratio":
            return self.__max_var_ratio

        if name == "std_mean":
            return self.__std_mean


    def __max_entropy(self, data, **kwargs):
        # Expectation and variance of form (batch_size, num_classes)
        # Expectation equals the prediction
        exp, var = self.__mp_model.predict(x=data)
        predictions = MP.Gaussian_Softmax(exp, var)

        # Need to scaled values because zeros
        class_probs = self.expectation(predictions)
        
        class_prob_logs = np.log(np.abs(class_probs) + .001)
        return -np.sum(class_probs*class_prob_logs, axis=1)

    
    def __bald(self, data, num_samples=100, **kwargs):
        exp, var = self.__mp_model.predict(x=data)

        exp_shape = list(exp.shape)
        output_shape = tuple([num_samples] + exp_shape) # (num samples, num datapoints, num_classes)
        sampled_data = norm(exp, np.sqrt(var)).rvs(size=output_shape)
        class_sample_probs = tf.keras.activations.softmax(tf.convert_to_tensor(sampled_data))

        sample_entropy = np.sum(class_sample_probs*np.log(class_sample_probs+.001), axis=-1)
        disagreement = np.sum(sample_entropy, axis=0)/num_samples

        exp, _var = MP.Gaussian_Softmax(exp, var)
        entropy = np.sum(exp*np.log(exp+.001), axis=1)

        return -entropy+disagreement


    def __max_var_ratio(self, data, **kwargs):
        exp, var = self.__mp_model.predict(x=data)
        predictions = MP.Gaussian_Softmax(exp, var)
        expectation = self.expectation(predictions)

        col_max_indices = np.argmax(expectation, axis=1)        
        row_indices = np.arange(len(data))
        max_var_ratio = 1- expectation[row_indices, col_max_indices]
        return max_var_ratio

    
    def __std_mean(self, data, **kwargs):
        exp, var = self.__mp_model.predict(data, **kwargs)
        predictions = MP.Gaussian_Softmax(exp, var)
        variance = self.variance(predictions)
        std = np.square(variance)
        return np.mean(std, axis=-1)
    


    # ----------
    # Setter/-Getter
    # ---------------------

    def get_mp_model(self):
        return self.__mp_model
