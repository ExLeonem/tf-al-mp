import numpy as np
import tensorflow as tf
from scipy.stats import norm
import tensorflow as tf

from tf_al.wrapper import Model
from tf_al.utils import beta_approximated_upper_joint_entropy
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
        self._compile_params = None


    def __call__(self, inputs, **kwargs):
        return self.__mp_model.predict(inputs, **kwargs)

    
    def fit(self, *args, **kwargs):
        history = super().fit(*args, **kwargs)
        self.__mp_model = self._create_mp_model(self._model)
        return history


    def compile(self, **kwargs):

        if kwargs is not None and len(kwargs.keys()) > 0:
            self._compile_params = kwargs

        self._model.compile(**self._compile_params)
        metrics = self._create_init_metrics(kwargs)
        metric_names = self._extract_metric_names(metrics)
        self.eval_metrics = self._init_metrics("stochastic", metric_names)


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
        predictions = MP.Gaussian_Softmax(exp, _var)
        
        output_metrics = {}
        for metric in self.eval_metrics:
            output_metrics[metric.name] = metric(targets, predictions)

        return output_metrics


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

        elif name == "baba":
            return self.__baba


    def __max_entropy(self, data, **kwargs):
        # Expectation and variance of form (batch_size, num_classes)
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
        std = np.sqrt(variance)
        return np.mean(std, axis=-1)
    

    def __baba(self, data, num_samples=100, **kwargs):
        """
            Normalized mutual information

            Implementation of acquisition function described in:
            BABA: Beta Approximation for Bayesian Active Learning, Jae Oh Woo
        """
        # predictions shape (batch, num_predictions, num_classes)
        exp, var = self.__mp_model.predict(x=data)

        # BALD term
        exp_shape = list(exp.shape)
        output_shape = tuple([num_samples] + exp_shape) # (num samples, num datapoints, num_classes)
        sampled_data = norm(exp, np.sqrt(var)).rvs(size=output_shape)
        class_sample_probs = tf.keras.activations.softmax(tf.convert_to_tensor(sampled_data))
        disagreement = self.__disagreement(class_sample_probs, num_samples)
        exp, var = MP.Gaussian_Softmax(exp, var)
        bald_term = -self.__entropy(exp)+disagreement
        
        # Beta approximation of upper joint entropy
        a = ((np.power(exp, 2)*(1-exp))/(var+.0001))-exp
        b = ((1/exp)-1)*a
        upper_joint_entropy = beta_approximated_upper_joint_entropy(a, b)
        return bald_term/np.abs(upper_joint_entropy)



    # ---------
    # Utilities
    # ---------------

    def __entropy(self, values):
        return np.sum(values*np.log(values+.001), axis=1)

    
    def __disagreement(self, class_sample_probs, num_samples):
        sample_entropy = np.sum(class_sample_probs*np.log(class_sample_probs+.001), axis=-1)
        return np.sum(sample_entropy, axis=0)/num_samples


    def reset(self, pool, dataset):
        self._model = tf.keras.models.clone_model(self._model)
        self._model.compile(**self._compile_params)
        self.__mp_model = self._create_mp_model(self._model)


    # ----------
    # Setter/-Getter
    # ---------------------

    def get_mp_model(self):
        return self.__mp_model
