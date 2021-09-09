

# Moment Propagation extension for the tf-al package

An extension for the tf-al (Active learning with tensorflow). Provides an additional wrapper
to use moment propagation models in bayesian active learning loops.


For more information on Moment Propagation check out [Moment Propagation by Kai Brach](https://github.com/kaibrach/Moment-Propagation).


## Dependencies

```toml
python = "^3.8"
tensorflow = "2.2.0"
tf-al = "^0.0.1"
```


## How to use


```python
from tf import ActiveLearningLoop
from tf_al_mp.wrapper import MomentPropagation


# ... Define dataset, set seeds, etc. checkout tf-al documentation for more information


# Define the tensorflow model to use (Needs to include at least a single dropout layer)
# The last layer should be Softmax() 
base_model = Sequential([
    Conv2D(32, 3, activation=tf.nn.relu, padding="same", input_shape=input_shape),
    Conv2D(64, 3, activation=tf.nn.relu, padding="same"),
    MaxPooling2D(),
    Dropout(.25),
    Flatten(),
    Dense(128, activation=tf.nn.relu),
    Dropout(.5),
    Dense(output),
    Softmax()          
])


# Use the model wrapper
mp_config = Config(
    fit={"epochs": 100, "batch_size": batch_size},
    eval={"batch_size": 900}
)
mp_model = MomentPropagation(base_model, mp_config, verbose=verbose)
mp_model.compile(optimizer=optimizer, loss=loss,  metrics=metrics)

# Create and run active learning loop
acquisition_function = "random"
active_learning_loop = ActiveLearningLoop(
    mp_model,
    dataset,
    acquisition_function,
    step_size=100,
    max_rounds=10
)

active_learning_loop.run()
```