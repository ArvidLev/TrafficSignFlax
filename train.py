#CNN model with training functions

import jax
from flax import linen as nn
from flax.training import train_state
import jax.numpy as jnp
import numpy as np
import optax
from typing import Any
import functools
from tqdm import tqdm


class CNN(nn.Module):
# VGG-11 network
# The network is based on a VGG19 model but smaller due to images being too small.
# The network is used for classifying 58 classes which is why it has 58 features in the last dense layer.
    @nn.compact
    def __call__(self, x, training):
        x = self._stack(x, 64, training)
        x = self._stack(x, 64, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    
        x = self._stack(x, 128, training)
        x = self._stack(x, 128, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._stack(x, 256, training)
        x = self._stack(x, 256, training)
        x = self._stack(x, 256, training)
        x = self._stack(x, 256, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))    

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=4096)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5, deterministic=not training)(x)

        x = nn.Dense(features=4096)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5, deterministic=not training)(x)
    
        x = nn.Dense(features=58)(x)
        x = nn.log_softmax(x)
        return x
  
    @staticmethod
    def _stack(x, features, training, dropout=None):
        x = nn.Conv(features=features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        return x


@jax.jit
def train_step(state, images, labels):
    # Defines how we train each batch of data.
    # Defines the loss_fn and then uses jax.value_and_grad function with respect to state.params
    # We use cross entopy as our loss function.
    def loss_fn(images, labels, rng, batch_stats, params):
        logits, batch_stats = CNN().apply({'params': params, 'batch_stats': batch_stats}, 
                             images, 
                             training=True,
                             rngs={'dropout': rng},
                             mutable=['batch_stats'])
        one_hot = jax.nn.one_hot(labels, 58)
        loss = -jnp.mean(jnp.sum(one_hot * logits, axis=-1))
        return loss, (logits, batch_stats)

    rng, subrng = jax.random.split(state.rng)
    batch_loss_fn = functools.partial(loss_fn, images, labels, rng, state.batch_stats)
    (batch_loss, (logits, batch_stats)), grads = jax.value_and_grad(batch_loss_fn, has_aux=True)(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.update_batch_stats(state, batch_stats['batch_stats'])
    state = state.update_rng(state, rng)

    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    stats ={'loss': batch_loss, 'accuracy': accuracy}
    return state, stats

@jax.jit
def eval_step(state, images, labels):
    # Defines how we evaluate the model on a batch of data.
    logits = CNN().apply({'params': state.params, 'batch_stats': state.batch_stats},
                          images,
                         training=False,
                          )
    return jnp.mean(jnp.argmax(logits, -1) == labels)



def train_epoch(state, dataloader):
    # Defines the training of the model on a dataset for one epoch.
    batch_metrics = []
    for i, (images, labels) in enumerate(tqdm(dataloader)):
        state, stats = train_step(state, images, labels)
        batch_metrics.append(stats)
    
    batch_metrics_np = jax.device_get(batch_metrics)
    batch_metrics_np = {k: np.mean([metric[k] for metric in batch_metrics_np])
        for k in batch_metrics_np[0]}
    

    return state, batch_metrics_np

def evaluate_model(state, dataloader):
    # Defines the evaluation of the model on a dataset for one epoch
    batch_metrics = []
    for i, (images, labels) in enumerate(dataloader):
        batch_metrics.append(eval_step(state, images, labels))
    
    batch_metrics = jax.device_get(batch_metrics)
    batch_metrics = [k.item() for k in batch_metrics]
    accuracy = np.mean(batch_metrics)
    return accuracy


class VGGState(train_state.TrainState):
    # Creates a custom training state for our model.
    # This keeps track of our parameters and batch statistics which we use for training the model.
    rng: Any
    batch_stats: Any

    @classmethod
    def create(cls, apply_fn, params, tx, rng, batch_stats):
        opt_state = tx.init(params)
        state = cls(0, apply_fn, params, tx, opt_state, rng, batch_stats)
        return state
    
    @classmethod
    def update_rng(cls, state, rng):
        return VGGState.create(state.apply_fn, state.params, state.tx, rng,
                               state.batch_stats)
    
    @classmethod
    def update_batch_stats(cls, state, batch_stats):
        return VGGState.create(state.apply_fn, state.params, state.tx,
                               state.rng, batch_stats)

def create_train_state(key, learning_rate, momentum):
    # Initializes a trainiing state for our model.
    # Stochastic gradient descent is used as the optimizer.
    cnn = CNN()
    params = cnn.init({'params': key, 'dropout': key}, jnp.ones([1, 28, 26, 3]), training=True)
    tx = optax.sgd(learning_rate, momentum)
    state = VGGState.create(apply_fn=cnn.apply, params=params['params'], tx=tx, rng=key, batch_stats=params['batch_stats'])
    return state


