
from __future__ import absolute_import, division, print_function, unicode_literals

from functools import reduce
import logging
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np
import numpy.linalg as la
from scipy.optimize import fmin as scipy_optimizer
from scipy.stats import weibull_min
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.utils import random_sphere
import tensorflow as tf

from functools import reduce
from art.utils import random_sphere
from art.config import ART_NUMPY_DTYPE
import sys
sys.path.append('..')  
from selection_method.other_rank_method.CertPri.utils import *
from scipy.stats import weibull_min
from scipy.optimize import fmin as scipy_optimizer


if TYPE_CHECKING:
    from art.attacks.attack import EvasionAttack
    from art.utils import CLASSIFIER_TYPE, CLASSIFIER_LOSS_GRADIENTS_TYPE, CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE
    
def inverper_c(
    classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
    x: np.ndarray,
    nb_batches: int,
    batch_size: int,
    radius: float,
    norm: float,
    c_init: float = 1.0,
    pool_factor: int = 10,
) -> float:
    """
    Compute CLEVER score for a targeted attack.

    | Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model.
    :param x: One input sample.
    :param nb_batches: Number of repetitions of the estimate.
    :param batch_size: Number of random examples to sample per batch.
    :param radius: Radius of the maximum perturbation.
    :param norm: Current support: 1, 2, np.inf.
    :param c_init: Initialization of Weibull distribution.
    :param pool_factor: The factor to create a pool of random samples with size pool_factor x n_s.
    :return: CLEVER score.
    """
    # Check if the targeted class is different from the predicted class
    y_pred = classifier.predict(np.array([x]))
    pred_class = np.argmax(y_pred, axis=1)[0]

    # Check if pool_factor is smaller than 1
    if pool_factor < 1:  # pragma: no cover
        raise ValueError("The `pool_factor` must be larger than 1.")

    # Some auxiliary vars
    rand_pool_grad_set = []
    grad_norm_set = []
    dim = reduce(lambda x_, y: x_ * y, x.shape, 1)
    shape = [pool_factor * batch_size]
    shape.extend(x.shape)

    # Generate a pool of samples
    rand_pool = np.reshape(
        random_sphere(nb_points=pool_factor * batch_size, nb_dims=dim, radius=radius, norm=norm),
        shape,
    )
    rand_pool += np.repeat(np.array([x]), pool_factor * batch_size, 0)
    rand_pool = rand_pool.astype(ART_NUMPY_DTYPE)
    if hasattr(classifier, "clip_values") and classifier.clip_values is not None:
        np.clip(rand_pool, classifier.clip_values[0], classifier.clip_values[1], out=rand_pool)

    # Change norm since q = p / (p-1)
    if norm == 1:
        norm = np.inf
    elif norm == np.inf:
        norm = 1
    elif norm != 2:  # pragma: no cover
        raise ValueError(f"Norm {norm} not supported")

    # Compute gradients for all samples in rand_pool
    for i in range(batch_size):
        rand_pool_batch = rand_pool[i * pool_factor : (i + 1) * pool_factor]

        # Compute gradients
        #grad_pred_class = classifier.optimizer.get_gradients(rand_pool_batch, classifier, pred_class)
        rand_pool_batch = tf.convert_to_tensor(rand_pool_batch)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(rand_pool_batch)
            predictions = classifier(rand_pool_batch)[:, pred_class]
            #print(predictions)
        # if tf.test.is_gpu_available():
        #     print("GPU is available for computation.")
        # else:
        #     print("GPU is NOT available for computation.")
        grad_pred_class = tape.gradient(predictions, rand_pool_batch)
        #print(grad_pred_class)
        #grad_pred_class = classifier.class_gradient(rand_pool_batch, label=pred_class)

        if np.isnan(grad_pred_class).any() :  # pragma: no cover
            raise Exception("The classifier results NaN gradients.")

        grad = grad_pred_class
        grad = np.reshape(grad, (pool_factor, -1))
        grad = np.linalg.norm(grad, ord=norm, axis=1)
        rand_pool_grad_set.extend(grad)

    rand_pool_grads = np.array(rand_pool_grad_set)

    # Loop over the batches
    for _ in range(nb_batches):
        # Random selection of gradients
        grad_norm = rand_pool_grads[np.random.choice(pool_factor * batch_size, batch_size)]
        grad_norm = np.max(grad_norm)
        grad_norm_set.append(grad_norm)

    # Maximum likelihood estimation for max gradient norms
    [_, loc, _] = weibull_min.fit(-np.array(grad_norm_set), c_init, optimizer=scipy_optimizer)

    # Compute function value
    values = classifier.predict(np.array([x]))
    value = values[:, pred_class] - 0.5

    # Compute scores
#     score = np.min([-value[0] / loc, radius])
    score = -value[0] / loc

    return score

def inverper_Loss(classifier, x, nb_batches, batch_size, radius, norm, c_init=1.0, pool_factor=5, loss_ratio=0.8):
    # Check if the targeted class is different from the predicted class
    y_pred = classifier.predict(np.array([x]))
    #y_pred = classifier.predict(np.reshape(x, [-1,224,224,3]))
    pred_class = np.argmax(y_pred, axis=1)[0]

    # Check if pool_factor is smaller than 1
    if pool_factor < 1:  # pragma: no cover
        raise ValueError("The `pool_factor` must be larger than 1.")

    # Some auxiliary vars
    rand_pool_grad_set = []
    grad_norm_set = []
    dim = reduce(lambda x_, y: x_ * y, x.shape, 1)
    shape = [pool_factor * batch_size]
    shape.extend(x.shape)

    # Generate a pool of samples
    rand_pool = np.reshape(
        random_sphere(nb_points=pool_factor * batch_size, nb_dims=dim, radius=radius, norm=norm),
        shape,
    )
    rand_pool += np.repeat(np.array([x]), pool_factor * batch_size, 0)
    rand_pool = rand_pool.astype(ART_NUMPY_DTYPE)
    if hasattr(classifier, "clip_values") and classifier.clip_values is not None:
        np.clip(rand_pool, classifier.clip_values[0], classifier.clip_values[1], out=rand_pool)

    # Change norm since q = p / (p-1)
    if norm == 1:
        norm = np.inf
    elif norm == np.inf:
        norm = 1
    elif norm != 2:  # pragma: no cover
        raise ValueError(f"Norm {norm} not supported")

    # Compute gradients for all samples in rand_pool
    for i in range(batch_size):
        rand_pool_batch = rand_pool[i * pool_factor : (i + 1) * pool_factor]

#         Compute gradients
#         grad_pred_class = classifier.class_gradient(rand_pool_batch, label=pred_class)
        grad_pred_class=[]
        for x_tmp in rand_pool_batch:
            
            grads = get_loss_gradients(x_tmp, classifier, target_one_hot=
                                       tf.reshape(tf.one_hot(pred_class,len(y_pred[0])),(1,len(y_pred[0]))))
#           grads = get_gradients(x_tmp, base_model, pred_class)
            #grads = PreGradientEstimator(samples = 6, sigma=1, model = classifier, x=x_tmp, 
                                 #bounds=(np.min(x_tmp),np.max(x_tmp)), noise_mu=0, nise_std=1, 
                                 #top_pred_idx = pred_class, clip=True)
            grad_pred_class.append(grads)

        #grad_pred_class = grad_pred_class[1:]
        # if np.isnan(grad_pred_class).any() :  # pragma: no cover
        #     raise Exception("The classifier results NaN gradients.")

        
        grad = grad_pred_class
        grad = np.reshape(grad, (pool_factor, -1))
        grad = np.linalg.norm(grad, ord=norm, axis=1)
        rand_pool_grad_set.extend(grad)

    rand_pool_grads = np.array(rand_pool_grad_set)

    # Loop over the batches
    np.random.seed(42)
    for _ in range(nb_batches):
        # Random selection of gradients
        grad_norm = rand_pool_grads[np.random.choice(pool_factor * batch_size, batch_size)]
        grad_norm = np.max(grad_norm)
        grad_norm_set.append(grad_norm)

    # Maximum likelihood estimation for max gradient norms
    [_, loc, _] = weibull_min.fit(-np.array(grad_norm_set), c_init, optimizer=scipy_optimizer)

    # Compute function value
    values = classifier.predict(np.array([x]))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=np.argmax(values), logits=values[0]) 
    value = loss - loss*loss_ratio

    # Compute scores
#     score = np.min([-value[0] / loc, radius])
    score = -value / loc

    return score

