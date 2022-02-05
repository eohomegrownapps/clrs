# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""JAX implementation of baseline processor networks."""

import abc
from os import stat
from typing import Any, Callable, Dict, List, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp


_Array = chex.Array
_Fn = Callable[..., Any]

class Processor(hk.Module, abc.ABC):
  @abc.abstractmethod
  def __call__(
      self,
      hidden: _Array,
      n_features: _Array,
      e_features: _Array,
      g_features: _Array,
      adj: _Array,
      first_step: bool
  ) -> _Array:
    """Processor inference step.

    Args:
      hidden: Node hidden state (zero if first step).
      n_features: Node features.
      e_features: Edge features.
      g_features: Graph features.
      adj: Graph adjacency matrix.
      first_step: Whether this is the first inference step.

    Returns:
      Output of processor inference step.
    """
    pass
  
  @abc.abstractmethod
  def preprocess_adjmat(
      self,
      adj_mat: _Array,
  ) -> _Array:
    """Preprocess adjacency matrix given in problem specification for use in model

    Args:
      adj_mat: Problem adjacency matrix (shape [B,N,N])

    Returns:
      Adjaacency matrix for input to the processor network (shape [B,N,N])
    
    Shape:
      B: batch size
      N: number of nodes
    """
    pass
  
  @staticmethod
  def from_model_params(model_params: Dict[str, Any]) -> "Processor":
    """Return an instance of the model instantiated with model parameters."""
    pass

class GAT(Processor):
  """Graph Attention Network (Velickovic et al., ICLR 2018)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      activation: Optional[_Fn] = None,
      residual: bool = True,
      name: str = 'gat_aggr',
  ):
    super().__init__(name=name)
    self.out_size = out_size
    self.nb_heads = nb_heads
    self.activation = activation
    self.residual = residual

  def __call__(
      self,
      hidden: _Array,
      n_features: _Array,
      e_features: _Array,
      g_features: _Array,
      adj: _Array,
      first_step: bool,
  ) -> _Array:
    """GAT inference step.

    Args:
      features: Node features.
      e_features: Edge features.
      g_features: Graph features.
      adj: Graph adjacency matrix.

    Returns:
      Output of GAT inference step.
    """
    features = jnp.concatenate([n_features, hidden], axis=-1)
    b, n, _ = features.shape
    assert e_features.shape[:-1] == (b, n, n)
    assert g_features.shape[:-1] == (b,)
    assert adj.shape == (b, n, n)

    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj - 1.0) * 1e9

    a_1 = hk.Linear(1)
    a_2 = hk.Linear(1)
    a_e = hk.Linear(1)
    a_g = hk.Linear(1)

    values = m(features)

    att_1 = a_1(features)
    att_2 = a_2(features)
    att_e = a_e(e_features)
    att_g = a_g(g_features)

    logits = (
        att_1 + jnp.transpose(att_2, (0, 2, 1)) + jnp.squeeze(att_e, axis=-1) +
        jnp.expand_dims(att_g, axis=-1))
    coefs = jax.nn.softmax(jax.nn.leaky_relu(logits) + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)

    if self.residual:
      ret += skip(features)

    if self.activation is not None:
      ret = self.activation(ret)

    return ret

  def preprocess_adjmat(self, adj_mat: _Array) -> _Array:
      return jnp.ones_like(adj_mat)

  @staticmethod
  def from_model_params(model_params: Dict[str, Any]) -> "Processor":
      return GAT(
        out_size=model_params["out_size"],
        nb_heads=model_params["nb_heads"],
        activation=model_params["activation"],
        residual=model_params["residual"],
      )
    
class MPNN(Processor):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = None,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      name: str = 'mpnn_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    self.activation = activation
    self.reduction = reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes

  def __call__(
      self,
      hidden: _Array,
      n_features: _Array,
      e_features: _Array,
      g_features: _Array,
      adj: _Array,
      first_step: bool,
  ) -> _Array:
    """MPNN inference step.

    Args:
      features: Node features.
      e_features: Edge features.
      g_features: Graph features.
      adj: Graph adjacency matrix.

    Returns:
      Output of MPNN inference step.
    """
    features = jnp.concatenate([n_features, hidden], axis=-1)
    b, n, _ = features.shape
    assert e_features.shape[:-1] == (b, n, n)
    assert g_features.shape[:-1] == (b,)
    assert adj.shape == (b, n, n)

    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    msg_1 = m_1(features)
    msg_2 = m_2(features)
    msg_e = m_e(e_features)
    msg_g = m_g(g_features)

    msgs = (
        jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
        msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))
    if self._msgs_mlp_sizes is not None:
      msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))

    if self.mid_act is not None:
      msgs = self.mid_act(msgs)

    if self.reduction == jnp.mean:
      msgs = jnp.sum(msgs * jnp.expand_dims(adj, -1), axis=-1)
      msgs = msgs / jnp.sum(adj, axis=-1, keepdims=True)
    else:
      msgs = self.reduction(msgs * jnp.expand_dims(adj, -1), axis=1)

    h_1 = o1(features)
    h_2 = o2(msgs)

    ret = h_1 + h_2

    if self.activation is not None:
      ret = self.activation(ret)

    return ret

  def preprocess_adjmat(self, adj_mat: _Array) -> _Array:
      return jnp.ones_like(adj_mat)
  
  @staticmethod
  def from_model_params(model_params: Dict[str, Any]) -> "Processor":
      return MPNN(
        out_size=model_params["out_size"],
        mid_act=model_params["mid_act"],
        activation=model_params["activation"],
        reduction=model_params["reduction"],
        msgs_mlp_sizes=model_params["msgs_mlp_sizes"],
      )

class PGN(MPNN):
  """Pointer Graph Network (Velickovic et al., NeurIPS 2020)."""
  def __init__(self, *args, **kwargs):
      self.pgn_mask = kwargs["pgn_mask"]
      del kwargs["pgn_mask"]
      super().__init__(*args, **kwargs)
  
  def preprocess_adjmat(self, adj_mat: _Array) -> _Array:
      return (adj_mat > 0.0) * 1.0
  
  @staticmethod
  def from_model_params(model_params: Dict[str, Any]) -> "Processor":
      return PGN(
        out_size=model_params["out_size"],
        mid_act=model_params["mid_act"],
        activation=model_params["activation"],
        reduction=model_params["reduction"],
        msgs_mlp_sizes=model_params["msgs_mlp_sizes"],
        pgn_mask=model_params["pgn_mask"]
      )

class DeepSets(MPNN):
  """Deep Sets (Zaheer et al., NeurIPS 2017)."""
  def preprocess_adjmat(self, adj_mat: _Array) -> _Array:
      return jnp.repeat(
          jnp.expand_dims(jnp.eye(adj_mat.shape[1]), 0), adj_mat.shape[0], axis=0)
  
  @staticmethod
  def from_model_params(model_params: Dict[str, Any]) -> "Processor":
      return DeepSets(
        out_size=model_params["out_size"],
        mid_act=model_params["mid_act"],
        activation=model_params["activation"],
        reduction=model_params["reduction"],
        msgs_mlp_sizes=model_params["msgs_mlp_sizes"],
      )