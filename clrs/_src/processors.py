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

DEFAULT_HIDDEN_INPUT = {
  "first_step": ['n_features', 'hidden'],
  "nth_step": ['n_features', 'hidden'],
}

class Processor(hk.Module, abc.ABC):
  def __init__(self, name: Optional[str] = None):
    super().__init__(name)
  
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
      Adjacency matrix for input to the processor network (shape [B,N,N])
    
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
      hidden_input: Dict[str, List[str]] = DEFAULT_HIDDEN_INPUT,
  ):
    super().__init__(name=name)
    self.out_size = out_size
    self.nb_heads = nb_heads
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.head_size = out_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.hidden_input = hidden_input

  def step(
    self,
    node_fts: _Array,
    edge_fts: _Array,
    graph_fts: _Array,
    adj_mat: _Array
  ) -> _Array:
    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = node_fts
    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = jnp.tile(bias_mat[..., None],
                        (1, 1, 1, self.nb_heads))     # [B, N, N, H]
    bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

    a_1 = hk.Linear(self.nb_heads)
    a_2 = hk.Linear(self.nb_heads)
    a_e = hk.Linear(self.nb_heads)
    a_g = hk.Linear(self.nb_heads)

    values = m(z)                                      # [B, N, H*F]
    values = jnp.reshape(
        values,
        values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
    values = jnp.transpose(values, (0, 2, 1, 3))              # [B, H, N, F]

    att_1 = jnp.expand_dims(a_1(z), axis=-1)
    att_2 = jnp.expand_dims(a_2(z), axis=-1)
    att_e = a_e(edge_fts)
    att_g = jnp.expand_dims(a_g(graph_fts), axis=-1)

    logits = (
        jnp.transpose(att_1, (0, 2, 1, 3)) +  # + [B, H, N, 1]
        jnp.transpose(att_2, (0, 2, 3, 1)) +  # + [B, H, 1, N]
        jnp.transpose(att_e, (0, 3, 1, 2)) +  # + [B, H, N, N]
        jnp.expand_dims(att_g, axis=-1)       # + [B, H, 1, 1]
    )                                         # = [B, H, N, N]
    coefs = jax.nn.softmax(jax.nn.leaky_relu(logits) + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)  # [B, H, N, F]
    ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
    ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

    if self.residual:
      ret += skip(z)

    if self.activation is not None:
      ret = self.activation(ret)

    return ret
  
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
    hidden_map = {'n_features': n_features, 'hidden': hidden, 'zeros': jnp.zeros_like(hidden)}
    hidden_input = self.hidden_input["first_step"] if first_step else self.hidden_input["nth_step"]
    features = jnp.concatenate([hidden_map[i] for i in hidden_input], axis=-1)
    return self.step(features, e_features, g_features, adj)

  def preprocess_adjmat(self, adj_mat: _Array) -> _Array:
      return jnp.ones_like(adj_mat)

  @staticmethod
  def from_model_params(model_params: Dict[str, Any]) -> "Processor":
      return GAT(
        out_size=model_params["out_size"],
        nb_heads=model_params["nb_heads"],
        activation=model_params["activation"],
        residual=model_params["residual"],
        hidden_input=model_params["hidden_input"],
      )


class GATv2(Processor):
  """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = None,
      residual: bool = True,
      name: str = 'gat_aggr',
      hidden_input: Dict[str, List[str]] = DEFAULT_HIDDEN_INPUT,
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.nb_heads = nb_heads
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.head_size = out_size // nb_heads
    if self.mid_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the message!')
    self.mid_head_size = self.mid_size // nb_heads
    self.activation = activation
    self.residual = residual

  def step(
    self,
    node_fts: _Array,
    edge_fts: _Array,
    graph_fts: _Array,
    adj_mat: _Array
  ) -> _Array:
    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = node_fts
    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = jnp.tile(bias_mat[..., None],
                        (1, 1, 1, self.nb_heads))     # [B, N, N, H]
    bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

    w_1 = hk.Linear(self.mid_size)
    w_2 = hk.Linear(self.mid_size)
    w_e = hk.Linear(self.mid_size)
    w_g = hk.Linear(self.mid_size)

    a_heads = []
    for _ in range(self.nb_heads):
      a_heads.append(hk.Linear(1))

    values = m(z)                                      # [B, N, H*F]
    values = jnp.reshape(
        values,
        values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
    values = jnp.transpose(values, (0, 2, 1, 3))              # [B, H, N, F]

    pre_att_1 = w_1(z)
    pre_att_2 = w_2(z)
    pre_att_e = w_e(edge_fts)
    pre_att_g = w_g(graph_fts)

    pre_att = (
        jnp.expand_dims(pre_att_1, axis=1) +     # + [B, 1, N, H*F]
        jnp.expand_dims(pre_att_2, axis=2) +     # + [B, N, 1, H*F]
        pre_att_e +                              # + [B, N, N, H*F]
        jnp.expand_dims(pre_att_g, axis=(1, 2))  # + [B, 1, 1, H*F]
    )                                            # = [B, N, N, H*F]

    pre_att = jnp.reshape(
        pre_att,
        pre_att.shape[:-1] + (self.nb_heads, self.mid_head_size)
    )  # [B, N, N, H, F]

    pre_att = jnp.transpose(pre_att, (0, 3, 1, 2, 4))  # [B, H, N, N, F]

    # This part is not very efficient, but we agree to keep it this way to
    # enhance readability, assuming `nb_heads` will not be large.
    logit_heads = []
    for head in range(self.nb_heads):
      logit_heads.append(
          jnp.squeeze(
              a_heads[head](jax.nn.leaky_relu(pre_att[:, head])),
              axis=-1)
      )  # [B, N, N]

    logits = jnp.stack(logit_heads, axis=1)  # [B, H, N, N]

    coefs = jax.nn.softmax(logits + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)  # [B, H, N, F]
    ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
    ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

    if self.residual:
      ret += skip(z)

    if self.activation is not None:
      ret = self.activation(ret)

    return ret

  
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
    hidden_map = {'n_features': n_features, 'hidden': hidden, 'zeros': jnp.zeros_like(hidden)}
    hidden_input = self.hidden_input["first_step"] if first_step else self.hidden_input["nth_step"]
    features = jnp.concatenate([hidden_map[i] for i in hidden_input], axis=-1)
    return self.step(features, e_features, g_features, adj)

  def preprocess_adjmat(self, adj_mat: _Array) -> _Array:
      return jnp.ones_like(adj_mat)

  @staticmethod
  def from_model_params(model_params: Dict[str, Any]) -> "Processor":
      return GAT(
        out_size=model_params["out_size"],
        nb_heads=model_params["nb_heads"],
        activation=model_params["activation"],
        residual=model_params["residual"],
        hidden_input=model_params["hidden_input"],
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
      hidden_input: Dict[str, List[str]] = DEFAULT_HIDDEN_INPUT,
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
    self.hidden_input = hidden_input

  def step(
    self,
    hidden: _Array,
    e_features: _Array,
    g_features: _Array,
    adj: _Array
  ) -> _Array:
    b, n, _ = hidden.shape
    assert e_features.shape[:-1] == (b, n, n)
    assert g_features.shape[:-1] == (b,)
    assert adj.shape == (b, n, n)

    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    msg_1 = m_1(hidden)
    msg_2 = m_2(hidden)
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

    h_1 = o1(hidden)
    h_2 = o2(msgs)

    ret = h_1 + h_2

    if self.activation is not None:
      ret = self.activation(ret)

    return ret

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
    hidden_map = {'n_features': n_features, 'hidden': hidden, 'zeros': jnp.zeros_like(hidden)}
    hidden_input = self.hidden_input["first_step"] if first_step else self.hidden_input["nth_step"]
    features = jnp.concatenate([hidden_map[i] for i in hidden_input], axis=-1)
    return self.step(features, e_features, g_features, adj)

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
        hidden_input=model_params["hidden_input"],
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
        hidden_input=model_params["hidden_input"],
        pgn_mask=model_params["pgn_mask"],
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
        hidden_input=model_params["hidden_input"],
      )