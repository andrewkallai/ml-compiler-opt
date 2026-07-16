# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CIR/MLIR Inlining Training config.

Matches the MLIR inliner feature map defined in
MLIRInlineModelFeatureMaps.h (ALL_FEATURES macro).
"""

import gin
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from compiler_opt.rl import feature_ops


# The C++ MLIR inliner (MLInlineAdvisor.cpp) prepends "action_" to all feature
# names when constructing the feature map (getMLIRFeatureMap -> action_#NAME).
# TF-Agents also prepends "action_" when creating the model action signature.
# So bare names like "callee_block_count" produce "action_callee_block_count"
# on both sides, matching the TFLite model input tensor names.
MLIR_FEATURE_NAMES = (
    # Callee region features
    'callee_block_count',
    'callee_region_count',
    'callee_operand_count',
    'callee_result_count',
    'callee_arg_count',
    'callee_is_isolated_from_above',
    # Caller region features
    'caller_block_count',
    'caller_region_count',
    'caller_operand_count',
    'caller_result_count',
    'caller_arg_count',
    'caller_is_isolated_from_above',
    # Call-site features
    'call_site_operand_count',
    'call_site_num_ctant_args',
    'callsite_height',
    # Graph-level features
    'graph_node_count',
    'graph_edge_count',
    'graph_callee_region_level',
    'graph_initial_total_ops',
    'graph_current_total_ops_ratio',
)


@gin.configurable()
def get_cir_signature_spec():
  """Returns (time_step_spec, action_spec) for CIR/MLIR inlining."""
  observation_spec = {
      key: tf.TensorSpec(dtype=tf.int64, shape=(), name=key)
      for key in MLIR_FEATURE_NAMES
  }
  reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
  time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)
  action_spec = tensor_spec.BoundedTensorSpec(
      dtype=tf.int64, shape=(), name='inlining_decision', minimum=0, maximum=1)
  return time_step_spec, action_spec


@gin.configurable
def get_cir_observation_processing_layer_creator(
    quantile_file_dir=None,
    with_sqrt=True,
    with_z_score_normalization=True,
    eps=1e-8):
  """Wrapper for observation_processing_layer for CIR features."""
  quantile_map = feature_ops.build_quantile_map(quantile_file_dir)

  def observation_processing_layer(obs_spec):
    # Try bare name first, fall back to action_ prefixed name for backward
    # compatibility with vocab files generated from C++ training logs.
    quantile_key = obs_spec.name
    if quantile_key not in quantile_map:
      quantile_key = 'action_' + quantile_key
    quantile = quantile_map[quantile_key]
    return tf.keras.layers.Lambda(
        feature_ops.get_normalize_fn(quantile, with_sqrt,
                                     with_z_score_normalization, eps))

  return observation_processing_layer


@gin.configurable()
def get_cir_nonnormalized_features():
  return ['inlining_decision', 'inlining_reward']
