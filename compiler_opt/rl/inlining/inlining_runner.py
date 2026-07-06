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
"""Module for collect data of inlining-for-size."""

import os
import tempfile

import gin
import subprocess
import tensorflow as tf

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import log_reader

_DEFAULT_IDENTIFIER = 'default'


@gin.configurable(module='runners')
class InliningRunner(compilation_runner.CompilationRunner):
  """Class for collecting data for inlining-for-size.

  Usage:
  inliner = InliningRunner(
                clang_path, llvm_size_path, launcher_path,
                moving_average_decay_rate)
  serialized_sequence_example, default_reward, moving_average_reward,
  policy_reward = inliner.collect_data(
      ir_path, tf_policy_path, default_reward, moving_average_reward)
  """

  def __init__(self,
               *,
               llvm_size_path: str,
               ir2vec_vocab_path: str | None = None,
               cir_opt_path: str = 'cir-opt',
               cir_translate_path: str = 'cir-translate',
               llc_path: str = 'llc',
               **kwargs):
    super().__init__(**kwargs)
    self._llvm_size_path = llvm_size_path
    self._ir2vec_vocab_path = ir2vec_vocab_path
    self._cir_opt_path = cir_opt_path
    self._cir_translate_path = cir_translate_path
    self._llc_path = llc_path

  def compile_and_get_size(self, command_line: corpus.FullyQualifiedCmdLine,
                           tf_policy_path: str | None,
                           workdir: str) -> tuple[float, str]:
    """Run inlining for the given IR file. Compiles by using a policy
    if tf_policy_path is not None.

    Args:
      command_line: the fully qualified command line.
      tf_policy_path: path to TF policy directory on local disk.
      workdir: working directory.

    Returns:
      A tuple containing:
        native_size: Native size of the final native code.
        log_path: Log path.

    Raises:
      subprocess.CalledProcessError: if process fails.
      cancellable_process.ProcessKilledError: (which it must pass through) on
      cancelled work.
      RuntimeError: if llvm-size produces unexpected output.
    """

    working_dir = tempfile.mkdtemp(dir=workdir)

    log_path = os.path.join(working_dir, 'log')
    output_native_path = os.path.join(working_dir, 'native')

    native_size = 0

    cir_path = os.path.join(working_dir, 'module.cir')
    cir_opt_path = os.path.join(working_dir, 'module_opt.cir')
    ll_path = os.path.join(working_dir, 'module.ll')

    # Check for pre-compiled CIR input from the CIR corpus
    cir_input_arg = [a for a in command_line if a.startswith('--cir-input=')]
    if cir_input_arg:
      # CIR corpus: copy the pre-compiled .cir file, skip Step 1
      cir_src = cir_input_arg[0].split('=', 1)[1]
      remaining_cmd = tuple(
          a for a in command_line if not a.startswith('--cir-input='))
      with open(cir_src, 'r') as f_in:
        with open(cir_path, 'w') as f_out:
          f_out.write(f_in.read())
    else:
      remaining_cmd = command_line
      # Step 1: Compile source to CIR via clang
      filtered_cmd = [a for a in remaining_cmd if not a.startswith('-emit-')]
      cmdline = []
      if self._launcher_path:
        cmdline.append(self._launcher_path)
      cmdline.extend([self._clang_path] + filtered_cmd)
      cmdline.extend(['-fclangir', '-emit-cir', '-o', cir_path])
      self._cancellation_manager.start_cancellable_process(cmdline)

    # Step 2: Run CIR ML inliner via cir-opt (with goto/label pre-processing)
    # FIDL-generated CIR uses cir.goto/cir.label within scoped regions which
    # the inliner cannot handle; run --cir-flatten-cfg --cir-goto-solver first.
    # Also fix up --cir-flatten-cfg's malformed alloca text output (missing
    # 'init' keyword on __cleanup_dest_slot allocas) that cir-translate rejects.
    cmdline = [self._cir_opt_path]
    inline_opts = 'enable-ml-inliner training-log=' + log_path
    if tf_policy_path:
      inline_opts += ' ml-inliner-model-path=' + tf_policy_path
    cmdline += [
        '--pass-pipeline=builtin.module(cir-flatten-cfg,cir-goto-solver,inline{' +
        inline_opts + '})',
    ]
    cmdline += [cir_path, '-o', cir_opt_path]
    self._cancellation_manager.start_cancellable_process(cmdline)
    # Fix malformed allocas produced by --cir-flatten-cfg
    with open(cir_opt_path, 'r') as f:
        data = f.read()
    data = data.replace('__cleanup_dest_slot", cleanup_dest_slot]',
                        '__cleanup_dest_slot", init]')
    with open(cir_opt_path, 'w') as f:
        f.write(data)

    # Step 3: Lower CIR to LLVM IR via cir-translate
    cmdline = [self._cir_translate_path, '--cir-to-llvmir', cir_opt_path,
               '-o', ll_path]
    self._cancellation_manager.start_cancellable_process(cmdline)

    # Step 4: Compile LLVM IR to object with no LLVM optimizations
    cmdline = [self._clang_path, '-O0', '-c', '-x', 'ir', ll_path,
               '-o', output_native_path]
    self._cancellation_manager.start_cancellable_process(cmdline)
    cmdline = [self._llvm_size_path, output_native_path]
    output = self._cancellation_manager.start_cancellable_process(
        cmdline, stdout=subprocess.PIPE, text=True)
    if not output:
      raise RuntimeError(f'Empty llvm-size output: {" ".join(cmdline)}')
    tmp = output.split('\n')
    if len(tmp) != 3:
      raise RuntimeError(f'Wrong llvm-size output {output}')
    tmp = tmp[1].split('\t')
    native_size = int(tmp[0])
    return native_size, log_path

  def compile_fn(
      self, command_line: corpus.FullyQualifiedCmdLine, tf_policy_path: str,
      reward_only: bool,
      workdir: str) -> dict[str, tuple[tf.train.SequenceExample, float]]:
    """Wraps around compile_and_get_size and returns a dict mapping.

    Args:
      command_line: the fully qualified command line.
      tf_policy_path: path to TF policy directory on local disk.
      reward_only: whether only return native size.

    Returns:
      A dict mapping from example identifier to tuple containing:
        sequence_example: A tf.SequenceExample proto describing compilation
        trace, None if reward_only == True.
        native_size: Native size of the final native code.
    """

    native_size, log_path = self.compile_and_get_size(command_line,
                                                      tf_policy_path, workdir)

    if native_size == 0:
      return {}

    if reward_only:
      return {_DEFAULT_IDENTIFIER: (None, native_size)}

    result = log_reader.read_log_as_sequence_examples(log_path)
    if len(result) != 1:
      return {}
    sequence_example = next(iter(result.values()))

    if not sequence_example.HasField('feature_lists'):
      return {}

    return {_DEFAULT_IDENTIFIER: (sequence_example, native_size)}
