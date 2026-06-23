#!/bin/bash

set -x
set -e
set -u

PREFIX=/storage/users/andrewka

# Activate Python venv for ML dependencies
source $PREFIX/opt_dolog/tensorflow_setup_env/bin/activate

# Source CIR toolchain (provides cir-opt, cir-translate, clang, llc, llvm-size)
source /home/users/andrewka/my_repos/mlgo_mlir/env_vars.sh

export FUCHSIA_DIR=$PREFIX/sw/fuchsia
export IDK_DIR=$PREFIX/sw/fuchsia-idk
export SYSROOT_DIR=${FUCHSIA_DIR}/prebuilt/third_party/sysroot

INSTALL_DIR=$PREFIX/LLVM_installs/llvm-mlgo-install
CIR_INSTALL_DIR=/home/users/andrewka/my_repos/forks/llvm-project/mlir_install

platform=linux-x64

export PATH=${FUCHSIA_DIR}/prebuilt/third_party/cmake/${platform}/bin:${PATH}
export PATH=${FUCHSIA_DIR}/prebuilt/third_party/ninja/${platform}/bin:${PATH}
CLANG_TOOLCHAIN_PREFIX=${FUCHSIA_DIR}/prebuilt/third_party/clang/linux-x64/bin/

set +u
export PKG_CONFIG_PATH="$PREFIX/sw/ZLib/zlib-1.3.1_install/lib/pkgconfig:$PREFIX/sw/Zstd/zstd-1.5.7_install/lib/pkgconfig:$PREFIX/sw/LibXML/libxml2-2.9.14_install/lib/pkgconfig:$PKG_CONFIG_PATH"
set -u

set +u
source ${FUCHSIA_DIR}/scripts/fx-env.sh && fx-update-path
set -u

cd ${FUCHSIA_DIR}
set +e
set -e

pip install mlgo-utils

export CORPUS=$PREFIX/corpus
#HERE*********************
cd /storage/users/andrewka/ig-ml-compiler-opt

# ============================================================
# DEFAULT TRACE GENERATION (uses InliningRunner with CIR pipeline)
# ============================================================
export DEFAULT_TRACE=$PREFIX/default_trace_cir
export DEFAULT_VOCAB=compiler_opt/rl/inlining/vocab

# rm -rf $DEFAULT_TRACE &&
#   PYTHONPATH=$PYTHONPATH:. python3 \
#     compiler_opt/tools/generate_default_trace.py \
#     --data_path=$CORPUS \
#     --output_path=$DEFAULT_TRACE \
#     --gin_files=compiler_opt/rl/inlining/gin_configs/common.gin \
#     --gin_bindings=config_registry.get_configuration.implementation=@configs.InliningConfig \
#     --gin_bindings=clang_path="'$CIR_INSTALL_DIR/bin/clang'" \
#     --gin_bindings=llvm_size_path="'$CIR_INSTALL_DIR/bin/llvm-size'" \
#     --gin_bindings=cir_opt_path="'$CIR_INSTALL_DIR/bin/cir-opt'" \
#     --gin_bindings=cir_translate_path="'$CIR_INSTALL_DIR/bin/cir-translate'" \
#     --gin_bindings=llc_path="'$CIR_INSTALL_DIR/bin/llc'" \
#     --sampling_rate=0.2

# ============================================================
# BEHAVIORAL CLONING (warmstart) — does NOT compile, only reads traces
# ============================================================
export WARMSTART_OUTPUT_DIR=$PREFIX/warmstart_cir

# rm -rf $WARMSTART_OUTPUT_DIR && \
#   PYTHONPATH=$PYTHONPATH:. python3 \
#   compiler_opt/rl/train_bc.py \
#   --root_dir=$WARMSTART_OUTPUT_DIR \
#   --data_path=$DEFAULT_TRACE \
#   --gin_files=compiler_opt/rl/inlining/gin_configs/behavioral_cloning_nn_agent.gin

# ============================================================
# PPO TRAINING — uses interactive clang (unchanged, LLVM inliner)
# ============================================================
export OUTPUT_DIR=$PREFIX/model_cir

# rm -rf $OUTPUT_DIR && \
PYTHONPATH=$PYTHONPATH:. python3 \
  compiler_opt/rl/train_locally.py \
  --num_workers=64 \
  --root_dir=$OUTPUT_DIR \
  --data_path=$CORPUS \
  --gin_bindings=clang_path="'$CIR_INSTALL_DIR/bin/clang'" \
  --gin_bindings=llvm_size_path="'$CIR_INSTALL_DIR/bin/llvm-size'" \
  --gin_files=compiler_opt/rl/inlining/gin_configs/ppo_nn_agent.gin

set -u
