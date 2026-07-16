# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Extract CIR for training.

Extract CIR for training, either from a compile_commands.json file produced by
cmake, or a linker parameter list file.

Compiles source files to CIR using clang with -fclangir -emit-cir and stores
.cir files alongside .cmd files for use by the ml-compiler-opt pipeline.
"""

import argparse
import json
import logging

import extract_cir_lib


def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description="A tool for making a CIR corpus from build artifacts",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input file or directory - either compile_commands.json, a linker "
        "parameter list, or a path to a directory containing object files.",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        help="Input file type - JSON, LLD params, directory, or bazel aquery.",
        choices=["json", "params", "directory", "bazel_aquery"],
        default="json",
        nargs="?",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of parallel workers. `None` for maximum available.",
        default=None,
        nargs="?",
    )
    parser.add_argument(
        "--llvm_objcopy_path",
        type=str,
        help="Path to llvm-objcopy",
        default="llvm-objcopy",
        nargs="?",
    )
    parser.add_argument(
        "--obj_base_dir",
        type=str,
        help="Base directory for object files. Defaults to current working dir.",
        default="",
        nargs="?",
    )
    parser.add_argument(
        "--cmd_filter",
        type=str,
        help="Include only those modules with a command line matching this regular "
        "expression. Set it to None to not perform any filtering.",
        default=None,
        nargs="?",
    )
    parser.add_argument(
        "--thinlto_build",
        type=str,
        help="Set if the build was performed with either 'distributed' or 'local' "
        "ThinLTO.",
        choices=["distributed", "local"],
        default=None,
        nargs="?",
    )
    parser.add_argument(
        "--clang_path",
        type=str,
        help="Path to clang for CIR compilation",
        default=None,
        nargs="?",
    )
    parser.add_argument(
        "--cmd_section_name",
        type=str,
        help="The section name passed to llvm-objcopy. For ELF object files, the "
        "default .llvmcmd is correct. For Mach-O object files, one should use "
        "something like __LLVM,__cmdline",
        default=".llvmcmd",
        nargs="?",
    )
    parser.add_argument(
        "--bitcode_section_name",
        type=str,
        help="The section name passed to llvm-objcopy. For ELF object files, the "
        "default .llvmbc is correct. For Mach-O object files, one should use "
        "__LLVM,__bitcode",
        default=".llvmbc",
        nargs="?",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()
    main(args)


def main(args):
    logging.basicConfig(level=args.verbosity)

    objs = []
    if args.input is not None and args.thinlto_build == "local":
        raise ValueError("--thinlto_build=local cannot be run with --input")
    if args.input is None:
        if args.thinlto_build != "local":
            raise ValueError("--input or --thinlto_build=local must be provided")
        objs = extract_cir_lib.load_for_lld_thinlto(
            args.obj_base_dir, args.output_dir
        )
    elif args.input_type == "json":
        with open(args.input, encoding="utf-8") as f:
            objs = extract_cir_lib.load_from_compile_commands(
                json.load(f), args.output_dir
            )
    elif args.input_type == "params":
        if not args.obj_base_dir:
            logging.info(
                "-obj_base_dir is unspecified, assuming current directory. "
                "If no objects are found, use this option to specify the root "
                "directory for the object file paths in the input file."
            )
        with open(args.input, encoding="utf-8") as f:
            objs = extract_cir_lib.load_from_lld_params(
                [l.strip() for l in f.readlines()], args.obj_base_dir, args.output_dir
            )
    elif args.input_type == "directory":
        logging.warning(
            "Using the directory input is only recommended if the build system "
            "your project uses does not support any structured output that "
            "ml-compiler-opt understands. If your build system provides a "
            "structured compilation database, use that instead"
        )
        objs = extract_cir_lib.load_from_directory(args.input, args.output_dir)
    elif args.input_type == "bazel_aquery":
        with open(args.input, encoding="utf-8") as aquery_json_handle:
            objs = extract_cir_lib.load_bazel_aquery(
                json.load(aquery_json_handle), args.obj_base_dir, args.output_dir
            )
    else:
        logging.error("Unknown input type: %s", args.input_type)

    relative_output_paths = extract_cir_lib.run_extraction(
        objs,
        args.num_workers,
        args.llvm_objcopy_path,
        args.cmd_filter,
        args.thinlto_build,
        args.cmd_section_name,
        args.bitcode_section_name,
        clang_path=args.clang_path,
    )

    extract_cir_lib.write_corpus_manifest(
        args.thinlto_build, relative_output_paths, args.output_dir
    )

    logging.info(
        "Converted %d files out of %d",
        len(objs) - relative_output_paths.count(None),
        len(objs),
    )


if __name__ == "__main__":
    parse_args_and_run()
