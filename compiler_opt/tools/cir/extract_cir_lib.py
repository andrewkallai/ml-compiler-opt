# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Library functions for CIR extraction."""

# TODO(boomanaiden154): Remove this import once we have upgrade to python 3.10
# which supports the relevant type annotations by default.
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import multiprocessing
import functools
import json
import logging

_UNSPECIFIED_OVERRIDE = ["<UNSPECIFIED>"]

# Flags that take a value — skip both the flag and its value.
_SKIP_WITH_VALUE = frozenset({
    '-o', '-MF', '-MQ', '-MT', '-dependency-file',
    '-fdebug-compilation-dir',
})


def should_include_module(cmdline: str, match_regexp: str | None) -> bool:
    """Determine if the module should be included."""
    if match_regexp is None:
        return True
    lines = cmdline.split("\0")
    return any(len(re.findall(match_regexp, l)) for l in lines)


def get_thinlto_index(cmdline: str, basedir: str) -> str | None:
    opts = cmdline.split("\0")
    for option in opts:
        if option.startswith("-fthinlto-index"):
            return os.path.join(basedir, option.split("=")[1])
    return None


def _rewrite_driver_command_to_cir(cmd_parts: list[str],
                                   source_file: str,
                                   output_cir_path: str) -> list[str] | None:
    """Rewrite a clang driver command to emit CIR instead of an object file."""
    parts = cmd_parts[1:] if cmd_parts else []

    result = []
    skip_next = False

    for i, arg in enumerate(parts):
        if skip_next:
            skip_next = False
            continue
        if arg in _SKIP_WITH_VALUE:
            skip_next = True
            continue
        if arg.startswith('-emit-'):
            continue
        if arg == '-c':
            continue
        if arg in ('-MD', '-MMD'):
            continue
        if arg.startswith('-sys-header-deps'):
            continue
        if arg in ('-g', '-g0', '-g1', '-g2', '-g3',
                   '-gdwarf', '-gdwarf-2', '-gdwarf-3',
                   '-gdwarf-4', '-gdwarf-5'):
            continue
        if arg.startswith('-gz'):
            continue
        if arg.startswith('-fembed-bitcode'):
            continue
        if arg in ('--save-temps', '-save-temps'):
            continue
        if arg.startswith('--save-temps='):
            continue
        if arg.startswith('-fcrash-diagnostics-dir'):
            continue
        if arg.startswith('-ftrivial-auto-var-init'):
            continue
        if arg.startswith('-grecord-'):
            continue
        if arg == '-Xclang':
            if i + 1 < len(parts):
                nxt = parts[i + 1]
                if ('fembed' in nxt or 'debug-info' in nxt or
                        nxt.startswith('-debug-info') or
                        'crash-diagnostics' in nxt):
                    skip_next = True
                    continue
            result.append(arg)
            continue
        if arg.endswith('.o') or '.o:' in arg:
            continue
        if arg.endswith(('.c', '.cc', '.cpp', '.cxx')):
            continue
        result.append(arg)

    result.append(source_file)
    result.extend(['-fclangir', '-emit-cir', '-o', output_cir_path])
    return result


def _extract_cc1_flags(cmd_parts: list[str]) -> list[str]:
    """Extract -cc1 style flags from a driver command."""
    if cmd_parts:
        base = os.path.basename(cmd_parts[0])
        if base in ('clang', 'clang++', 'cc', 'c++') or cmd_parts[0].endswith(
                ('clang', 'clang++', 'cc', 'c++')):
            parts = cmd_parts[1:]
        else:
            parts = list(cmd_parts)
    else:
        parts = []

    flags = ['-cc1']
    for arg in parts:
        if arg.startswith(('-D', '-I', '-U', '-std=', '-m', '-f', '-O',
                           '-W', '-w', '-target', '-triple')):
            flags.append(arg)
        elif arg == '-target':
            flags.append(arg)
        elif arg.startswith('--sysroot'):
            flags.append(arg)
        elif arg.startswith('-isysroot'):
            flags.append(arg)
        elif arg.startswith('-resource-dir'):
            flags.append(arg)
        elif arg.startswith('-isystem'):
            flags.append(arg)
    return flags


class TrainingIRExtractor:
    """CIR extraction from an object file with embedded bitcode."""

    def __init__(self, obj_relative_path, output_base_dir, obj_base_dir=None,
                 source_file=None, cmd_parts=None, build_dir=None):
        """Set up a TrainingIRExtractor.

        Args:
          obj_relative_path: relative path to the input object file.
          output_base_dir: the directory under which the output will be produced.
          obj_base_dir: the base directory for all the input object files.
          source_file: the source file path (for CIR compilation).
          cmd_parts: the original compiler command parts (for CIR compilation).
          build_dir: the build directory for source-relative paths.
        """
        self._obj_relative_path = obj_relative_path
        self._output_base_dir = output_base_dir
        self._obj_base_dir = obj_base_dir if obj_base_dir is not None else ""
        self._source_file = source_file
        self._cmd_parts = cmd_parts if cmd_parts is not None else []
        self._build_dir = build_dir if build_dir is not None else ""

    def obj_base_dir(self):
        return self._obj_base_dir

    def output_base_dir(self):
        return self._output_base_dir

    def relative_output_path(self):
        return self._obj_relative_path

    def input_obj(self):
        return os.path.join(self.obj_base_dir(), self._obj_relative_path)

    def lld_src_bc(self):
        return os.path.join(
            self._obj_base_dir, self._obj_relative_path + ".3.import.bc"
        )

    def lld_src_thinlto(self):
        return os.path.join(self._obj_base_dir, self._obj_relative_path + ".thinlto.bc")

    def dest_dir(self):
        return os.path.join(
            self.output_base_dir(), os.path.dirname(self._obj_relative_path)
        )

    def module_name(self):
        name = os.path.basename(self._obj_relative_path)
        if name.endswith('.o'):
            return name[:-2]
        return name

    def cmd_file(self):
        return os.path.join(self.dest_dir(), self.module_name() + ".cmd")

    def bc_file(self):
        return os.path.join(self.dest_dir(), self.module_name() + ".bc")

    def cir_file(self):
        return os.path.join(self.dest_dir(), self.module_name() + ".cir")

    def thinlto_index_file(self):
        return os.path.join(self.dest_dir(), self.module_name() + ".thinlto.bc")

    def _get_extraction_cmd_command(
        self, llvm_objcopy_path: str, cmd_section_name: str
    ):
        return [
            llvm_objcopy_path,
            "--dump-section=" + cmd_section_name + "=" + self.cmd_file(),
            self.input_obj(),
            "/dev/null",
        ]

    def _get_extraction_bc_command(
        self, llvm_objcopy_path: str, bitcode_section_name: str
    ):
        return [
            llvm_objcopy_path,
            "--dump-section=" + bitcode_section_name + "=" + self.bc_file(),
            self.input_obj(),
            "/dev/null",
        ]

    def _extract_clang_artifacts(
        self,
        llvm_objcopy_path: str,
        cmd_filter: str | None,
        is_thinlto: bool,
        cmd_section_name: str,
        bitcode_section_name: str,
        clang_path: str | None = None,
    ) -> str | None:
        """Extract embedded bitcode or compile to CIR if clang_path is given."""
        if clang_path:
            return self._extract_cir(clang_path)
        return self._extract_bitcode(
            llvm_objcopy_path, cmd_filter, is_thinlto,
            cmd_section_name, bitcode_section_name,
        )

    def _extract_bitcode(
        self,
        llvm_objcopy_path: str,
        cmd_filter: str | None,
        is_thinlto: bool,
        cmd_section_name: str,
        bitcode_section_name: str,
    ) -> str | None:
        """Run llvm-objcopy to extract the .bc and command line."""
        if not os.path.exists(self.input_obj()):
            logging.info("%s does not exist.", self.input_obj())
            return None
        os.makedirs(self.dest_dir(), exist_ok=True)
        try:
            subprocess.check_output(
                self._get_extraction_cmd_command(llvm_objcopy_path, cmd_section_name),
                stderr=subprocess.STDOUT,
                encoding="utf-8",
            )
            if cmd_filter is not None or is_thinlto:
                with open(self.cmd_file(), encoding="utf-8") as f:
                    cmdline = f.read()
                if cmd_filter is not None and not should_include_module(
                    cmdline, cmd_filter
                ):
                    os.remove(self.cmd_file())
                    return None
                thinlto_index = (
                    get_thinlto_index(cmdline, self._obj_base_dir)
                    if is_thinlto
                    else None
                )
            subprocess.check_output(
                self._get_extraction_bc_command(llvm_objcopy_path, bitcode_section_name),
                stderr=subprocess.STDOUT,
                encoding="utf-8",
            )
            if is_thinlto:
                if thinlto_index is not None:
                    shutil.copy2(thinlto_index, self.thinlto_index_file())
                else:
                    logging.warning(
                        "failed to extract thinlto index for %s", self.input_obj()
                    )
            return self.relative_output_path()
        except subprocess.CalledProcessError as e:
            logging.info(
                "Error extracting artifacts from %s: %s", self.input_obj(), e
            )
            return None

    def _extract_cir(self, clang_path: str) -> str | None:
        """Compile source to CIR using clang."""
        # Resolve source file path
        if self._source_file:
            source_path = os.path.join(self._build_dir, self._source_file)
        else:
            # Fallback: guess source from obj path
            obj_path = self.input_obj()
            obj_stem = obj_path[:-2] if obj_path.endswith('.o') else obj_path
            source_path = None
            for ext in ('.cc', '.cpp', '.c', '.cxx'):
                candidate = obj_stem + ext
                if os.path.exists(candidate):
                    source_path = candidate
                    break

        if not source_path or not os.path.exists(source_path):
            logging.info("No source file found for %s", self.input_obj())
            return None

        os.makedirs(self.dest_dir(), exist_ok=True)
        cir_path = self.cir_file()

        if self._cmd_parts:
            cir_cmd = _rewrite_driver_command_to_cir(
                self._cmd_parts, self._source_file or source_path, cir_path)
            full_cmd = [clang_path] + cir_cmd
            run_dir = self._build_dir or '.'
        else:
            full_cmd = [
                clang_path, '-fclangir', '-emit-cir', '-o', cir_path, source_path,
            ]
            run_dir = '.'

        try:
            subprocess.check_output(
                full_cmd, cwd=run_dir, stderr=subprocess.STDOUT, encoding='utf-8',
            )
        except subprocess.CalledProcessError as e:
            logging.info("CIR compilation failed for %s: %s", source_path, e)
            if os.path.exists(cir_path):
                os.remove(cir_path)
            return None

        if not os.path.exists(cir_path):
            return None

        # Write .cmd file with -cc1 flags
        if self._cmd_parts:
            cc1_flags = _extract_cc1_flags(self._cmd_parts)
            with open(self.cmd_file(), 'w', encoding='utf-8') as f:
                f.write('\0'.join(cc1_flags))

        return self.relative_output_path()

    def extract(
        self,
        llvm_objcopy_path: str,
        cmd_filter: str | None,
        thinlto_build: str,
        cmd_section_name: str,
        bitcode_section_name: str,
        clang_path: str | None = None,
    ) -> str | None:
        """Extract bitcode or compile CIR from the object file."""
        return self._extract_clang_artifacts(
            llvm_objcopy_path,
            cmd_filter,
            thinlto_build is not None,
            cmd_section_name,
            bitcode_section_name,
            clang_path,
        )


def load_from_compile_commands(
    compile_commands: list[dict], output_dir: str
) -> list[TrainingIRExtractor]:
    """Create TrainingIRExtractors from compile_commands.json entries."""
    objs = []
    for entry in compile_commands:
        if 'arguments' in entry:
            parts = entry['arguments']
        elif 'command' in entry:
            import shlex
            parts = shlex.split(entry['command'])
        else:
            continue
        directory = entry.get('directory', '')
        source_file = entry.get('file', '')
        if not source_file:
            continue

        # Find the -o flag to get the object file path
        obj_rel = None
        for i, part in enumerate(parts):
            if part == '-o' and i + 1 < len(parts):
                obj_rel = (
                    os.path.relpath(
                        os.path.join(directory, parts[i + 1]), directory
                    ) if os.path.isabs(parts[i + 1]) else parts[i + 1]
                )
                break
        if obj_rel is None:
            continue

        objs.append(TrainingIRExtractor(
            obj_relative_path=obj_rel,
            output_base_dir=output_dir,
            obj_base_dir=directory,
            source_file=source_file,
            cmd_parts=parts,
            build_dir=directory,
        ))
    return objs


def load_from_lld_params(
    lines: list[str], obj_base_dir: str, output_dir: str
) -> list[TrainingIRExtractor]:
    """Creates an object file array by looking at a linker's @params file."""
    return [
        TrainingIRExtractor(
            obj_relative_path=line,
            output_base_dir=output_dir,
            obj_base_dir=obj_base_dir if obj_base_dir is not None else "",
        )
        for line in lines
        if line.endswith(".o")
    ]


def load_from_directory(
    obj_base_dir: str, output_dir: str
) -> list[TrainingIRExtractor]:
    paths = [str(p) for p in pathlib.Path(obj_base_dir).glob("**/*.o")]

    def make_spec(obj_file: str):
        return TrainingIRExtractor(
            obj_relative_path=os.path.relpath(obj_file, start=obj_base_dir),
            output_base_dir=output_dir,
            obj_base_dir=obj_base_dir,
        )

    return [make_spec(path) for path in paths]


def load_for_lld_thinlto(
    obj_base_dir: str, output_dir: str
) -> list[TrainingIRExtractor]:
    paths = [str(p) for p in pathlib.Path(obj_base_dir).glob("**/*.3.import.bc")]

    def make_spec(obj_file: str):
        return TrainingIRExtractor(
            obj_relative_path=os.path.relpath(obj_file, start=obj_base_dir)[:-12],
            output_base_dir=output_dir,
            obj_base_dir=obj_base_dir,
        )

    return [make_spec(path) for path in paths]


def load_bazel_aquery(aquery_json, obj_base_dir: str, output_dir: str):
    linker_params = []

    for action_info in aquery_json["actions"]:
        if action_info["mnemonic"] != "CppLink":
            continue
        linker_params = action_info["arguments"]

    return load_from_lld_params(linker_params, obj_base_dir, output_dir)


def run_extraction(
    objs: list[TrainingIRExtractor],
    num_workers: int,
    llvm_objcopy_path: str,
    cmd_filter: str | None,
    thinlto_build: str,
    cmd_section_name: str,
    bitcode_section_name: str,
    clang_path: str | None = None,
):
    """Extracts all specified object files into the corpus directory.

    Args:
      objs: A list of TrainingIRExtractor Objects that represent the object files
        to extract bitcode/commands from.
      num_workers: The number of parallel processes to spawn to run the
        extraction.
      llvm_objcopy_path: The path to the llvm-objcopy to use for dumping sections.
      cmd_filter: A regular expression that is used to select for compilations
        performed with specific flags.
      thinlto_build: Whether or not this is a ThinLTO build, and if so, the type.
      cmd_section_name: The name of the command line section.
      bitcode_section_name: The name of the bitcode section.
      clang_path: Path to clang for CIR compilation. If None, uses llvm-objcopy
        bitcode extraction instead.
    """
    extract_artifacts = functools.partial(
        TrainingIRExtractor.extract,
        llvm_objcopy_path=llvm_objcopy_path,
        cmd_filter=cmd_filter,
        thinlto_build=thinlto_build,
        cmd_section_name=cmd_section_name,
        bitcode_section_name=bitcode_section_name,
        clang_path=clang_path,
    )

    with multiprocessing.Pool(num_workers) as pool:
        relative_output_paths = pool.map(extract_artifacts, objs)
        pool.close()
        pool.join()
    return relative_output_paths


def write_corpus_manifest(
    thinlto_build: str, relative_output_paths: list[str], output_dir: str
):
    """Writes a corpus_manifest.json containing all necessary information about
    the corpus.

    Args:
      thinlto_build: Whether or not the build was done with ThinLTO and if so,
        what kind of ThinLTO.
      relative_output_paths: The relative (to the corpus directory) output paths
        of all the bitcode files that should be placed in the corpus manifest.
      output_dir: The corpus directory where the corpus manifest should be
        placed.
    """
    if thinlto_build == "local":
        corpus_description = {"global_command_override": _UNSPECIFIED_OVERRIDE}
    else:
        corpus_description = {}

    corpus_description.update(
        {
            "has_thinlto": thinlto_build is not None,
            "modules": [path for path in relative_output_paths if path is not None],
        }
    )

    with open(
        os.path.join(output_dir, "corpus_description.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(corpus_description, f, indent=2)
