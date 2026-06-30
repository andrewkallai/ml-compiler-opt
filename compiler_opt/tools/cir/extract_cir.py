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
"""Extract CIR for training.

Extract CIR for training from a compile_commands.json file produced by cmake.
Compiles source files to CIR using clang with -fclangir -emit-cir and stores
.cir files alongside .cmd files for use by the ml-compiler-opt pipeline.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import multiprocessing
import functools
import shlex

# Flags that take a value — skip both the flag and its value.
_SKIP_WITH_VALUE = frozenset({
    '-o', '-MF', '-MQ', '-MT', '-dependency-file',
    '-fdebug-compilation-dir',
})


def _is_tool_arg(arg: str) -> bool:
    """Check if an argument is a compiler tool path."""
    base = os.path.basename(arg)
    return base in ('clang', 'clang++', 'cc', 'c++')


def _rewrite_driver_command_to_cir(cmd_parts: list[str],
                                   source_file: str,
                                   output_cir_path: str) -> list[str] | None:
    """Rewrite a clang driver command to emit CIR instead of an object file.

    Takes the driver-level command parts from compile_commands.json and:
    1. Removes the compiler tool name (full path)
    2. Removes -emit-*, -c, -o, build-system flags, and debug flags
    3. Removes -fembed-bitcode* and related -Xclang flags
    4. Adds -fclangir -emit-cir -o <output_cir_path>
    5. Uses source_file (from 'file' field) as the positional input
    """
    # Strip the compiler binary (full path like /path/to/clang++)
    if cmd_parts and _is_tool_arg(cmd_parts[0]):
        parts = cmd_parts[1:]
    elif cmd_parts and cmd_parts[0].endswith(('clang', 'clang++', 'cc', 'c++')):
        parts = cmd_parts[1:]
    else:
        parts = list(cmd_parts)

    result = []
    skip_next = False

    for i, arg in enumerate(parts):
        if skip_next:
            skip_next = False
            continue

        # Skip flags and their following value
        if arg in _SKIP_WITH_VALUE:
            skip_next = True
            continue

        # Skip -emit-* flags
        if arg.startswith('-emit-'):
            continue

        # Skip -c (compile to object)
        if arg == '-c':
            continue

        # Skip dependency-file generation flags
        if arg in ('-MD', '-MMD'):
            continue
        if arg.startswith('-sys-header-deps'):
            continue

        # Skip -g* debug flags
        if arg in ('-g', '-g0', '-g1', '-g2', '-g3',
                   '-gdwarf', '-gdwarf-2', '-gdwarf-3',
                   '-gdwarf-4', '-gdwarf-5'):
            continue

        # Skip -gz* compression flags
        if arg.startswith('-gz'):
            continue

        # Skip -fembed-bitcode*
        if arg.startswith('-fembed-bitcode'):
            continue

        # Skip -save-temps and related
        if arg in ('--save-temps', '-save-temps'):
            continue
        if arg.startswith('--save-temps='):
            continue

        # Skip -fcrash-diagnostics-dir
        if arg.startswith('-fcrash-diagnostics-dir'):
            continue

        # Skip -grecord-gcc-switches and -grecord-command-line
        if arg.startswith('-grecord-'):
            continue

        # Skip -Xclang flags related to embedding, debug info, crash diag
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

        # Skip positional args that look like object files (.o)
        if arg.endswith('.o') or '.o:' in arg:
            continue

        # Skip positional args that look like source files — we use source_file
        if arg.endswith(('.c', '.cc', '.cpp', '.cxx')):
            continue

        result.append(arg)

    # Use the source file from the 'file' field
    result.append(source_file)
    result.extend(['-fclangir', '-emit-cir', '-o', output_cir_path])
    return result


def _extract_cc1_flags(cmd_parts: list[str]) -> list[str]:
    """Extract -cc1 style flags from a driver command."""
    if cmd_parts and _is_tool_arg(cmd_parts[0]):
        parts = cmd_parts[1:]
    elif cmd_parts and cmd_parts[0].endswith(('clang', 'clang++', 'cc', 'c++')):
        parts = cmd_parts[1:]
    else:
        parts = list(cmd_parts)

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


class CIRExtractor:
    """CIR extraction from a compile_commands.json entry."""

    def __init__(self, source_rel_path: str, output_base_dir: str,
                 clang_path: str, build_dir: str, cmd_parts: list[str]):
        self._source_rel_path = source_rel_path
        self._output_base_dir = output_base_dir
        self._clang_path = clang_path
        self._build_dir = build_dir
        self._cmd_parts = cmd_parts

    def relative_output_path(self) -> str:
        return self._source_rel_path

    def dest_dir(self) -> str:
        return os.path.join(
            self._output_base_dir,
            os.path.dirname(self._source_rel_path),
        )

    def module_name(self) -> str:
        """Return the module name with .cir extension."""
        base = os.path.basename(self._source_rel_path)
        for ext in ('.cc', '.cpp', '.c', '.cxx'):
            if base.endswith(ext):
                return base[:-len(ext)] + '.cir'
        return base + '.cir'

    def cir_file(self) -> str:
        return os.path.join(self.dest_dir(), self.module_name())

    def cmd_file(self) -> str:
        return os.path.join(self.dest_dir(), self.module_name() + '.cmd')

    def extract(self) -> str | None:
        """Compile source to CIR and write .cir + .cmd files."""
        os.makedirs(self.dest_dir(), exist_ok=True)

        source_path = os.path.join(self._build_dir, self._source_rel_path)
        if not os.path.exists(source_path):
            logging.warning('Source file does not exist: %s', source_path)
            return None

        cir_path = self.cir_file()
        cir_cmd = _rewrite_driver_command_to_cir(
            self._cmd_parts, self._source_rel_path, cir_path)
        if cir_cmd is None:
            logging.warning('No source file found in command for %s',
                            self._source_rel_path)
            return None

        full_cmd = [self._clang_path] + cir_cmd

        try:
            logging.debug('Running: %s ... -fclangir -emit-cir -o ...',
                          self._clang_path)
            subprocess.check_output(
                full_cmd,
                cwd=self._build_dir,
                stderr=subprocess.STDOUT,
                encoding='utf-8',
            )
        except subprocess.CalledProcessError as e:
            logging.warning('CIR compilation failed for %s (exit %d)',
                            self._source_rel_path, e.returncode)
            logging.debug('Output: %s', e.output[:500] if e.output else '')
            if os.path.exists(cir_path):
                os.remove(cir_path)
            return None

        if not os.path.exists(cir_path):
            logging.warning('CIR file not produced for %s',
                            self._source_rel_path)
            return None

        # Write .cmd file with -cc1 flags
        cc1_flags = _extract_cc1_flags(self._cmd_parts)
        with open(self.cmd_file(), 'w', encoding='utf-8') as f:
            f.write('\0'.join(cc1_flags))

        return self.relative_output_path()


def load_from_compile_commands(
    json_array: list[dict[str, str]],
    output_dir: str,
    clang_path: str,
) -> list[CIRExtractor]:
    """Create CIR extractors from compile_commands.json entries."""
    extractors = []
    for cmd in json_array:
        build_dir = cmd['directory']

        if 'arguments' in cmd:
            cmd_parts = cmd['arguments']
        elif 'command' in cmd:
            cmd_parts = shlex.split(cmd['command'])
        else:
            continue

        source_file = cmd.get('file', '')
        if not source_file:
            logging.warning('No file field in compile_commands entry')
            continue

        extractors.append(CIRExtractor(
            source_rel_path=source_file,
            output_base_dir=output_dir,
            clang_path=clang_path,
            build_dir=build_dir,
            cmd_parts=cmd_parts,
        ))

    return extractors


def write_corpus_manifest(
    relative_output_paths: list[str | None],
    output_dir: str,
):
    """Write corpus_description.json for the CIR corpus."""
    modules = [p for p in relative_output_paths if p is not None]
    corpus_description = {
        'has_thinlto': False,
        'modules': modules,
    }
    with open(
        os.path.join(output_dir, 'corpus_description.json'),
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(corpus_description, f, indent=2)


def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Extract CIR for ML training from a Fuchsia build.',
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to compile_commands.json',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for .cir and .cmd files',
    )
    parser.add_argument(
        '--clang_path',
        type=str,
        default='clang',
        help='Path to clang with CIR support',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        nargs='?',
        help='Number of parallel workers',
    )
    parser.add_argument(
        '--cmd_filter',
        type=str,
        default=None,
        nargs='?',
        help='Regex to filter compilation commands by flags',
    )
    parser.add_argument(
        '--verbosity',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity',
    )
    args = parser.parse_args()
    main(args)


def main(args):
    logging.basicConfig(level=args.verbosity, force=True)

    with open(args.input, encoding='utf-8') as f:
        compile_commands = json.load(f)

    # Apply cmd_filter if specified
    if args.cmd_filter:
        filtered = []
        for cmd in compile_commands:
            cmd_str = cmd.get('command', ' '.join(cmd.get('arguments', [])))
            if re.search(args.cmd_filter, cmd_str):
                filtered.append(cmd)
        compile_commands = filtered
        logging.info('Filtered to %d entries matching "%s"',
                     len(compile_commands), args.cmd_filter)

    extractors = load_from_compile_commands(
        compile_commands, args.output_dir, args.clang_path,
    )
    logging.info('Loaded %d extractors from compile_commands.json',
                 len(extractors))

    with multiprocessing.Pool(args.num_workers) as pool:
        relative_output_paths = pool.map(
            functools.partial(CIRExtractor.extract), extractors)
        pool.close()
        pool.join()

    write_corpus_manifest(relative_output_paths, args.output_dir)

    success_count = sum(1 for p in relative_output_paths if p is not None)
    logging.info('Successfully extracted %d of %d CIR files',
                 success_count, len(extractors))


if __name__ == '__main__':
    parse_args_and_run()
