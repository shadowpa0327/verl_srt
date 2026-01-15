# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build script for SRT plugin C++ extension.
# Based on ArcticInference build system.
#
# USAGE:
#   # Single command install (recommended)
#   pip install -e recipe/srt/srt_plugin
#
#   # Or build extension only
#   cd recipe/srt/srt_plugin && python setup.py build_ext --inplace

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CMakeExtension(Extension):

    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):

    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Ensure the output directory exists
        extdir.mkdir(parents=True, exist_ok=True)

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        build_args = []

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                item for item in os.environ["CMAKE_ARGS"].split(" ") if item
            ]

        if self.compiler.compiler_type != "msvc":
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass
        else:
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [
                    "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))
                ]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True
        )

        # For editable installs, also copy .so to source directory
        # This ensures the extension is available when running from source
        setup_dir = Path(__file__).parent.resolve()
        suffix_cache_dir = setup_dir / "suffix_cache"
        if suffix_cache_dir.exists():
            # Find the built .so file matching current Python version
            import sysconfig
            ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
            so_name = f"_C{ext_suffix}"
            so_file = extdir / so_name
            if so_file.exists():
                dest = suffix_cache_dir / so_name
                if so_file != dest:
                    shutil.copy2(so_file, dest)
                    print(f"Copied {so_name} to {suffix_cache_dir}")


# Get the directory where setup.py is located
setup_dir = Path(__file__).parent.resolve()

# The C++ extension for suffix cache
# Use simple name "_C" to avoid path issues during editable install
# The .so gets copied to suffix_cache/ in build_extension()
ext_modules = [
    CMakeExtension(
        "_C",
        str(setup_dir / "suffix_cache" / "csrc")
    ),
]

setup(
    name="srt-plugin-ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
)
