# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Unified build script for SRT plugin C++ extensions.
#
# This package includes three C++ extensions:
# 1. suffix_cache._C (nanobind) - In-process snapshot mode suffix decoding
# 2. shm_cache.suffix_cache._C (pybind11) - Shared memory mode suffix cache
# 3. shm_cache.cache_updater._C (pybind11) - Shared memory mode gRPC cache updater
#
# USAGE:
#   # Single command install (recommended)
#   uv pip install -e recipe/srt/srt_plugin
#
#   # Or build extension only
#   cd recipe/srt/srt_plugin && python setup.py build_ext --inplace
#
# DEPENDENCIES:
#   For shm_cache extensions, you need system dependencies:
#   - protobuf, grpc, xxhash, boost (see install.sh)

import os
import re
import shutil
import subprocess
import sys
import platform
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


def check_shm_dependencies():
    """Check if required system dependencies for shm_cache are available."""
    missing_deps = []

    if not shutil.which("protoc"):
        missing_deps.append("protobuf-compiler")
    if not shutil.which("grpc_cpp_plugin"):
        missing_deps.append("protobuf-compiler-grpc")
    if not shutil.which("pkg-config"):
        missing_deps.append("pkg-config")

    if missing_deps:
        system = platform.system()
        if system == "Linux":
            install_cmd = "sudo apt install -y libprotobuf-dev protobuf-compiler libgrpc-dev libgrpc++-dev protobuf-compiler-grpc libxxhash-dev libboost-all-dev cmake pkg-config"
        elif system == "Darwin":
            install_cmd = "brew install protobuf grpc xxhash boost cmake pkg-config"
        else:
            install_cmd = "Please install the required dependencies manually"

        print("\n" + "="*70)
        print("WARNING: Missing optional dependencies for shm_cache extensions!")
        print("="*70)
        print("\nThe following system packages are required but not found:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print(f"\nTo install dependencies on {system}:")
        print(f"  {install_cmd}")
        print("\nshm_cache extensions will be SKIPPED. suffix_cache will still be built.")
        print("="*70 + "\n")
        return False
    return True


_protobuf_generated = False


def generate_protobuf_files(proto_dir: Path) -> None:
    """Generate C++ files from .proto file."""
    global _protobuf_generated
    if _protobuf_generated:
        return

    proto_file = proto_dir / "rollout-cache.proto"
    if not proto_file.exists():
        print(f"Warning: {proto_file} not found, skipping protobuf generation")
        return

    pb_cc = proto_dir / "rollout-cache.pb.cc"
    grpc_pb_cc = proto_dir / "rollout-cache.grpc.pb.cc"

    if pb_cc.exists() and grpc_pb_cc.exists():
        proto_mtime = proto_file.stat().st_mtime
        if pb_cc.stat().st_mtime > proto_mtime and grpc_pb_cc.stat().st_mtime > proto_mtime:
            print(f"Protobuf files in {proto_dir} are up to date")
            _protobuf_generated = True
            return

    print(f"Generating protobuf files in {proto_dir}...")
    sys.stdout.flush()

    grpc_plugin = shutil.which("grpc_cpp_plugin")
    if not grpc_plugin:
        for path in ["/usr/bin/grpc_cpp_plugin", "/usr/local/bin/grpc_cpp_plugin"]:
            if os.path.exists(path):
                grpc_plugin = path
                break

    if not grpc_plugin:
        raise RuntimeError("grpc_cpp_plugin not found")

    subprocess.run(
        ["protoc", f"--cpp_out={proto_dir}", f"--proto_path={proto_dir}", str(proto_file)],
        check=True
    )
    subprocess.run(
        ["protoc", f"--grpc_out={proto_dir}", f"--proto_path={proto_dir}",
         f"--plugin=protoc-gen-grpc={grpc_plugin}", str(proto_file)],
        check=True
    )
    print(f"Successfully generated protobuf files")
    _protobuf_generated = True


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "", extension_type: str = "nanobind",
                 copy_to: str = None) -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        self.extension_type = extension_type
        self.copy_to = copy_to  # Relative path to copy the .so to after build


class CMakeBuild(build_ext):
    def run(self):
        print(f"\n{'='*70}")
        print(f"CMakeBuild.run() starting...")
        print(f"Number of extensions: {len(self.extensions)}")
        for ext in self.extensions:
            print(f"  Extension: {ext.name} (type: {getattr(ext, 'extension_type', 'unknown')})")
        print(f"{'='*70}\n")
        sys.stdout.flush()
        super().run()

    def build_extension(self, ext: CMakeExtension) -> None:
        ext_type = getattr(ext, 'extension_type', 'nanobind')

        # Generate protobuf files for pybind11 extensions
        if ext_type == "pybind11":
            shm_cache_dir = Path(ext.sourcedir).parent
            proto_dir = shm_cache_dir / "proto"
            generate_protobuf_files(proto_dir)

        # Use unique output directory per extension to avoid clobbering
        # (CMake always creates _C.so, so we need separate dirs)
        ext_fullpath_rel = self.get_ext_fullpath(ext.name)
        ext_fullpath = Path.cwd() / ext_fullpath_rel
        base_extdir = ext_fullpath.parent.resolve()
        # Create unique subdirectory for each extension
        extdir = base_extdir / f"_build_{ext.name}"
        extdir.mkdir(parents=True, exist_ok=True)

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        if ext_type == "pybind11":
            try:
                import pybind11
                cmake_args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
            except ImportError:
                pass

        build_args = []

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja
                    cmake_args += ["-GNinja", f"-DCMAKE_MAKE_PROGRAM:FILEPATH={Path(ninja.BIN_DIR) / 'ninja'}"]
                except ImportError:
                    pass
        else:
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]
            if not single_config:
                cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}"]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Building extension: {ext.name} (type: {ext_type})")
        print(f"Source directory: {ext.sourcedir}")
        print(f"Build directory: {build_temp}")
        print(f"Output directory: {extdir}")
        print(f"{'='*70}\n")
        sys.stdout.flush()

        subprocess.run(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True)
        subprocess.run(["cmake", "--build", ".", *build_args], cwd=build_temp, check=True)

        # Move _C.so to base_extdir with correct name for setuptools
        import sysconfig
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
        cmake_output = extdir / f"_C{ext_suffix}"
        final_output = base_extdir / f"{ext.name}{ext_suffix}"

        if cmake_output.exists():
            shutil.move(str(cmake_output), str(final_output))
            print(f"Moved _C{ext_suffix} to {final_output}")

        print(f"\n✓ Successfully built {ext.name}")
        print(f"{'='*70}\n")
        sys.stdout.flush()

        # Copy to source directory for editable installs
        self._copy_to_source(ext, base_extdir, final_output)

    def _copy_to_source(self, ext: CMakeExtension, extdir: Path, built_so: Path) -> None:
        """Copy built extension to source directory as _C.so for imports."""
        copy_to = getattr(ext, 'copy_to', None)
        if not copy_to:
            return

        if not built_so.exists():
            return

        import sysconfig
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

        setup_dir = Path(__file__).parent.resolve()
        target_dir = setup_dir / copy_to
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy as _C.so since that's what the Python imports expect
        dest = target_dir / f"_C{ext_suffix}"
        if built_so != dest:
            shutil.copy2(built_so, dest)
            print(f"Copied to {dest}")


def get_version():
    return os.environ.get("SRT_PLUGIN_VERSION", "0.3.0")


# Get the root directory
setup_dir = Path(__file__).parent.resolve()

# Build extension modules list
ext_modules = []

# 1. suffix_cache._C (nanobind) - always build
ext_modules.append(
    CMakeExtension(
        "_C",
        str(setup_dir / "suffix_cache" / "csrc"),
        extension_type="nanobind",
        copy_to="suffix_cache"
    )
)

# 2-3. shm_cache extensions (pybind11) - only if dependencies available
if check_shm_dependencies():
    ext_modules.extend([
        CMakeExtension(
            "shm_suffix_cache",
            str(setup_dir / "shm_cache" / "suffix_cache"),
            extension_type="pybind11",
            copy_to="shm_cache/suffix_cache"
        ),
        CMakeExtension(
            "shm_cache_updater",
            str(setup_dir / "shm_cache" / "cache_updater"),
            extension_type="pybind11",
            copy_to="shm_cache/cache_updater"
        ),
    ])
else:
    print("\nNOTE: Building without shm_cache extensions.")
    print("To enable shm_cache, install dependencies and rebuild.\n")


setup(
    name="srt-plugin-ext",
    version=get_version(),
    description="SRT suffix decoding plugin for vLLM with C++ acceleration",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
)
