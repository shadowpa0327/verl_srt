# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""
Setup script for specRL package.

This package contains two C++ extension modules built with pybind11:
- specrl.cache_updater: For updating the rollout cache
- specrl.suffix_cache: For suffix tree based cache management
"""

import os
import re
import shutil
import subprocess
import sys
import platform
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


def check_system_dependencies():
    """Check if required system dependencies are available."""
    missing_deps = []
    
    # Check for protoc
    if not shutil.which("protoc"):
        missing_deps.append("protobuf-compiler")
    
    # Check for grpc_cpp_plugin
    if not shutil.which("grpc_cpp_plugin"):
        missing_deps.append("protobuf-compiler-grpc")
    
    # Check for pkg-config
    if not shutil.which("pkg-config"):
        missing_deps.append("pkg-config")
    
    if missing_deps:
        system = platform.system()
        if system == "Linux":
            install_cmd = f"sudo apt install -y {' '.join(['libprotobuf-dev', 'protobuf-compiler', 'libgrpc-dev', 'libgrpc++-dev', 'protobuf-compiler-grpc', 'libxxhash-dev', 'libboost-all-dev', 'cmake', 'pkg-config'])}"
        elif system == "Darwin":  # macOS
            install_cmd = f"brew install protobuf grpc xxhash boost cmake pkg-config"
        else:
            install_cmd = "Please install the required dependencies manually"
        
        print("\n" + "="*70)
        print("ERROR: Missing required system dependencies!")
        print("="*70)
        print("\nThe following system packages are required but not found:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print(f"\nTo install dependencies on {system}:")
        print(f"  {install_cmd}")
        print("\nFor more details, see: https://github.com/He-Jingkai/specRL#installation")
        print("="*70 + "\n")
        sys.exit(1)


class CMakeExtension(Extension):
    """A CMake-based extension module."""

    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


# Flag to track if protobuf files have been generated
_protobuf_generated = False


def generate_protobuf_files(root_dir: Path) -> None:
    """Generate C++ files from shared .proto file using protoc and grpc_cpp_plugin."""
    global _protobuf_generated
    
    # Only generate once
    if _protobuf_generated:
        return
    
    proto_dir = root_dir / "proto"
    proto_file = proto_dir / "rollout-cache.proto"
    
    if not proto_file.exists():
        print(f"Warning: {proto_file} not found, skipping protobuf generation")
        return

    # Check if generated files already exist and are newer than proto file
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

    # Find grpc_cpp_plugin
    grpc_plugin = shutil.which("grpc_cpp_plugin")
    if not grpc_plugin:
        # Try common locations
        for path in ["/usr/bin/grpc_cpp_plugin", "/usr/local/bin/grpc_cpp_plugin"]:
            if os.path.exists(path):
                grpc_plugin = path
                break
    
    if not grpc_plugin:
        raise RuntimeError(
            "grpc_cpp_plugin not found. Please install grpc: "
            "sudo apt install -y protobuf-compiler-grpc"
        )

    print(f"  Running protoc for C++ files...")
    sys.stdout.flush()
    # Generate protobuf C++ files
    subprocess.run(
        ["protoc", f"--cpp_out={proto_dir}", f"--proto_path={proto_dir}", str(proto_file)],
        check=True
    )

    print(f"  Running protoc for gRPC files...")
    sys.stdout.flush()
    # Generate gRPC C++ files
    subprocess.run(
        [
            "protoc",
            f"--grpc_out={proto_dir}",
            f"--proto_path={proto_dir}",
            f"--plugin=protoc-gen-grpc={grpc_plugin}",
            str(proto_file)
        ],
        check=True
    )
    print(f"Successfully generated protobuf files in {proto_dir}")
    sys.stdout.flush()
    _protobuf_generated = True


class CMakeBuild(build_ext):
    """Custom build_ext command that uses CMake to build extensions."""

    def run(self):
        """Override run to add debugging."""
        print(f"\n{'='*70}")
        print(f"CMakeBuild.run() starting...")
        print(f"Number of extensions: {len(self.extensions)}")
        for ext in self.extensions:
            print(f"  Extension: {ext.name}")
        print(f"{'='*70}\n")
        sys.stdout.flush()
        super().run()

    def build_extension(self, ext: CMakeExtension) -> None:
        # Generate protobuf files before building
        # ext.sourcedir is like /path/to/specrl/cache_updater or /path/to/specrl/suffix_cache
        # proto directory is at /path/to/specrl/proto (sibling of cache_updater/suffix_cache)
        root_dir = Path(ext.sourcedir).parent  # Gets us to /path/to/specrl
        print(f"DEBUG: ext.sourcedir = {ext.sourcedir}")
        print(f"DEBUG: root_dir = {root_dir}")
        print(f"DEBUG: proto_dir = {root_dir / 'proto'}")
        sys.stdout.flush()
        generate_protobuf_files(root_dir)

        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Import optional dependencies
        try:
            import pybind11
            pybind11_cmake_dir = pybind11.get_cmake_dir()
        except ImportError:
            pybind11_cmake_dir = None

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        if pybind11_cmake_dir:
            cmake_args.append(f"-Dpybind11_DIR={pybind11_cmake_dir}")

        # Platform-specific optimizations - controlled by ENABLE_AGGRESSIVE_OPTS env var
        # Don't override CMakeLists.txt settings here, let CMake handle it based on the env var
        # The optimization flags are set in the CMakeLists.txt files

        build_args = []

        # Adding CMake arguments set as environment variable
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

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
                cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        print(f"\n{'='*70}")
        print(f"Building extension: {ext.name}")
        print(f"Source directory: {ext.sourcedir}")
        print(f"Build directory: {build_temp}")
        print(f"{'='*70}")
        print(f"\nConfiguring with CMake...")
        sys.stdout.flush()

        # Check which cmake is being used
        cmake_path = shutil.which("cmake")
        print(f"Using cmake: {cmake_path}")
        cmake_version_result = subprocess.run(
            ["cmake", "--version"],
            capture_output=True,
            text=True
        )
        print(f"CMake version: {cmake_version_result.stdout.split()[2] if cmake_version_result.returncode == 0 else 'unknown'}")
        sys.stdout.flush()

        # Run cmake with unbuffered output
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("CMake configuration timed out after 600 seconds")

        # Set a 600-second timeout (10 minutes) for CMake configuration
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)
        
        try:
            # Run cmake without capturing output to avoid deadlock
            # CMake changes behavior when stdout is a pipe
            cmake_cmd = ["cmake", ext.sourcedir] + cmake_args
            
            proc = subprocess.Popen(
                cmake_cmd,
                cwd=build_temp,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            # Read and print output line by line
            for line in proc.stdout:
                print(line, end='')
                sys.stdout.flush()
            
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, "cmake")
        finally:
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)
        
        print(f"\n✓ CMake configure completed")
        print(f"\nCompiling C++ code...\n")
        sys.stdout.flush()
        
        proc = subprocess.Popen(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
            stdout=None,
            stderr=None
        )
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, "cmake --build")
        
        print(f"\n✓ Successfully built {ext.name}")
        print(f"{'='*70}\n")
        sys.stdout.flush()


def get_version():
    """Get version from pyproject.toml or environment variable."""
    version = os.environ.get("SPECRL_VERSION", "0.1.0")
    return version


if __name__ == "__main__":
    # Check system dependencies before building
    check_system_dependencies()
    
    # Get the root directory
    ROOT_DIR = Path(__file__).parent.resolve()

    setup(
        name="specrl",
        version=get_version(),
        description="Speculative Decoding RL - Cache management with suffix tree support",
        long_description=(ROOT_DIR / "README.md").read_text(encoding="utf-8") if (ROOT_DIR / "README.md").exists() else "",
        long_description_content_type="text/markdown",
        author="jingkai.he@bytedance.com",
        license="Apache-2.0",
        python_requires=">=3.8",
        packages=["specrl", "specrl.cache_updater", "specrl.suffix_cache"],
        package_dir={
            "specrl": "specrl",
            "specrl.cache_updater": "specrl/cache_updater",
            "specrl.suffix_cache": "specrl/suffix_cache",
        },
        ext_modules=[
            CMakeExtension("specrl.cache_updater._C", str(ROOT_DIR / "specrl" / "cache_updater")),
            CMakeExtension("specrl.suffix_cache._C", str(ROOT_DIR / "specrl" / "suffix_cache")),
        ],
        cmdclass={"build_ext": CMakeBuild},
        install_requires=[],
        extras_require={
            "dev": [
                "pytest>=7.0.0",
                "pytest-cov",
                "black",
                "isort",
                "mypy",
            ],
        },
        zip_safe=False,
    )
