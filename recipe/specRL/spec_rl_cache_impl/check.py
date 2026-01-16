#!/usr/bin/env python3
"""
Quick check script for specRL installation.

Usage:
    conda activate syslibs
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    python check.py
"""

import subprocess
import sys


def run_check(name, code):
    """Run check in subprocess to avoid protobuf conflicts."""
    print(f"{name}...", end=" ", flush=True)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode == 0 or "OK" in result.stdout:
        print("OK")
        return True
    else:
        print("FAILED")
        if result.stderr:
            for line in result.stderr.split('\n')[:3]:
                if line.strip():
                    print(f"  {line}")
        return False


def main():
    print("=" * 40)
    print("specRL Installation Check")
    print("=" * 40)
    print()

    results = []

    # Check 1: cache_updater
    results.append(run_check("1. cache_updater", '''
from specrl.cache_updater import SuffixCacheUpdater
SuffixCacheUpdater()
print("OK")
'''))

    # Check 2: suffix_cache
    results.append(run_check("2. suffix_cache", '''
from specrl.suffix_cache import SuffixCache, SuffixSpecResult, RolloutCacheServer
print("OK")
'''))

    # Check 3: server
    results.append(run_check("3. server (1GB)", '''
from specrl.suffix_cache import RolloutCacheServer
server = RolloutCacheServer("[::]:6399", shared_memory_size_gb=1)
assert server.initialize(), "init failed"
assert server.start(), "start failed"
server.shutdown()
print("OK")
'''))

    print()
    if all(results):
        print("All checks passed!")
        return 0
    else:
        print("Some checks failed.")
        print("\nMake sure you have:")
        print("  conda activate syslibs")
        print("  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH")
        return 1


if __name__ == "__main__":
    sys.exit(main())
