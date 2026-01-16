#!/usr/bin/env python3
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
Test script to verify specRL installation.

Run this script after installation to check if all modules are working correctly:
    python test_install.py
"""

import sys
import subprocess
import textwrap


def run_test_in_subprocess(test_code, test_name):
    """Run test code in a separate Python process to ensure isolation."""
    print(f"Testing {test_name} module (in isolated process)...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Print the output from subprocess
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
        
        # Check if test passed
        if result.returncode == 0:
            return True
        else:
            if result.stderr:
                print(f"  ✗ Error output:")
                for line in result.stderr.strip().split('\n'):
                    print(f"    {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Test timed out")
        return False
    except Exception as e:
        print(f"  ✗ Error running subprocess: {e}")
        return False


def test_cache_updater():
    """Test cache_updater module import and basic functionality."""
    test_code = textwrap.dedent("""
        import sys
        try:
            from specrl.cache_updater import SuffixCacheUpdater
            print("✓ SuffixCacheUpdater imported successfully")
            
            # Test instantiation (without server connection)
            updater = SuffixCacheUpdater()
            print("✓ SuffixCacheUpdater() created successfully")
            
            # Test with server addresses
            updater_with_addr = SuffixCacheUpdater(["localhost:50051"])
            print("✓ SuffixCacheUpdater(['localhost:50051']) created successfully")
            
            sys.exit(0)
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    """)
    
    return run_test_in_subprocess(test_code, "specrl.cache_updater")


def test_suffix_cache():
    """Test suffix_cache module import and basic functionality."""
    test_code = textwrap.dedent("""
        import sys
        try:
            from specrl.suffix_cache import SuffixCache, SuffixSpecResult, RolloutCacheServer
            print("✓ SuffixCache imported successfully")
            print("✓ SuffixSpecResult imported successfully")
            print("✓ RolloutCacheServer imported successfully")
            
            # Test SuffixCache instantiation
            cache = SuffixCache()
            print("✓ SuffixCache() created successfully")
            
            # Test SuffixSpecResult instantiation
            result = SuffixSpecResult()
            print("✓ SuffixSpecResult() created successfully")
            
            # Check SuffixSpecResult attributes
            assert hasattr(result, 'token_ids'), "Missing token_ids attribute"
            assert hasattr(result, 'parents'), "Missing parents attribute"
            assert hasattr(result, 'probs'), "Missing probs attribute"
            assert hasattr(result, 'score'), "Missing score attribute"
            assert hasattr(result, 'match_len'), "Missing match_len attribute"
            print("✓ SuffixSpecResult attributes verified")
            
            sys.exit(0)
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    """)
    
    return run_test_in_subprocess(test_code, "specrl.suffix_cache")


def main():
    print("=" * 50)
    print("specRL Installation Test")
    print("=" * 50)
    
    results = []
    
    # Test each module
    results.append(("specrl.cache_updater", test_cache_updater()))
    results.append(("specrl.suffix_cache", test_suffix_cache()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    all_passed = True
    for module, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {module}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("All tests passed! specRL is installed correctly.")
        return 0
    else:
        print("Some tests failed. Please check the installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
