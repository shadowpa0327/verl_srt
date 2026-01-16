#!/usr/bin/env python3
"""
Simple test for shared memory operations in /dev/shm.
Tests basic functionality without needing full RolloutCacheServer.
"""

import os
import mmap
import struct
import tempfile

SHM_DIR = "/dev/shm"
TEST_SHM_NAME = "specrl_test_shm"


def test_shm_basic():
    """Test basic shared memory read/write."""
    shm_path = os.path.join(SHM_DIR, TEST_SHM_NAME)

    print("=" * 50)
    print("Test 1: Basic shared memory operations")
    print("=" * 50)

    try:
        # Create shared memory file
        print(f"Creating shared memory at: {shm_path}")
        with open(shm_path, "wb") as f:
            # Write 1MB of data
            size = 1024 * 1024
            f.write(b'\x00' * size)
        print(f"  Created {size} bytes")

        # Memory map it
        print("Memory mapping the file...")
        fd = os.open(shm_path, os.O_RDWR)
        mm = mmap.mmap(fd, size)

        # Write some test data
        test_data = b"Hello from specRL shared memory test!"
        mm[0:len(test_data)] = test_data
        print(f"  Wrote: {test_data.decode()}")

        # Read it back
        read_data = mm[0:len(test_data)]
        print(f"  Read:  {read_data.decode()}")

        assert read_data == test_data, "Data mismatch!"
        print("  Data verification: OK")

        # Write integers
        offset = 100
        values = [42, 1337, 9999, 12345]
        for i, val in enumerate(values):
            struct.pack_into('i', mm, offset + i * 4, val)
        print(f"  Wrote integers at offset {offset}: {values}")

        # Read integers back
        read_values = [struct.unpack_from('i', mm, offset + i * 4)[0] for i in range(len(values))]
        print(f"  Read integers: {read_values}")
        assert read_values == values, "Integer data mismatch!"

        # Cleanup
        mm.close()
        os.close(fd)
        os.unlink(shm_path)
        print("  Cleanup: OK")

        print("\nTest 1: PASSED")
        return True

    except Exception as e:
        print(f"\nTest 1: FAILED - {e}")
        if os.path.exists(shm_path):
            os.unlink(shm_path)
        return False


def test_shm_multiprocess():
    """Test shared memory between processes."""
    import multiprocessing

    shm_path = os.path.join(SHM_DIR, TEST_SHM_NAME + "_mp")
    size = 4096

    print("\n" + "=" * 50)
    print("Test 2: Multi-process shared memory")
    print("=" * 50)

    def writer_process(path, size):
        """Writer process - creates and writes to shared memory."""
        fd = os.open(path, os.O_RDWR)
        mm = mmap.mmap(fd, size)

        # Write a magic number (use unsigned int)
        struct.pack_into('I', mm, 0, 0xDEADBEEF)

        # Write sequence
        for i in range(10):
            struct.pack_into('i', mm, 4 + i * 4, i * 100)

        mm.close()
        os.close(fd)

    def reader_process(path, size, result_queue):
        """Reader process - reads from shared memory."""
        import time
        time.sleep(0.1)  # Wait for writer

        fd = os.open(path, os.O_RDONLY)
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)

        magic = struct.unpack_from('I', mm, 0)[0]
        values = [struct.unpack_from('i', mm, 4 + i * 4)[0] for i in range(10)]

        mm.close()
        os.close(fd)

        result_queue.put((magic, values))

    try:
        # Create shared memory file
        print(f"Creating shared memory at: {shm_path}")
        with open(shm_path, "wb") as f:
            f.write(b'\x00' * size)

        # Create processes
        result_queue = multiprocessing.Queue()
        writer = multiprocessing.Process(target=writer_process, args=(shm_path, size))
        reader = multiprocessing.Process(target=reader_process, args=(shm_path, size, result_queue))

        print("Starting writer and reader processes...")
        writer.start()
        reader.start()

        writer.join()
        reader.join()

        # Check results
        magic, values = result_queue.get(timeout=5)
        print(f"  Reader got magic: {hex(magic)}")
        print(f"  Reader got values: {values}")

        assert magic == 0xDEADBEEF, f"Magic mismatch: {hex(magic)}"
        expected = [i * 100 for i in range(10)]
        assert values == expected, f"Values mismatch: {values} vs {expected}"

        # Cleanup
        os.unlink(shm_path)
        print("  Cleanup: OK")

        print("\nTest 2: PASSED")
        return True

    except Exception as e:
        print(f"\nTest 2: FAILED - {e}")
        if os.path.exists(shm_path):
            os.unlink(shm_path)
        return False


def test_shm_large():
    """Test larger shared memory allocation."""
    shm_path = os.path.join(SHM_DIR, TEST_SHM_NAME + "_large")

    print("\n" + "=" * 50)
    print("Test 3: Large shared memory (100MB)")
    print("=" * 50)

    try:
        size = 100 * 1024 * 1024  # 100MB

        print(f"Allocating {size // (1024*1024)}MB shared memory...")
        with open(shm_path, "wb") as f:
            f.truncate(size)

        print("Memory mapping...")
        fd = os.open(shm_path, os.O_RDWR)
        mm = mmap.mmap(fd, size)

        # Write at various offsets
        offsets = [0, size // 4, size // 2, size - 100]
        for off in offsets:
            mm[off:off+8] = b"TESTDATA"
        print(f"  Wrote at offsets: {offsets}")

        # Verify
        for off in offsets:
            assert mm[off:off+8] == b"TESTDATA", f"Verify failed at offset {off}"
        print("  Verification: OK")

        mm.close()
        os.close(fd)
        os.unlink(shm_path)
        print("  Cleanup: OK")

        print("\nTest 3: PASSED")
        return True

    except Exception as e:
        print(f"\nTest 3: FAILED - {e}")
        if os.path.exists(shm_path):
            os.unlink(shm_path)
        return False


def test_boost_shm_compatibility():
    """Test if we can create shared memory compatible with Boost.Interprocess naming."""
    # Boost.Interprocess uses /dev/shm with the name directly
    shm_name = "SUFFIX_CACHE_TEST"
    shm_path = os.path.join(SHM_DIR, shm_name)

    print("\n" + "=" * 50)
    print("Test 4: Boost.Interprocess compatible naming")
    print("=" * 50)

    try:
        size = 1024 * 1024  # 1MB

        print(f"Creating shared memory with Boost-style name: {shm_name}")
        with open(shm_path, "wb") as f:
            f.truncate(size)

        # Check it exists
        assert os.path.exists(shm_path), "Shared memory file not created"
        print(f"  File exists at: {shm_path}")

        # Check permissions
        stat = os.stat(shm_path)
        print(f"  Size: {stat.st_size} bytes")
        print(f"  Mode: {oct(stat.st_mode)}")

        os.unlink(shm_path)
        print("  Cleanup: OK")

        print("\nTest 4: PASSED")
        return True

    except Exception as e:
        print(f"\nTest 4: FAILED - {e}")
        if os.path.exists(shm_path):
            os.unlink(shm_path)
        return False


def main():
    print("specRL Shared Memory Tests")
    print("=" * 50)
    print(f"Shared memory directory: {SHM_DIR}")
    print(f"Directory exists: {os.path.exists(SHM_DIR)}")
    print(f"Directory writable: {os.access(SHM_DIR, os.W_OK)}")
    print("")

    results = []
    results.append(("Basic SHM", test_shm_basic()))
    results.append(("Multi-process", test_shm_multiprocess()))
    results.append(("Large allocation", test_shm_large()))
    results.append(("Boost naming", test_boost_shm_compatibility()))

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())
