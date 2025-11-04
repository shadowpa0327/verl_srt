import threading
import time
import torch

MATRIX_SIZE = 4096      # adjust for your VRAM
SYNC_EVERY = 0          # set >0 to synchronize every N iterations (0 = never)
SLEEP_SEC = 0.0         # small pause per iteration to ease thermals (e.g., 0.005)

def worker(device_idx: int):
    torch.cuda.set_device(device_idx)
    device = torch.device(f"cuda:{device_idx}")
    print(f"[GPU {device_idx}] starting")

    # --- Preallocate on this GPU ---
    a = torch.randn((MATRIX_SIZE, MATRIX_SIZE), device=device)
    b = torch.randn((MATRIX_SIZE, MATRIX_SIZE), device=device)
    c = torch.empty((MATRIX_SIZE, MATRIX_SIZE), device=device)

    # Use a dedicated stream (optional but clean)
    stream = torch.cuda.Stream(device=device)
    iters = 0

    # Warm-up (helps JIT kernels/heuristics)
    with torch.cuda.stream(stream):
        torch.matmul(a, b, out=c)
    stream.synchronize()

    while True:
        with torch.cuda.stream(stream):
            torch.matmul(a, b, out=c)   # no host-device transfers; fully on-GPU

        iters += 1

        # Optional periodic sync for backpressure/visibility
        if SYNC_EVERY and (iters % SYNC_EVERY == 0):
            stream.synchronize()

        if SLEEP_SEC:
            time.sleep(SLEEP_SEC)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices available")

    n = torch.cuda.device_count()
    print(f"Found {n} GPU(s)")

    threads = []
    for i in range(n):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    # Keep main thread alive; Ctrl+C to stop
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main()
