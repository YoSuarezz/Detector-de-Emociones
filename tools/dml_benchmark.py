# dml_benchmark.py
import time
import torch
import torch_directml

def run_matmul_bench(device, loops=100, size=4096):
    print(f"Matmul bench: device={device}, loops={loops}, size={size}x{size}")
    a = torch.randn((size, size), device=device)
    b = torch.randn((size, size), device=device)
    # warmup
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda = getattr(torch, "cuda", None)  # keep safe if not present
    t0 = time.time()
    for i in range(loops):
        c = torch.matmul(a, b)
        # force some synchronization to measure
        if hasattr(device, "synchronize"):
            try:
                device.synchronize()
            except Exception:
                pass
    t1 = time.time()
    print("Matmul time total:", t1 - t0, "s, avg:", (t1 - t0) / loops, "s/loop")

def run_conv_bench(device, loops=200):
    # small conv bench (simulate CNN conv)
    x = torch.randn((32, 1, 64, 128), device=device)  # batch, channels, H, W
    conv = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1).to(device)
    # warmup
    for _ in range(5):
        _ = conv(x)
    t0 = time.time()
    for _ in range(loops):
        y = conv(x)
    t1 = time.time()
    print("Conv time total:", t1 - t0, "s, avg:", (t1 - t0) / loops, "s/loop")

def main():
    try:
        dml = torch_directml.device()
        print("Using DML device:", dml)
    except Exception as e:
        print("Cannot create torch_directml device:", e)
        return

    # move default device?
    # run a medium matmul (size tuneable - reduce if OOM)
    try:
        run_matmul_bench(dml, loops=20, size=2048)  # reduce loops/size if OOM
    except Exception as e:
        print("Matmul bench failed:", e)

    try:
        run_conv_bench(dml, loops=100)
    except Exception as e:
        print("Conv bench failed:", e)

if __name__ == "__main__":
    main()
