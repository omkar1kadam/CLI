import subprocess

def check_cuda():
    print("🔍 Checking CUDA via nvidia-smi...\n")
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("❌ NVIDIA drivers or CUDA not found. Please install CUDA Toolkit.")

if __name__ == "__main__":
    check_cuda()
