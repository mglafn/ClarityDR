import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"✅ Found {gpu_count} GPU(s)")
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    
    # Get device properties
    props = torch.cuda.get_device_properties(0)
    print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"   Compute Capability: {props.major}.{props.minor}")
    
    # Test with smaller tensors first
    print("\n Testing small tensor allocation...")
    try:
        # Start very small
        x = torch.tensor([1.0]).cuda()
        print(f"   ✅ Small tensor created: {x}")
        
        # Try a slightly larger tensor
        print("\n Testing medium tensor allocation...")
        y = torch.randn(100, 100).cuda()
        print(f"   ✅ Medium tensor shape: {y.shape}")
        
        # Test computation
        print("\n Testing GPU computation...")
        z = y @ y.T
        print(f"   ✅ Matrix multiplication successful: {z.shape}")
        
        # Try a larger allocation
        print("\n Testing larger tensor allocation...")
        large = torch.randn(1000, 1000).cuda()
        print(f"   ✅ Large tensor shape: {large.shape}")
        
    except RuntimeError as e:
        print(f"❌ GPU operation failed: {e}")
        print("\nTrying with environment variable adjustments...")
        
else:
    print("❌ No GPU detected")