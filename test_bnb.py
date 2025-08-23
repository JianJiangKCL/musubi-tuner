#!/usr/bin/env python
import torch
import bitsandbytes as bnb

print("Testing bitsandbytes...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test a simple 8-bit operation
if torch.cuda.is_available():
    # Create a simple linear layer
    layer = bnb.nn.Linear8bitLt(10, 10)
    x = torch.randn(1, 10).cuda()
    
    # Test forward pass
    output = layer(x)
    print(f"8-bit linear layer test passed! Output shape: {output.shape}")
    
    # Test Adam optimizer
    model = torch.nn.Linear(10, 10).cuda()
    optimizer = bnb.optim.Adam8bit(model.parameters())
    print("8-bit Adam optimizer created successfully!")
    
    print("\nBitsandbytes is working correctly!")
else:
    print("CUDA not available, cannot test GPU operations")