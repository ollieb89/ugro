#!/usr/bin/env python3
"""
Single GPU Test Script - Verify Environment
Tests that PyTorch, CUDA, Unsloth, and models load correctly on a single GPU.
Run this on each machine before launching distributed training.

Usage:
    python train_single_test.py
    
Expected output:
    ✓ CUDA available: True
    ✓ GPU count: 1
    ✓ Model loaded successfully
    ✓ Forward pass successful
    ✓ All tests passed!
"""

import sys

import torch


def test_cuda() -> bool:
    """Test CUDA availability and GPU info."""
    print("\n" + "=" * 60)
    print("CUDA TESTS")
    print("=" * 60)
    
    try:
        cuda_available = torch.cuda.is_available()
        print(f"✓ CUDA available: {cuda_available}")
        
        if not cuda_available:
            print("❌ CUDA not available!")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPU count: {gpu_count}")
        
        current_device = torch.cuda.current_device()
        print(f"✓ Current device: {current_device}")
        
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU name: {device_name}")
        
        device_props = torch.cuda.get_device_properties(0)
        total_memory_gb = device_props.total_memory / 1e9
        print(f"✓ GPU memory: {total_memory_gb:.2f} GB")
        
        cuda_version = torch.version.cuda
        print(f"✓ CUDA version: {cuda_version}")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False


def test_pytorch() -> bool:
    """Test PyTorch version and basic operations."""
    print("\n" + "=" * 60)
    print("PYTORCH TESTS")
    print("=" * 60)
    
    try:
        pytorch_version = torch.__version__
        print(f"✓ PyTorch version: {pytorch_version}")
        
        # Test tensor creation and GPU transfer
        test_tensor = torch.randn(10, 10)
        test_tensor = test_tensor.to("cuda:0")
        print(f"✓ Tensor creation and GPU transfer: OK")
        
        # Test simple computation
        _ = torch.matmul(test_tensor, test_tensor)
        print(f"✓ Matrix multiplication: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False


def test_transformers() -> bool:
    """Test transformers library import."""
    print("\n" + "=" * 60)
    print("TRANSFORMERS TESTS")
    print("=" * 60)
    
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        
        from transformers import AutoTokenizer
        print(f"✓ AutoTokenizer import: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        return False


def test_peft() -> bool:
    """Test PEFT (LoRA) library import."""
    print("\n" + "=" * 60)
    print("PEFT TESTS")
    print("=" * 60)
    
    try:
        import peft
        print(f"✓ PEFT import: OK")
        
        from peft import LoraConfig, get_peft_model
        print(f"✓ LoraConfig and get_peft_model: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ PEFT test failed: {e}")
        return False


def test_unsloth() -> tuple[bool, object | None, object | None]:
    """Test Unsloth import and model loading."""
    print("\n" + "=" * 60)
    print("UNSLOTH TESTS")
    print("=" * 60)
    
    try:
        from unsloth import FastLanguageModel
        print(f"✓ Unsloth import: OK")
        
        # Try loading TinyLlama (smallest model, fastest test)
        print(f"\nLoading TinyLlama model...")
        print(f"  - Model: unsloth/tinyllama-bnb-4bit")
        print(f"  - Max seq length: 2048")
        print(f"  - Data type: float16")
        print(f"  - Quantization: 4-bit")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/tinyllama-bnb-4bit",
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        
        print(f"✓ Model loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params / 1e6:.2f}M")
        
        # Check tokenizer
        sample_text = "Hello, how are you?"
        tokens = tokenizer(sample_text, return_tensors="pt")
        print(f"✓ Tokenizer working: '{sample_text}' -> {tokens['input_ids'].shape[1]} tokens")
        
        return True, model, tokenizer
        
    except Exception as e:
        print(f"❌ Unsloth test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_forward_pass(model: object, tokenizer: object) -> bool:
    """Test forward pass through the model."""
    print("\n" + "=" * 60)
    print("FORWARD PASS TEST")
    print("=" * 60)
    
    try:
        # Prepare input
        text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        
        # Move to GPU
        for key in inputs.keys():
            inputs[key] = inputs[key].to("cuda:0")
        
        print(f"Input shape: {inputs['input_ids'].shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ Forward pass successful")
        print(f"✓ Logits shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distributed() -> bool:
    """Test distributed training imports."""
    print("\n" + "=" * 60)
    print("DISTRIBUTED TRAINING TESTS")
    print("=" * 60)
    
    try:
        import torch.distributed as dist
        print(f"✓ torch.distributed import: OK")
        
        from torch.nn.parallel import DistributedDataParallel as DDP
        print(f"✓ DistributedDataParallel import: OK")
        
        # Check NCCL
        nccl_version = torch.cuda.nccl.version()
        print(f"✓ NCCL version: {nccl_version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Distributed test failed: {e}")
        return False


def test_memory_capacity() -> bool:
    """Test if model fits in available GPU memory."""
    print("\n" + "=" * 60)
    print("MEMORY CAPACITY TEST")
    print("=" * 60)
    
    try:
        # Get GPU memory info
        device_props = torch.cuda.get_device_properties(0)
        total_memory = device_props.total_memory / 1e9
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free = (total_memory - allocated)
        
        print(f"GPU Memory Status:")
        print(f"  - Total: {total_memory:.2f} GB")
        print(f"  - Allocated: {allocated:.2f} GB")
        print(f"  - Reserved: {reserved:.2f} GB")
        print(f"  - Free: {free:.2f} GB")
        
        # Check if Llama-7B would fit
        llama7b_required = 9.0  # GB (4-bit + LoRA + optimizer)
        if free >= llama7b_required:
            print(f"✓ Llama-2-7B should fit: {free:.2f} GB > {llama7b_required:.2f} GB required")
        else:
            print(f"⚠️  Llama-2-7B might be tight: {free:.2f} GB < {llama7b_required:.2f} GB required")
            print(f"   Recommendation: Use gradient accumulation or reduce sequence length")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def main() -> int:
    """Run all tests."""
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  DISTRIBUTED TRAINING ENVIRONMENT TEST".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    
    results = {}
    
    # Run tests
    results['cuda'] = test_cuda()
    results['pytorch'] = test_pytorch()
    results['transformers'] = test_transformers()
    results['peft'] = test_peft()
    unsloth_ok, model, tokenizer = test_unsloth()
    results['unsloth'] = unsloth_ok
    
    if unsloth_ok and model is not None:
        results['forward_pass'] = test_forward_pass(model, tokenizer)
    else:
        results['forward_pass'] = False
    
    results['distributed'] = test_distributed()
    results['memory'] = test_memory_capacity()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name.ljust(20)} {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! Your environment is ready for distributed training.\n")
        print("Next steps:")
        print("  1. Repeat this test on each of your 3 machines")
        print("  2. Once all machines pass, run distributed training launcher")
        print("  3. See quickstart-guide.md for next steps")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix issues before proceeding.")
        print("\nTroubleshooting tips:")
        print("  - CUDA: Check nvidia-smi output")
        print("  - PyTorch: Ensure PyTorch is installed with CUDA support")
        print("  - Unsloth: Try: pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
        print("  - Memory: Check available GPU memory with nvidia-smi")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
