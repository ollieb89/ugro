#!/bin/bash
# Test W&B Integration with UGRO HPO

echo "🧪 Testing W&B Integration for UGRO HPO"
echo "========================================"

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "❌ W&B not installed. Install with: pixi add wandb"
    exit 1
fi

# Set W&B project for testing
export WANDB_PROJECT="ugro-hpo-test"

# Check if W&B is logged in
echo "📋 Checking W&B login status..."
python -c "import wandb; print(f'Logged in as: {wandb.api.viewer().entity()}')" 2>/dev/null || {
    echo "⚠️  Not logged in to W&B. Please run: wandb login"
    echo "   Or set WANDB_API_KEY environment variable"
}

# Run a simple test
echo ""
echo "🚀 Running simple HPO test with W&B..."
python test_wandb_integration.py

echo ""
echo "✅ Test complete!"
echo ""
echo "📊 Check your W&B dashboard at:"
echo "   https://wandb.ai/<your-username>/ugro-hpo-test"
echo ""
echo "🔧 To test with full Ray Tune integration:"
echo "   ugro hpo sweep \\"
echo "     --study-name wandb-ray-test \\"
echo "     --search-space config/llama_lora_hpo.yaml \\"
echo "     --n-trials 5 \\"
echo "     --parallel-jobs 2 \\"
echo "     --wandb-project ugro-hpo-test \\"
echo "     --model unsloth/tinyllama-bnb-4bit \\"
echo "     --max-steps 50"
