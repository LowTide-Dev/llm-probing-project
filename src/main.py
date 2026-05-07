# test_load.py
from load_models import load_model

# Test MatSciBERT (no token needed)
tok, model, arch = load_model("matscibert", device="cpu")
print(f"MatSciBERT loaded: arch={arch}, layers={model.config.num_hidden_layers}")

# Test Llama (requires HF token + approved access)
tok, model, arch = load_model("llama", device="cpu")
print(f"Llama loaded: arch={arch}, layers={model.config.num_hidden_layers}")