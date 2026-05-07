from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch

# Two-model lineup:
#   matscibert  — domain-specific BERT encoder (materials science pretraining)
#   llama       — modern general-purpose decoder (Llama-3.2-3B, 2024)
#
# Note: Llama requires a HuggingFace token and access approval at
#       https://huggingface.co/meta-llama/Llama-3.2-3B
#       Set your token via: huggingface-cli login


MODEL_REGISTRY = {
    "matscibert": {
        "hf_id": "m3rg-iitd/matscibert",
        "arch": "encoder",
        "n_layers": 12,
        "hidden_size": 768,
    },
    "llama": {
        "hf_id": "meta-llama/Llama-3.2-3B",
        "arch": "decoder",
        "n_layers": 28,
        "hidden_size": 3072,
    },
}


def load_model(model_key: str, device: str = "cpu"):
    """
    Load tokenizer and model for the given model key.
    Returns (tokenizer, model, arch) where arch is 'encoder' or 'decoder'.
    """
    cfg = MODEL_REGISTRY[model_key]
    hf_id = cfg["hf_id"]
    arch = cfg["arch"]

    print(f"[load_models] Loading {model_key} ({hf_id}) on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    if arch == "encoder":
        model = AutoModel.from_pretrained(hf_id, output_hidden_states=True)

    elif arch == "decoder":
        # output_hidden_states must be set via config, not from_pretrained kwargs,
        # for decoder-only models in recent transformers versions
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        model.config.output_hidden_states = True

        # Llama tokenizer has no pad token by default — set to eos
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return tokenizer, model, arch