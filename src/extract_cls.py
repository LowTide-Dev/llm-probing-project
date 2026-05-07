import numpy as np
import torch


def get_all_layer_embeddings(text, tokenizer, model, device="cpu", arch="encoder"):
    """
    Returns layer-wise embeddings for a single text string.

    For BERT-style encoders (arch="encoder"):
        - 'cls'  → [CLS] token (index 0) at each layer
        - 'mean' → mean of all token hidden states at each layer

    For decoder-only models (arch="decoder"):
        - 'cls'  → last token hidden state at each layer
                   (decoders have no [CLS]; last token is the natural summary)
        - 'mean' → mean of all token hidden states at each layer

    Returns:
        dict with keys 'cls' and 'mean', each a list of np.arrays per layer.
        Embedding layer (index 0) is skipped; only transformer layers are returned.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (n_layers + 1) tensors, shape [1, seq_len, hidden_dim]
    # Index 0 is the raw embedding layer — skip it, probe layers 1..n
    hidden_states = outputs.hidden_states[1:]

    if arch == "encoder":
        # [CLS] token is always position 0 for BERT-style models
        cls_per_layer  = [hs[0, 0, :].cpu().float().numpy() for hs in hidden_states]
        mean_per_layer = [hs[0, :, :].mean(dim=0).cpu().float().numpy() for hs in hidden_states]

    elif arch == "decoder":
        # Find the last real (non-padding) token position
        attention_mask = inputs["attention_mask"][0]          # shape [seq_len]
        last_token_idx = attention_mask.sum().item() - 1      # 0-indexed

        cls_per_layer  = [hs[0, last_token_idx, :].cpu().float().numpy() for hs in hidden_states]
        mean_per_layer = [
            # Mean over real tokens only (exclude padding)
            hs[0, :last_token_idx + 1, :].mean(dim=0).cpu().float().numpy()
            for hs in hidden_states
        ]

    else:
        raise ValueError(f"Unknown arch: '{arch}'. Expected 'encoder' or 'decoder'.")

    return {"cls": cls_per_layer, "mean": mean_per_layer}