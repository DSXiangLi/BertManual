# -*-coding:utf-8 -*-
import torch
from torch import nn
import numpy as np

def classification_inference(data_loader, model, device):
    model.eval()

    all_preds = []
    all_probs = []
    for batch in data_loader:
        # Load batch to GPU
        input_ids, token_type_ids, attention_mask, label_ids = tuple(t.to(device) for t in batch.values())

        # Compute logits
        with torch.no_grad():
            logits = model(input_ids, token_type_ids, attention_mask)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        all_preds += np.argmax(probs, axis=1).tolist()
        all_probs += probs.tolist()

    output = {
        'pred': all_preds,
        'prob': all_probs
    }
    return output