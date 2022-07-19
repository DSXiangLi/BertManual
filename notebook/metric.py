# -*-coding:utf-8 -*-
import torch
import numpy as np
from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score, Recall, Precision


def binary_cls_metrics(model, valid_loader, loss_fn, device, threshold=0.5):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    model.eval()

    metrics = {
        'acc': Accuracy(threshold=threshold).to(device),
        'auc': AUROC().to(device),
        'ap': AveragePrecision().to(device),
        'f1': F1Score().to(device),
        'recall': Recall(threshold=threshold).to(device),
        'precision': Precision(threshold=threshold).to(device)
    }
    val_loss = []

    for batch in valid_loader:
        input_ids, token_type_ids, attention_mask, label_ids = tuple(t.to(device) for t in batch.values())

        with torch.no_grad():
            logits = model(input_ids, token_type_ids, attention_mask)

        loss = loss_fn(logits, label_ids)
        val_loss.append(loss.item())

        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = (probs[:, 1] > threshold).int().to(device)
        for metric in metrics.values():
            if metric in ['auc', 'ap']:
                metric.update(probs, label_ids)
            else:
                metric.update(preds, label_ids)

    multi_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
    multi_metrics['val_loss'] = np.mean(val_loss)
    return multi_metrics


def multi_cls_metrics(model, valid_loader, loss_fn, device):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    model.eval()

    # Tracking variables
    metrics = {}
    for avg in ['micro', 'macro']:
        metrics.update({
            f'acc_{avg}': Accuracy(average=avg, num_classes=model.label_size).to(device),
            f'f1_{avg}': F1Score(num_classes=model.label_size).to(device),
            f'recall_{avg}': Recall(average=avg, num_classes=model.label_size).to(device),
            f'precision_{avg}': Precision(average=avg, num_classes=model.label_size).to(device)
        })
    # for AUC and AP only macro average is supported for multiclass
    metrics.update({
        f'auc_macro': AUROC(average='macro', num_classes=model.label_size).to(device),
        f'ap_macro': AveragePrecision(average='macro', num_classes=model.label_size).to(device),

    })
    val_loss = []

    for batch in valid_loader:
        input_ids, token_type_ids, attention_mask, label_ids = tuple(t.to(device) for t in batch.values())

        with torch.no_grad():
            logits = model(input_ids, token_type_ids, attention_mask)

        loss = loss_fn(logits, label_ids)
        val_loss.append(loss.item())

        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(probs)
        for metric in metrics.values():
            if 'auc' in metric or 'ap' in metric:
                metric.update(probs, label_ids)
            else:
                metric.update(preds, label_ids)

    multi_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
    multi_metrics['val_loss'] = np.mean(val_loss)
    return multi_metrics


def binary_cls_log(epoch, binary_metrics):
    print("\n")
    print(f"{'Epoch':^7} | {'Val Acc':^9} | {'Val AUC':^9} | {'Val AP':^9} | {'Precision':^9} | {'Recall':^9} | {'Val F1':^9}")
    print('-'*80)
    print(f"{epoch + 1:^7} | {binary_metrics['acc']:^9.3%} | {binary_metrics['auc']:^9.3%} |",
          f"{binary_metrics['ap']:^9.3%} | {binary_metrics['precision']:^9.3%} |",
          f"{binary_metrics['recall']:^9.3%} | {binary_metrics['f1']:^9.3%} ")
    print("\n")


def multi_cls_log(epoch, multi_metrics):
    print("\n")
    print(f"{'Epoch':^7} | {'Macro Acc':^9} | {'Macro AUC':^9} | {'Macro AP':^9} |",
          f"{'Macro Precision':^15} | {'Macro Recall':^12} | {'Macro F1':^9}")
    print('-' * 90)
    print(f"{epoch + 1:^7} | {multi_metrics['acc_macro']:^9.3%} | {multi_metrics['auc_macro']:^9.3%} |",
          f"{multi_metrics['ap_macro']:^9.3%} | {multi_metrics['precision_macro']:^15.3%} |",
          f"{multi_metrics['recall_macro']:^12.3%} | {multi_metrics['f1_macro']:^9.3%} ")

    print(f"{'Epoch':^7} | {'Micro Acc':^9} | {'Micro AUC':^9} | {'Micro AP':^9} |",
          f"{'Micro Precision':^15} | {'Micro Recall':^12} | {'Micro F1':^9}")
    print('-' * 90)
    print(f"{epoch + 1:^7} | {multi_metrics['acc_micro']:^9.3%} | {'-':^9} |",
          f"{'-':^9} | {multi_metrics['precision_micro']:^15.3%} |",
          f"{multi_metrics['recall_micro']:^12.3%} | {multi_metrics['f1_micro']:^9.3%} ")
    print("\n")
