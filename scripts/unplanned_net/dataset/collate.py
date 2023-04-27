import torch
from torch.utils.data._utils.collate import default_collate


def collator(batch):
    batch = default_collate(batch)
    index, inputs_text, inputs, seq_len, baseline,  label, time_to_event = batch
    inputs_text = {k:[list(i) for i in zip(*v)] for k,v in inputs_text.items()}
    return index, inputs_text, inputs, seq_len, baseline,  label, time_to_event