import torch
from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        # grab labels (you stored them under "label")
        labels = inputs.get("labels", inputs.get("label"))
        # forward pass (drop label so model doesn't double‐expect it)
        inputs_for_model = {k: v for k, v in inputs.items() if k not in ("label", "labels")}
        outputs = model(**inputs_for_model)
        logits = outputs.logits
        # build loss fn with weights on the same device
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss