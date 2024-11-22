import numpy as np
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import Trainer


class VarMisUseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss_fct = BCEWithLogitsLoss()
        sftmax = nn.Softmax(dim=1)
        bce_loss = nn.BCELoss()

        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # classification loss
        classification_loss = loss_fct(outputs.logits[:, 0, 0], labels[:, 0, 0])

        # TODO perform [localization_labels != -100] before softmax
        # localization loss
        localization_labels = labels[:, 1:, 0]
        localization_logits = sftmax(outputs.logits[:, 1:, 0])[localization_labels != -100]
        localization_loss = bce_loss(localization_logits, localization_labels[localization_labels != -100])

        # # repair loss
        repair_labels = labels[:, 1:, 1]
        repair_logits = sftmax(outputs.logits[:, 1:, 1])[repair_labels != -100]
        repair_loss = bce_loss(repair_logits, repair_labels[repair_labels != -100])

        loss = (classification_loss + localization_loss + repair_loss) / 3

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_preds):
    logits, labels = eval_preds

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    classification = sigmoid(logits[:, 0, 0]) > 0.5
    ground_truth = labels[:, 0, 0] == 1
    accuracy = np.round(np.mean(classification == ground_truth), 4)

    buggy_logit = logits[ground_truth]
    buggy_label = labels[ground_truth]

    # localization accuracy
    localization_logit = buggy_logit[:, 1:, 0]
    localization_label = buggy_label[:, 1:, 0]

    localization_logit[localization_label == -100] = float("-inf")

    localizations_pred = np.argmax(localization_logit, axis=1)
    localizations_label = np.argmax(localization_label, axis=1)
    localizations_hits = localizations_pred == localizations_label
    localization_accuracy = np.round(np.mean(localizations_hits), 4)

    # localization + repair accuracy
    repair_logit = buggy_logit[:, 1:, 1]
    repair_label = buggy_label[:, 1:, 1]

    repair_logit[repair_label == -100] = float("-inf")

    repair_pred = np.argmax(repair_logit, axis=1)
    hits = 0.
    total = 0.
    for j, index in enumerate(repair_pred):
        if repair_label[j][index] == 1 and localizations_hits[j]:
            hits += 1
        total += 1

    repair_label = np.argmax(repair_label, axis=1)

    repair_accuracy = np.round(hits / total, 4)

    return {
        "classification_accuracy": accuracy,
        "localization_accuracy": localization_accuracy,
        "localization+repair_accuracy": repair_accuracy,
    }
