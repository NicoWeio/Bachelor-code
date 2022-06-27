import torch.nn.functional as F
import torch

def corn_label_from_logits(logits):
    # https://github.com/Raschka-research-group/coral-pytorch/blob/6b85e287118476095bac85d6f3dabc6ffb89a326/coral_pytorch/dataset.py#L123
    probas = torch.sigmoid(logits)
    # print(probas)
    probas = torch.cumprod(probas, dim=1)
    # print(probas)
    predict_levels = probas > 0.5
    # print(predict_levels)
    predicted_labels = torch.sum(predict_levels, dim=1)
    # print(predicted_labels)
    return predicted_labels


def corn_proba_from_logits(logits):
    # logits = logits.detach().numpy()

    # shared steps with corn_label_from_logits ↓
    # https://github.com/Raschka-research-group/coral-pytorch/blob/6b85e287118476095bac85d6f3dabc6ffb89a326/coral_pytorch/dataset.py#L123
    order_probas = torch.sigmoid(logits)
    order_probas = torch.cumprod(order_probas, dim=1)

    # calculation of per-class probas ↓
    class_probas = - torch.diff(order_probas, prepend=torch.ones(order_probas.shape[0], 1))
    class_probas = torch.cat((class_probas, order_probas[:, -1:]), 1)

    return class_probas


def corn_loss(logits, y_train, num_classes):
    """Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.
    Parameters
    ----------
    logits : torch.tensor, shape=(num_examples, num_classes-1)
        Outputs of the CORN layer.
    y_train : torch.tensor, shape=(num_examples)
        Torch tensor containing the class labels.
    num_classes : int
        Number of unique class labels (class labels should start at 0).
    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value.
    """
    sets = []
    for i in range(num_classes-1):
        label_mask = y_train > i-1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(F.logsigmoid(pred)*train_labels
                          + (F.logsigmoid(pred) - pred)*(1-train_labels))
        losses += loss

    return losses/num_classes
