import torch.nn.functional as F
import torch


def corn_label_from_logits(logits):
    # https://github.com/Raschka-research-group/coral-pytorch/blob/6b85e287118476095bac85d6f3dabc6ffb89a326/coral_pytorch/dataset.py#L123
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels


def corn_proba_from_logits(logits):
    # shared steps with corn_label_from_logits ↓
    # https://github.com/Raschka-research-group/coral-pytorch/blob/6b85e287118476095bac85d6f3dabc6ffb89a326/coral_pytorch/dataset.py#L123
    order_probas = torch.sigmoid(logits)
    order_probas = torch.cumprod(order_probas, dim=1)

    # calculation of per-class probas ↓
    class_probas = - torch.diff(order_probas, prepend=torch.ones(order_probas.shape[0], 1))
    class_probas = torch.cat((class_probas, order_probas[:, -1:]), 1)

    return class_probas


def corn_loss(logits, y_train, num_classes, weights=None):
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
    weights : torch.tensor, shape=(num_examples)
        [NEW] Weights of the training examples.
    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value.
    """
    # input checks
    assert logits.shape[1] == num_classes - 1
    if weights is not None:
        assert weights.shape[0] == logits.shape[0]

    # generate datasets for each task
    sets = []
    for i in range(num_classes-1):
        label_mask = y_train > i-1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        weight_tensor = weights[label_mask] if weights is not None else torch.ones_like(label_tensor)
        sets.append((label_mask, label_tensor, weight_tensor))

    # compute loss for each task and sum it up
    num_examples = 0
    losses = 0.
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]
        train_weights = s[2]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(
            (F.logsigmoid(pred)*train_labels
             + (F.logsigmoid(pred) - pred)*(1-train_labels))
            * train_weights
        )

        losses += loss

    return losses/num_classes
