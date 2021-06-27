import torch
import torch.nn.functional as F


def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples


def accuracy_check_real(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == torch.where(labels == 1)[1]).sum().item()
            num_samples += labels.size(0)
    return total / num_samples


def confidence_update(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        batch_outputs = model(batchX)
        temp_un_conf = F.softmax(batch_outputs, dim=1)
        # un_confidence stores the weight of each example
        confidence[batch_index, :] = temp_un_conf * batchY
        # weight[batch_index] = 1.0/confidence[batch_index, :].sum(dim=1)
        base_value = confidence.sum(dim=1).unsqueeze(1).repeat(
            1, confidence.shape[1])
        confidence = confidence / base_value
    return confidence


def confidence_update_lw(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        device = batchX.device
        batch_outputs = model(batchX)
        sm_outputs = F.softmax(batch_outputs, dim=1)

        onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
        onezero[batchY > 0] = 1
        counter_onezero = 1 - onezero
        onezero = onezero.to(device)
        counter_onezero = counter_onezero.to(device)

        new_weight1 = sm_outputs * onezero
        new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
            confidence.shape[1], 1).transpose(0, 1)
        new_weight2 = sm_outputs * counter_onezero
        new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
            confidence.shape[1], 1).transpose(0, 1)
        new_weight = new_weight1 + new_weight2

        confidence[batch_index, :] = new_weight
        return confidence
