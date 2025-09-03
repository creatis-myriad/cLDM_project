import torch
from torch.nn import functional as F



def kl_div(mu, std):
    KLD = -0.5 * torch.mean(1 + std - mu.pow(2) - std.exp())
    return KLD

def MSE(x, recon_x):
    mse = F.mse_loss(x, recon_x, reduction="mean")
    return mse

def AttributeBased_regu(z_ar, metrics, delta):
    # Compute input distance matrix
    broad_input = torch.reshape(z_ar, (-1,)).repeat(1, len(z_ar))
    input_dist_mat = broad_input - broad_input.transpose(1, 0)

    # Compute target distance matrix
    broad_target = torch.reshape(metrics, (-1,)).repeat(1, len(metrics))
    target_dist_mat = broad_target - broad_target.transpose(1, 0)

    # Compute regularization loss
    input_tanh = torch.tanh(input_dist_mat * delta)
    target_sign = torch.sign(target_dist_mat)
    loss = F.l1_loss(input_tanh, target_sign)
    return loss

def CrossEntropy(target, output, weight_classes):
    LossCE = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weight_classes, dtype=torch.float32).to(target.device), 
        ignore_index=-100, 
        reduction='mean',
        label_smoothing=0.0,
    )
    loss = LossCE(output, target.squeeze().to(torch.int64))
    return loss

