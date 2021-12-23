import torch
class AutomaticWeightedLoss(torch.nn.Module):
    def __init__(self, is_regression, reduction='sum'):
        super(AutomaticWeightedLoss, self).__init__()
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, losses):
        dtype = losses.dtype
        device = losses.device
        stds = (torch.exp(self.log_vars) ** (1 / 2)).to(device).to(dtype)
        self.is_regression = self.is_regression.to(device).to(dtype)
        coeffs = 1 / ((self.is_regression + 1) * (stds ** 2))  # 要不要×1/2
        multi_task_losses = coeffs * losses + torch.log(stds)
        print('multitask coeffs is :',coeffs)
        if self.reduction == 'sum':
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == 'mean':
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses
