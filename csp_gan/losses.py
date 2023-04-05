import torch
import torch.nn as nn

class R1(nn.Module):
    def __init__(self):
        super(R1, self).__init__()

    def forward(self, prediction_real: torch.Tensor, real_sample: torch.Tensor, sigma: int) -> torch.Tensor:
        grad_real = torch.autograd.grad(outputs=prediction_real.sum(), inputs=real_sample, create_graph=True)[0]
        grad_penalty: torch.Tensor = 0.5 * sigma * (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        return grad_penalty


class R2(nn.Module):
    def __init__(self):
        super(R1, self).__init__()

    def forward(self, prediction_fake: torch.Tensor, fake_sample: torch.Tensor, sigma: int) -> torch.Tensor:
        grad_real = torch.autograd.grad(outputs=prediction_fake.sum(), inputs=fake_sample, create_graph=True)[0]
        grad_penalty: torch.Tensor = 0.5 * sigma * (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        return grad_penalty
