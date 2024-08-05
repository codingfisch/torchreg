import torch
import torch.nn.functional as F

from .utils import create_grid, jacobi_gradient


def dice_loss(x1, x2):
    return 1 - dice_score(x1, x2)


def dice_score(x1, x2):
    dim = [2, 3, 4] if len(x2.shape) == 5 else [2, 3]
    inter = torch.sum(x1 * x2, dim=dim)
    union = torch.sum(x1 + x2, dim=dim)
    return (2. * inter / union).mean()


class LinearElasticity(torch.nn.Module):
    def __init__(self, mu=2., lam=1., refresh_id_grid=False):
        super(LinearElasticity, self).__init__()
        self.mu = mu
        self.lam = lam
        self.id_grid = None
        self.refresh_id_grid = refresh_id_grid

    def forward(self, u):
        if self.id_grid is None or self.refresh_id_grid:
            self.id_grid = create_grid(u.shape[1:4], u.device)
        gradients = jacobi_gradient(u, self.id_grid)
        u_xz, u_xy, u_xx = jacobi_gradient(gradients[None, 2], self.id_grid)
        u_yz, u_yy, u_yx = jacobi_gradient(gradients[None, 1], self.id_grid)
        u_zz, u_zy, u_zx = jacobi_gradient(gradients[None, 0], self.id_grid)
        e_xy = .5 * (u_xy + u_yx)
        e_xz = .5 * (u_xz + u_zx)
        e_yz = .5 * (u_yz + u_zy)
        sigma_xx = 2 * self.mu * u_xx + self.lam * (u_xx + u_yy + u_zz)
        sigma_xy = 2 * self.mu * e_xy
        sigma_xz = 2 * self.mu * e_xz
        sigma_yy = 2 * self.mu * u_yy + self.lam * (u_xx + u_yy + u_zz)
        sigma_yz = 2 * self.mu * e_yz
        sigma_zz = 2 * self.mu * u_zz + self.lam * (u_xx + u_yy + u_zz)
        return (sigma_xx ** 2 + sigma_xy ** 2 + sigma_xz ** 2 +
                sigma_yy ** 2 + sigma_yz ** 2 + sigma_zz ** 2).mean()


class NCC(torch.nn.Module):
    def __init__(self, kernel_size=7, epsilon_numerator=1e-5, epsilon_denominator=1e-5):
        super(NCC, self).__init__()
        self.kernel_size = kernel_size
        self.eps_nr = epsilon_numerator
        self.eps_dr = epsilon_denominator

    def forward(self, pred, targ):
        kernel = torch.ones([*targ.shape[:2]] + 3 * [self.kernel_size], device=targ.device)
        t_sum = F.conv3d(targ, kernel, padding=self.kernel_size // 2)
        p_sum = F.conv3d(pred, kernel, padding=self.kernel_size // 2)
        t2_sum = F.conv3d(targ ** 2, kernel, padding=self.kernel_size // 2)
        p2_sum = F.conv3d(pred ** 2, kernel, padding=self.kernel_size // 2)
        tp_sum = F.conv3d(targ * pred, kernel, padding=self.kernel_size // 2)
        cross = tp_sum - t_sum * p_sum / kernel.sum()
        t_var = F.relu(t2_sum - t_sum ** 2 / kernel.sum())
        p_var = F.relu(p2_sum - p_sum ** 2 / kernel.sum())
        cc = (cross ** 2 + self.eps_nr) / (t_var * p_var + self.eps_dr)
        return -torch.mean(cc)
