import torch
import torch.nn.functional as F
INTERP_KWARGS = {'mode': 'trilinear', 'align_corners': True}


def smooth_kernel(kernel_size, sigma):
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32, device=sigma.device) for size in kernel_size])
    kernel = 1
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * (2 * torch.pi)**.5) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
    return kernel / kernel.sum()


def jacobi_determinant(u, id_grid=None):
    gradient = jacobi_gradient(u, id_grid)
    dx, dy, dz = gradient[..., 2], gradient[..., 1], gradient[..., 0]
    jdet0 = dx[2] * (dy[1] * dz[0] - dy[0] * dz[1])
    jdet1 = dx[1] * (dy[2] * dz[0] - dy[0] * dz[2])
    jdet2 = dx[0] * (dy[2] * dz[1] - dy[1] * dz[2])
    jdet = jdet0 - jdet1 + jdet2
    return F.pad(jdet[None, None, 2:-2, 2:-2, 2:-2], (2, 2, 2, 2, 2, 2), mode='replicate')[0, 0]


def jacobi_gradient(u, id_grid=None):
    if id_grid is None:
        id_grid = create_grid(u.shape[1:4], u.device)
    x = 0.5 * (u + id_grid) * (torch.tensor(u.shape[1:4], device=u.device, dtype=u.dtype) - 1)
    window = torch.tensor([-.5, 0, .5], device=u.device)
    w = torch.zeros((3, 1, 3, 3, 3), device=u.device, dtype=u.dtype)
    w[2, 0, :, 1, 1] = window
    w[1, 0, 1, :, 1] = window
    w[0, 0, 1, 1, :] = window
    x = x.permute(4, 0, 1, 2, 3)
    x = F.conv3d(x, w)
    x = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')  # 'circular' for bfloat16
    return x.permute(0, 2, 3, 4, 1)


def create_grid(shape, device):
    return F.affine_grid(torch.eye(4, device=device)[None, :3], [1, 3, *shape], align_corners=INTERP_KWARGS['align_corners'])
