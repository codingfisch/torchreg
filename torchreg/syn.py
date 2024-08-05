import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import LinearElasticity
from .utils import INTERP_KWARGS, create_grid, smooth_kernel
LIN_ELAST_FUNC = lambda x: LinearElasticity(mu=2., lam=1.)(x)


class SyNBase:
    def __init__(self, time_steps=7, factor_diffeo=.1):
        self.time_steps = time_steps
        self.factor_diffeo = factor_diffeo
        self._grid = None

    def apply_flows(self, x, y, v_xy, v_yx):
        half_flows = self.diffeomorphic_transform(torch.cat([v_xy, v_yx, -v_xy, -v_yx]))
        half_images = self.spatial_transform(torch.cat([x, y]), half_flows[:2])
        full_flows = self.composition_transform(half_flows[:2], half_flows[2:].flip(0))
        full_images = self.spatial_transform(torch.cat([x, y]), full_flows)
        images = {'xy_half': half_images[:1], 'yx_half': half_images[1:2],
                  'xy_full': full_images[:1], 'yx_full': full_images[1:2]}
        flows = {'xy_half': half_flows[:1], 'yx_half': half_flows[1:2],
                 'xy_full': full_flows[:1], 'yx_full': full_flows[1:2]}
        flows = {k: flow.permute(0, 2, 3, 4, 1) for k, flow in flows.items()}
        return images, flows

    def diffeomorphic_transform(self, v):
        v = self.factor_diffeo * v / (2 ** self.time_steps)
        for i in range(self.time_steps):
            v = v + self.spatial_transform(v, v)
        return v

    def composition_transform(self, v1, v2):
        return v2 + self.spatial_transform(v1, v2)

    def spatial_transform(self, x, v):
        if self._grid is None:
            self._grid = create_grid(v.shape[2:], x.device)
        return F.grid_sample(x, self._grid + v.permute(0, 2, 3, 4, 1), align_corners=True, padding_mode='reflection')


class SyNRegistration(SyNBase):
    def __init__(self, scales=(4, 2, 1), iterations=(30, 30, 10), learning_rate=1e-2, verbose=True,
                 dissimilarity_function=torch.nn.MSELoss(), regularization_function=LIN_ELAST_FUNC,
                 optimizer=torch.optim.Adam, sigma_img=.2, sigma_flow=.2, lambda_=2e-5, time_steps=7):
        super().__init__(time_steps=time_steps)
        self.scales = scales
        self.iterations = iterations
        self.learning_rates = [learning_rate] * len(scales) if isinstance(learning_rate, float) else learning_rate
        self.verbose = verbose
        self.dissimilarity_function = dissimilarity_function
        self.regularization_function = regularization_function
        self.optimizer = optimizer
        self.sigma_img = sigma_img
        self.sigma_flow = sigma_flow
        self.lambda_ = lambda_
        self.v_xy = None
        self.v_yx = None
        self._grid = None

    def __call__(self, moving, static, v_xy=None, v_yx=None, return_moved=True):
        if v_xy is None:
            v_xy = torch.zeros((moving.shape[0], 3, *moving.shape[2:]))
        if v_yx is None:
            v_yx = torch.zeros((static.shape[0], 3, *static.shape[2:]))
        self.v_xy = v_xy.type(static.dtype).to(static.device)
        self.v_yx = v_yx.type(static.dtype).to(static.device)
        for scale, iters, lr in zip(self.scales, self.iterations, self.learning_rates):
            moving_shape, static_shape = [s for s in moving.shape[2:]], [s for s in static.shape[2:]]
            shape = [int(round(s / scale)) for s in static_shape]
            self._grid = create_grid(shape, static.device)
            x = F.interpolate(moving, shape, **INTERP_KWARGS) if shape != moving_shape else moving.clone()
            y = F.interpolate(static, shape, **INTERP_KWARGS) if shape != static_shape else static.clone()
            if self.sigma_img:
                sigma_img = self.sigma_img * 200 / torch.tensor(shape).int()
                x = gauss_smoothing(x, sigma_img)
                y = gauss_smoothing(y, sigma_img)
            self.fit(x, y, iters, lr)
        self._grid = create_grid(static.shape[2:], static.device)
        if return_moved:
            images, flows = self.apply_flows(moving, static, self.v_xy, self.v_yx)
            return images['xy_full'], images['yx_full'], flows['xy_full'], flows['yx_full']

    def fit(self, x, y, iterations, learning_rate):
        v_xy = F.interpolate(self.v_xy, x.shape[2:], **INTERP_KWARGS)
        v_xy = torch.nn.Parameter(v_xy, requires_grad=True)
        v_yx = F.interpolate(self.v_yx, x.shape[2:], **INTERP_KWARGS)
        v_yx = torch.nn.Parameter(v_yx, requires_grad=True)
        sigma_flow = self.sigma_flow * torch.ones(3)
        optimizer = self.optimizer([v_xy, v_yx], learning_rate)
        progress_bar = tqdm(range(iterations), disable=not self.verbose)
        for _ in progress_bar:
            optimizer.zero_grad()
            images, flows = self.apply_flows(x, y, gauss_smoothing(v_xy, sigma_flow), gauss_smoothing(v_yx, sigma_flow))
            dissimilarity = (self.dissimilarity_function(x, images['yx_full']) +
                             self.dissimilarity_function(y, images['xy_full']) +
                             self.dissimilarity_function(images['yx_half'], images['xy_half']))
            regularization = (self.regularization_function(flows['yx_full']) +
                              self.regularization_function(flows['xy_full']))
            loss = dissimilarity + self.lambda_ * regularization
            progress_bar.set_description(f'Loss: {loss.item()}, '
                                         f'Dissimilarity: {dissimilarity.item()}, '
                                         f'Regularization: {regularization.item()}')
            loss.backward()
            optimizer.step()
        v_xy, v_yx = v_xy.detach(), v_yx.detach()
        v_xy, v_yx = gauss_smoothing(v_xy, sigma_flow), gauss_smoothing(v_yx, sigma_flow)
        self.v_xy = F.interpolate(v_xy, self.v_xy.shape[2:], **INTERP_KWARGS) if self.v_xy.shape != v_xy.shape else v_xy
        self.v_yx = F.interpolate(v_yx, self.v_yx.shape[2:], **INTERP_KWARGS) if self.v_yx.shape != v_yx.shape else v_yx


def gauss_smoothing(x, sigma):
    half_kernel_size = np.array(x.shape[2:]) // 50
    kernel_size = 1 + 2 * half_kernel_size.clip(min=1)
    kernel = smooth_kernel(kernel_size.tolist(), sigma).to(x.device)
    kernel = kernel[None, None].repeat(x.shape[1], 1, 1, 1, 1)
    x = F.pad(x, (kernel_size.repeat(2)[::-1] // 2).tolist(), mode='replicate')
    return F.conv3d(x.type(torch.float32), kernel, groups=x.shape[1]).type(x.dtype)
