import torch
import torch.nn.functional as F
from tqdm import tqdm


class AffineRegistration:
    def __init__(self, scales=(4, 2), iterations=(500, 100), is_3d=True, learning_rate=1e-2,
                 verbose=True, dissimilarity_function=torch.nn.MSELoss(), optimizer=torch.optim.Adam,
                 init_translation=None, init_rotation=None, init_zoom=None, init_shear=None,
                 with_translation=True, with_rotation=True, with_zoom=True, with_shear=False,
                 align_corners=True, interp_mode=None, padding_mode='border'):
        self.scales = scales
        self.iterations = iterations[:len(scales)]
        self.is_3d = is_3d
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.dissimilarity_function = dissimilarity_function
        self.optimizer = optimizer
        self.inits = (init_translation, init_rotation, init_zoom, init_shear)
        self.withs = (with_translation, with_rotation, with_zoom, with_shear)
        self.align_corners = align_corners
        self.interp_mode = 'trilinear' if is_3d else 'bilinear' if interp_mode is None else interp_mode
        self.padding_mode = padding_mode
        self._parameters = None

    def __call__(self, moving, static, return_moved=True):
        if len(moving.shape) - 4 != self.is_3d or len(static.shape) - 4 != self.is_3d:
            raise ValueError(f'Expected moving and static to be {4 + self.is_3d}D Tensors (2 + Spatial Dims.). '
                             f'Got size {moving.shape} and {static.shape}.')
        if moving.shape != static.shape:
            raise ValueError(f'Expected moving and static to have the same size. '
                             f'Got size {moving.shape} and {static.shape}.')

        self._parameters = init_parameters(self.is_3d, len(static), static.device, *self.withs, *self.inits)
        interp_kwargs = {'mode': self.interp_mode, 'align_corners': self.align_corners}
        moving_ = F.interpolate(moving, static.shape[2:], **interp_kwargs)
        for scale, iters in zip(self.scales, self.iterations):
            moving_small = F.interpolate(moving_, scale_factor=1 / scale, **interp_kwargs)
            static_small = F.interpolate(static, scale_factor=1 / scale, **interp_kwargs)
            self._fit(moving_small, static_small, iters)
        return self.transform(moving, static.shape[2:]).detach() if return_moved else None

    def _fit(self, moving, static, iterations):
        optimizer = self.optimizer(self._parameters, self.learning_rate)
        progress_bar = tqdm(range(iterations), disable=not self.verbose)
        for self.iter in progress_bar:
            optimizer.zero_grad()
            moved = self.transform(moving, static.shape[2:], with_grad=True)
            loss = self.dissimilarity_function(moved, static)
            progress_bar.set_description(f'Shape: {[*static.shape]}; Dissimiliarity: {loss.item()}')
            loss.backward()
            optimizer.step()

    def transform(self, moving, shape=None, with_grad=False):
        affine = self.get_affine(with_grad)
        return affine_transform(moving, affine, shape, self.interp_mode, self.padding_mode, self.align_corners)

    def get_affine(self, with_grad=False):
        affine = compose_affine(*self._parameters)
        return affine if with_grad else affine.detach()


def affine_transform(x, affine, shape=None, mode='bilinear', padding_mode='border', align_corners=True):
    shape = x.shape[2:] if shape is None else shape
    grid = F.affine_grid(affine, [len(x), len(shape), *shape], align_corners)
    sample_mode = 'bilinear' if mode == 'trilinear' else mode  # grid_sample converts 'bi-' to 'trilinear' internally
    return F.grid_sample(x, grid, sample_mode, padding_mode, align_corners)


def init_parameters(is_3d=True, batch_size=1, device='cpu', with_translation=True, with_rotation=True, with_zoom=True,
                    with_shear=True, init_translation=None, init_rotation=None, init_zoom=None, init_shear=None):
    _check_parameter_shapes(init_translation, init_rotation, init_zoom, init_shear, is_3d, batch_size)
    n_dim = 2 + is_3d
    translation = torch.zeros(batch_size, n_dim).to(device) if init_translation is None else init_translation
    rotation = torch.stack(batch_size * [torch.eye(n_dim)]).to(device) if init_rotation is None else init_rotation
    zoom = torch.ones(batch_size, n_dim).to(device) if init_zoom is None else init_zoom
    shear = torch.zeros(batch_size, n_dim).to(device) if init_shear is None else init_shear
    params = [translation, rotation, zoom, shear]
    with_grad = [with_translation, with_rotation, with_zoom, with_shear]
    return [torch.nn.Parameter(param, requires_grad=grad) for param, grad in zip(params, with_grad)]


def compose_affine(translation, rotation, zoom, shear):
    _check_parameter_shapes(translation, rotation, zoom, shear, zoom.shape[-1] == 3, zoom.shape[0])
    square_matrix = torch.diag_embed(zoom)
    if zoom.shape[-1] == 3:
        square_matrix[..., 0, 1:] = shear[..., :2]
        square_matrix[..., 1, 2] = shear[..., 2]
    else:
        square_matrix[..., 0, 1] = shear[..., 0]
    square_matrix = rotation @ square_matrix
    return torch.cat([square_matrix, translation[:, :, None]], dim=-1)


def _check_parameter_shapes(translation, rotation, zoom, shear, is_3d=True, batch_size=1):
    n_dim = 2 + is_3d
    params = {'translation': translation, 'rotation': rotation, 'zoom': zoom, 'shear': shear}
    for name, param in params.items():
        if param is not None:
            desired_shape = (batch_size, n_dim, n_dim) if name == 'rotation' else (batch_size, n_dim)
            if param.shape != desired_shape:
                raise ValueError(f'Expected {name} to be size {desired_shape} since batch_size is {batch_size} '
                                 f'and is_3d is {is_3d} -> {2 + is_3d} dimensions. Got size {param.shape}.')
