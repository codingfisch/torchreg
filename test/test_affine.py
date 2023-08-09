import torch
from unittest import TestCase
from torchreg import AffineRegistration
from torchreg.affine import compose_affine, affine_transform, init_parameters, _check_parameter_shapes


class TestAffineRegistration(TestCase):
    def test_fit(self):
        for batch_size in [1, 2]:
            for n_dim in [2, 3]:
                reg = AffineRegistration(scales=(1,), is_3d=n_dim == 3, learning_rate=1e-1, verbose=False)
                moving = synthetic_image(batch_size, n_dim, shift=1)
                static = synthetic_image(batch_size, n_dim, shift=0)
                fitted_moved = reg(moving, static, return_moved=True)
                fitted_affine = reg.get_affine()
                affine = torch.stack(batch_size * [torch.eye(n_dim + 1)[:n_dim]])
                affine[:, -1, -1] += -1/3
                self.assertTrue(torch.allclose(fitted_affine, affine, atol=1e-2))
                moved = affine_transform(moving, affine)
                self.assertTrue(torch.allclose(fitted_moved, moved, atol=1e-2))

    def test_affine_transform(self):
        for batch_size in [1, 2]:
            for n_dim in [2, 3]:
                moving = synthetic_image(batch_size, n_dim, shift=1)
                static = synthetic_image(batch_size, n_dim, shift=0)
                affine = torch.stack(batch_size * [torch.eye(n_dim + 1)[:n_dim]])
                affine[:, -1, -1] += -1/3
                moved = affine_transform(moving, affine)
                self.assertTrue(torch.allclose(moved, static, atol=1e-6))

    def test_init_parameters(self):
        for batch_size in [1, 2]:
            for is_3d in [False, True]:
                params = init_parameters(is_3d=is_3d, batch_size=batch_size)
                self.assertIsInstance(params, list)
                self.assertEqual(len(params), 4)
                for param in params:
                    self.assertTrue(isinstance(param, torch.nn.Parameter))
                _check_parameter_shapes(*params, is_3d=is_3d, batch_size=batch_size)

    def test_compose_affine(self):
        for batch_size in [1, 2]:
            for n_dim in [2, 3]:
                translation = torch.zeros(batch_size, n_dim)
                rotation = torch.stack(batch_size * [torch.eye(n_dim)])
                zoom = torch.ones(batch_size, n_dim)
                shear = torch.zeros(batch_size, n_dim)
                affine = compose_affine(translation, rotation, zoom, shear)
                id_affine = torch.stack(batch_size * [torch.eye(n_dim + 1)[:n_dim]])
                self.assertTrue(torch.equal(affine, id_affine))


def synthetic_image(batch_size, n_dim, shift):
    shape = [batch_size, 1, 7, 7, 7][:2 + n_dim]
    x = torch.zeros(*shape)
    if n_dim == 3:
        x[:, :, 2 - shift:5 - shift, 2:5, 2:5] = 1
    else:
        x[:, :, 2 - shift:5 - shift, 2:5] = 1
    return x
