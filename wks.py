# -*- coding: utf-8 -*-
import numpy as np

import common.heat.laplace_function_builder as laplace_function_builder

import common.mesh.triangle_metric as triangle_metric
import common.laplacian.laplacian as laplacian

from common.plotting import mvc_plotting


def compute_wave_kernel_signature_from_eigendecomposition(eigen_vectors, eigen_values, energy_steps=None, absolute_sigma=None, num_steps=None, relative_sigma=None):
    eigen_values = np.abs(eigen_values)
    idx = np.argsort(eigen_values)
    eigen_values = eigen_values[idx[1:]]
    eigen_vectors = eigen_vectors[:, idx[1:]]

    if energy_steps is None:
        assert num_steps is not None
        energy_steps = np.log(np.linspace(eigen_values[1], eigen_values[-1], num_steps))

    if absolute_sigma is None:
        if relative_sigma is not None:
            absolute_sigma = (energy_steps.max() - energy_steps.min()) * relative_sigma
        else:
            # from paper
            delta = (energy_steps.max() - energy_steps.min()) / energy_steps.size
            absolute_sigma = 7 * delta

    nv = eigen_vectors.shape[0]
#    nev = eigen_vectors.shape[1]
    num_steps = energy_steps.size

    desc = np.zeros((nv, num_steps))
    for idx, e in enumerate(energy_steps):

        coeff = np.exp(-(e-np.log(eigen_values))**2/(2*absolute_sigma))
        desc[:, idx] = 1/coeff.sum() * (eigen_vectors**2).dot(coeff)

    return desc

def compare_wave_kernel_signature(target, query):
    distance = np.abs((target - query[None, :])/(target + query[None, :]+1e-10)).sum(1)

    return distance


def compare_L2(target, query):
    distance = ((target - query[None, :])**2).sum(1)

    return distance


def compute_wave_kernel_signature_from_tris(tris, points, nev, energy_steps, sigma, **kwargs):
    edge_length_matrix = triangle_metric.get_edge_length_matrix(tris, points)

    Cotan, vertex_area, LCotan = laplacian.build_cotan_laplacian_components_new(tris, edge_length_matrix)

    builder = laplace_function_builder.LaplaceFunctionBuilder(vertex_area, Cotan, nev)

    return compute_wave_kernel_signature_from_eigendecomposition(builder.eigen_vectors, builder.eigen_values, energy_steps=energy_steps, relative_sigma=sigma, **kwargs)

if __name__ == "__main__":

    from experiments.spectral_deformation.devlog.utils import load_database_mesh

    base_path = os.path.expanduser(os.environ['CGPRAK'])
    filename_from = os.path.join(base_path,'Tosca','hi-res','cat1.mat')
    #filename_from = 'entropy/entropy_birds_08.obj'

    tris, points = load_database_mesh(filename_from, prescale=None)
    nev = 50 # 100, 300
    energy_steps = None
    sigma = 1.0/30*2
    desc = compute_wave_kernel_signature_from_tris(tris, points, nev, energy_steps, sigma, num_steps=100)

    if False:
        plot = mvc_plotting.InteractiveMeshPlotWithVariables([('time_step', 0, desc.shape[1]-1)], tris, points, title='wks')
        def colorize(myplot, model, time_step, **kwargs):
            model.scalars = desc[:, time_step]
        plot.controller.callback_variable_change = colorize
        plot.start()

    if True:
        plot = mvc_plotting.InteractiveMeshPlot(tris, points, title='wks similarity')
        def colorize(model, vertexid, **kwargs):
            distance = compare_wave_kernel_signature(desc, desc[vertexid, :])
            distance = compare_L2(desc, desc[vertexid, :])
    #        distance[:] = 1
            model.scalars = distance / np.max(distance)
        plot.controller.callback_mouse_pick = colorize