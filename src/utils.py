import meshio
import numpy as np
import math
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection


###########################################################################
# plot a matrix of pores

def coords_fn(theta, L0, porosity, c1, c2):
    r0 = L0 * math.sqrt(2 * porosity) / math.sqrt(math.pi * (2 + c1**2 + c2**2))
    return r0 * (1 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta))


def build_base_pore(n_points, L0, porosity, c1, c2):
    thetas = [float(i) * 2 * math.pi / n_points for i in range(n_points)]
    radii = [coords_fn(float(i) * 2 * math.pi / n_points, L0, porosity, c1, c2) for i in range(n_points)]
    points = [(rtheta * np.cos(theta), rtheta * np.sin(theta))
              for rtheta, theta in zip(radii, thetas)]
    return np.array(points), np.array(radii), np.array(thetas)


def build_pore_polygon(base_pore_points, offset):
    points = [[p[0] + offset[0], p[1] + offset[1]] for p in base_pore_points]
    points = affine_group(points)
    points = np.asarray(points)
    pore = Polygon(points)
    return pore


def affine_transformation(point):
    point[0], point[1] = point[0] + 0.*point[1], point[1] - 0.*point[1]
    return point


def affine_group(points):
    points = [affine_transformation(point) for point in points]
    return points


def plot_pores_matrix(name, c1_arr, c2_arr):
    porosity = 0.5
    L0 = 0.5
    pore_radial_resolution = 120
    patches = []
    colors = []

    points = [[0, 0], [len(c1_arr)*L0, 0], [len(c1_arr)*L0, len(c2_arr)*L0], [0, len(c2_arr)*L0]]
    points = affine_group(points)

    frame = Polygon(np.asarray(points))
    patches.append(frame)
    colors.append('#808080') 

    for i in range(len(c1_arr)):
        for j in range(len(c2_arr)):
            c1 = c1_arr[i]
            c2 = c2_arr[j]
            base_pore_points, radii, thetas = build_base_pore(pore_radial_resolution, L0, porosity, c1, c2)            
            pore = build_pore_polygon(
                base_pore_points, offset=(L0 * (i + 0.5), L0 * (j + 0.5)))

            patches.append(pore)
            colors.append('#FFFFFF')
 

    fig, ax = plt.subplots()
    p = PatchCollection(patches, edgecolor=colors, facecolor=colors)
    ax.add_collection(p)
    plt.axis('equal')
    plt.axis('off')
    fig.savefig(f'data/png/rve/{name}.png', bbox_inches='tight')


def plot_pores():
    plot_pores_matrix('frame', np.linspace(-0.2, 0, 3), np.linspace(-0.1, 0.1, 3))
    plot_pores_matrix('macro', np.linspace(0., 0, 6), np.linspace(0., 0., 6))
    plot_pores_matrix('micro', np.linspace(0., 0, 2), np.linspace(0., 0., 2))


###########################################################################
# plot meshes to indicate different configurations

def show_solution(path, name):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()
    points = data.GetPoints()
    npts = points.GetNumberOfPoints()
    x = vtk_to_numpy(points.GetData())
    u = vtk_to_numpy(data.GetPointData().GetVectors('u'))
    x_ = x + u
    triangles = vtk_to_numpy(data.GetCells().GetData())
    ntri = triangles.size // 4  # number of cells
    tri = np.take(triangles, [n for n in range(
        triangles.size) if n % 4 != 0]).reshape(ntri, 3)

    fig = plt.figure(figsize=(8, 8))
    plt.triplot(x_[:, 0], x_[:, 1], tri, color='black', alpha=1., linewidth=0.25)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    fig.savefig(f'data/pdf/rve/dummy/{name}.pdf', bbox_inches='tight')

    return x[:, :2], x_[:, :2]


def plot_configs():
    path_ref = f'data/pvd/rve/dummy/bc/forward/configs000000.vtu'
    show_solution(path_ref, 'ref')
    path_ref = f'data/pvd/rve/dummy/bc/forward/configs000001.vtu'
    show_solution(path_ref, 'lag')
    path_ref = f'data/pvd/rve/dummy/bc/forward/configs000002.vtu'
    show_solution(path_ref, 'eul')


###########################################################################
# output to stl files for 3D printing

def output_stl():
    mesh = meshio.read('data/pvd/rve/auxetic/normal/inverse/u000000.vtu')
    mesh.write('data/stl/rve.stl')


if __name__ == '__main__':
    # plot_pores()
    plot_configs()
    plt.show()