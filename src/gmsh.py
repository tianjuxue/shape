import gmsh
import sys
import numpy as np
import meshio


def msh_to_xdmf_2d(msh_file, xdmf_file):
    msh = meshio.read(msh_file)
    domainType = 'triangle'
    domain_cells = np.vstack(np.array([cells.data for cells in msh.cells if cells.type == domainType]))
    points = msh.points[:, :-1]
    domain_mesh = meshio.Mesh(points=points, cells=[(domainType, domain_cells)])
    meshio.write(xdmf_file, domain_mesh)


def create_RVE_mesh(interactive=False):
    gmsh.initialize(sys.argv)
    model = gmsh.model
    model.add('RVE')

    L0 = 0.5
    porosity = 0.5
    R = L0 * np.sqrt(2*porosity) / np.sqrt(2*np.pi)
    n_divisions = 2
    n_arcs = 4
    h = 0.02 

    p1 = model.geo.addPoint(0, 0, 0, h)
    p2 = model.geo.addPoint(n_divisions*L0, 0, 0, h)
    p3 = model.geo.addPoint(n_divisions*L0, n_divisions*L0, 0, h)
    p4 = model.geo.addPoint(0, n_divisions*L0, 0, h)

    bottom = model.geo.addLine(p1, p2)
    top = model.geo.addLine(p4, p3)
    left = model.geo.addLine(p1, p4)
    right = model.geo.addLine(p2, p3)
    square = model.geo.addCurveLoop([bottom, right, -top, -left])

    arcs = []
    curveloops = []
    for x_ind in range(n_divisions):
        arcs.append([])
        curveloops.append([])
        for y_ind in range(n_divisions):
            center = model.geo.addPoint(x_ind*L0 + L0/2, y_ind*L0 + L0/2, 0, h)
            points = []
            for j in range(n_arcs):
                points.append(model.geo.addPoint(x_ind*L0 + L0/2 + R*np.cos(2*np.pi*j/n_arcs), y_ind*L0 + L0/2 + R*np.sin(2*np.pi*j/n_arcs), 0, h))
            arc0 = model.geo.addCircleArc(points[0], center, points[1])
            arc1 = model.geo.addCircleArc(points[2], center, points[1])
            arc2 = model.geo.addCircleArc(points[2], center, points[3])
            arc3 = model.geo.addCircleArc(points[0], center, points[3])
            curveloop = model.geo.addCurveLoop([arc0, -arc1, arc2, -arc3])
            curveloops[x_ind].append(curveloop)
            arcs[x_ind].append([arc0, arc1, arc2, arc3])

    curveloops = np.array(curveloops)
    arcs = np.array(arcs)

    domain = model.geo.addPlaneSurface([square] + list(curveloops.reshape(-1)))
    phy_group = gmsh.model.addPhysicalGroup(2, [domain])
    # gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.model.geo.synchronize()

    for x_ind in range(n_divisions):
        for y_ind in range(n_divisions):
            if x_ind == 0 and y_ind == 0:
                gmsh.model.mesh.setPeriodic(1, [arcs[0, 0, 1]], [arcs[0, 0, 0]], [-1, 0, 0, L0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
                gmsh.model.mesh.setPeriodic(1, [arcs[0, 0, 2]], [arcs[0, 0, 0]], [-1, 0, 0, L0, 0, -1, 0, L0, 0, 0, 1, 0, 0, 0, 0, 1])
                gmsh.model.mesh.setPeriodic(1, [arcs[0, 0, 3]], [arcs[0, 0, 0]], [1, 0, 0, 0, 0, -1, 0, L0, 0, 0, 1, 0, 0, 0, 0, 1])
            else:
                transform = [1, 0, 0, x_ind*L0, 0, 1, 0,  y_ind*L0, 0, 0, 1, 0, 0, 0, 0, 1]
                for i in range(n_arcs):
                    gmsh.model.mesh.setPeriodic(1, [arcs[x_ind, y_ind, i]], [arcs[0, 0, i]], transform)  
 
    gmsh.model.mesh.setPeriodic(1, [right], [left], [1, 0, 0, n_divisions*L0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    gmsh.model.mesh.setPeriodic(1, [top], [bottom], [1, 0, 0, 0, 0, 1, 0, n_divisions*L0, 0, 0, 1, 0, 0, 0, 0, 1])
 
    model.mesh.generate(2)
    msh_file = 'data/msh/RVE.msh'
    gmsh.write(msh_file)
    if interactive:
        gmsh.fltk.run()
    gmsh.finalize()
    xdmf_file = 'data/xdmf/rve/mesh/mesh.xdmf'
    msh_to_xdmf_2d(msh_file, xdmf_file)


if __name__ == '__main__':
    create_RVE_mesh()
