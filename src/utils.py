import meshio

if __name__ == '__main__':
    mesh = meshio.read('data/pvd/auxetic/inverse/u000038.vtu')
    mesh.write('data/stl/rve.stl')