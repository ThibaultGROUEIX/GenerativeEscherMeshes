from math import cos, pi, sin
import igl
import numpy as np
import math


def get_2d_square_mesh(resolution, num_labels=1):
    """get_2d_square_mesh

    Args:
        resolution (_type_): 50
        num_labels (int, optional): Split the faces in separate groups. If num_labels=2, the split is diagonal, else num_labels has to be a square number
        and the triangle are split using a grid. Defaults to 1.

    Returns:
        _type_: vertices, faces, and per-face labels
    """
    nx, ny = (resolution, resolution)
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = xv.ravel()
    yv = yv.ravel()
    points = np.stack((xv, yv), axis=1)

    faces_1 = np.concatenate([np.array([[i, i + 1, i + resolution + 1]]) for i in range((resolution) ** 2)])
    mask_1 = np.stack(
        [
            True if (i % resolution != (resolution - 1)) and (i + resolution + 1 < resolution**2) else False
            for i in range(resolution**2)
        ]
    )
    faces_1[resolution-2] = np.array([resolution-2, resolution-1, resolution-2 + resolution])
    faces_1[resolution ** 2 - 2*resolution] = np.array([resolution ** 2 - 2*resolution, resolution ** 2 - 2*resolution + 1, resolution ** 2 - 2*resolution + resolution])
    faces_1 = faces_1[mask_1]
    
    faces_2 = np.concatenate([np.array([[i + resolution, i, i + resolution + 1]]) for i in range((resolution) ** 2)])
    mask_2 = np.stack(
        [
            True if (i % resolution != (resolution - 1)) and (i + resolution + 1 < resolution**2) else False
            for i in range(resolution**2)
        ]
    )
    faces_2[resolution-2] = np.array([resolution-1, resolution-1 + resolution, resolution-2 + resolution])
    faces_2[resolution ** 2 - 2*resolution] = np.array([resolution ** 2 - resolution + 1, resolution ** 2 - resolution, resolution ** 2 - 2*resolution + 1])
   
    faces_2 = faces_2[mask_2]
    faces = np.concatenate([faces_1, faces_2])

    mask = []
    for tri in faces:
        a, b, c = tri
        pa, pb, pc = points[a], points[b], points[c]
        # if one point of the triangle falls in the upper right diagonal, mask=True
        if num_labels == 2 or num_labels == 1:
            if pa[0] > pa[1] or pb[0] > pb[1] or pc[0] > pc[1]:
                mask.append(False)
            else:
                mask.append(True)
        else:
            grid_size = np.sqrt(num_labels)
            bin_size = 1 / grid_size
            # rescale to 0,1
            pa, pb, pc = (pa + 1) / 2, (pb + 1) / 2, (pc + 1) / 2
            # find bin index
            x_bin = min(max(pa[0] // bin_size, pb[0] // bin_size, pc[0] // bin_size), grid_size - 1)
            y_bin = min(max(pa[1] // bin_size, pb[1] // bin_size, pc[1] // bin_size), grid_size - 1)
            label = x_bin + y_bin * grid_size
            mask.append(label)
    mask = np.stack(mask)

    # print(mask.max())
    if num_labels == 1:
        faces_split = [faces]
    elif num_labels == 2:
        faces_split = [faces[mask], faces[~mask]]
    else:
        faces_split = []
        for label in range(num_labels):
            faces_split.append(faces[mask == label])

    # print(points.shape, faces.shape, mask.shape)
    # return points, faces, None, None
    return points, faces, faces_split, mask


def get_empty_2d_square_mesh(resolution):
    x = np.linspace(-1, 1, resolution)
    bottom = np.stack((x, np.zeros_like(x) - 1), axis=1)
    top = np.stack((x, np.zeros_like(x) + 1), axis=1)
    x = x[1:-1]
    right = np.stack((np.zeros_like(x) + 1, x), axis=1)
    # remove two points to avoid duplicates
    left = np.stack((np.zeros_like(x) - 1, x), axis=1)
    points = np.concatenate([bottom, top, right, left])
    from scipy.spatial import Delaunay

    faces = Delaunay(points).simplices
    return points, faces, np.zeros_like(faces) + 1, left, right, top, bottom


def get_unit_test_target(resolution):
    x = np.linspace(-1, 1, resolution)
    size = 0.2
    bottom = np.stack((x, np.zeros_like(x) - 1), axis=1)
    bottom[3, 1] = bottom[3, 1] + size
    bottom[4] = bottom[3]
    bottom[4, 0] = bottom[1, 0]
    bottom[5] = bottom[4]
    bottom[5, 1] = bottom[3, 1] + 2 * size
    bottom[6] = bottom[5]
    bottom[6, 0] = bottom[7, 0]
    top = bottom + np.array([[0, 2]])
    x = x[1:-1]
    right = np.stack((np.zeros_like(x) + 1, x), axis=1)
    # remove two points to avoid duplicates
    left = np.stack((np.zeros_like(x) - 1, x), axis=1)
    points = np.concatenate([bottom, top, right, left])
    from scipy.spatial import Delaunay

    faces = Delaunay(points).simplices
    return points, faces, np.zeros_like(faces) + 1, left, right, top, bottom

def get_hexagonal_mesh(vertices_per_edge = 65):
    """Generator for coordinates in a hexagon."""
    steps = np.floor(math.log2(vertices_per_edge - 1)).astype(int)
    vertices_per_edge = 2**steps + 1
    assert steps.is_integer(),"the number of vertices per edge should be a power of 2 plus, i.e., 2^n+1 "
    steps = int(steps)
    edge_length = 2 * math.sqrt(3) / 3 
    vertices = []
    vertices.append([0,0])
    triangles = []
    for i in range(6):
        angle = i*2*math.pi/6
        point = [math.cos(angle),math.sin(angle)]
        vertices.append(point)
        cur = i+1
        next = i+2
        if next > 6:
            next = 1
        triangle = [0,cur,next]
        triangles.append(triangle)
    vertices = np.array(vertices)
    triangles = np.array(triangles)
    for step in range(steps):
        vertices, triangles = igl.upsample(vertices, triangles)

    bdry = igl.boundary_loop(triangles)
    for i in range(len(bdry)):
        bdry = np.roll(bdry,1)
        ind = bdry[0]
        if vertices[ind,0] == 1 and vertices[ind,1] ==0: #this is the first vertex of the basic hexagon and cos(0) sin(0)
            break
    else:
        raise Exception("couldn't find the start vertex")
    sides = {}
    for i in range(6):
        start = i*(vertices_per_edge-1)
        end = start+vertices_per_edge
        sides[i] = bdry[start:end]
    sides[5] = np.append(sides[5],sides[0][0])
    from escher.geometry.sanity_checks import check_triangle_orientation
    print("Hexagon created: sanity check of the triangles orientation...")
    check_triangle_orientation(vertices,triangles)
    print("done.")
    return vertices,triangles,sides


def get_hexagonal_mesh_old(vertices_per_edge = 20):
    """Generator for coordinates in a hexagon.
    Deprecated because triangle are not uniform enough"""
    import math
    edge_length = 2 * math.sqrt(3) / 3 
    vertices = []
    x, y = -math.sqrt(3)/3, -1
    vertices.append([x,y])
    for angle in range(0, 360, 60):
        for i in range(1, vertices_per_edge+1):
            x += math.cos(math.radians(angle)) * edge_length / vertices_per_edge
            y += math.sin(math.radians(angle)) * edge_length / vertices_per_edge
            vertices.append([x,y])
    vertices = np.array(vertices)
    # vertices = vertices - np.mean(vertices, axis=0)
    # vertices = vertices / np.max(np.abs(vertices))
    boundary_points = vertices / (2 * math.sqrt(3) / 3)

    import igl 
    from scipy.spatial import Delaunay
    from scipy.spatial.qhull import QhullError

    def generate_interior_points(boundary_points, num_points=100):
        """
        Generate random interior points within the convex hull of the boundary.
        This is a simple method that works well for convex shapes.
        """
        min_x, min_y = np.min(boundary_points, axis=0)
        max_x, max_y = np.max(boundary_points, axis=0)

        points = []
        while len(points) < num_points:
            print(len(points))
            random_point = [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]
            if Delaunay(boundary_points).find_simplex(random_point) >= 0:
                points.append(random_point)

        return np.array(points)

    # Generate interior points
    interior_points = generate_interior_points(boundary_points, 600)  # Generate 50 interior points

    # Combine boundary and interior points
    all_points = np.vstack((boundary_points, interior_points))

    # Perform Delaunay triangulation
    tri = Delaunay(all_points)

    boundary_points = {i: np.arange(i*20, (i+1)*20 + 1) for i in range(6)} 
    return all_points, tri.simplices, boundary_points


if __name__ == "__main__":
    from escher.rendering.render_mesh_matplotlib import render_mesh_matplotlib, render_points_matplotlib

    # Test get_square_mesh
    # points, faces, faces_split, mask = get_2d_square_mesh(50, 2)
    # render_mesh_matplotlib(points, faces, None, ["test.png"])
    # for i, face in enumerate(faces_split):
    #     render_mesh_matplotlib(points, face, None, [f"test{i}.png"])

    # test get_hexagonal_mesh
    all_points, faces, boundary_points = get_hexagonal_mesh()
    render_points_matplotlib(all_points, ["testp.png"])
    render_mesh_matplotlib(all_points, faces, boundary_points, ["test_mesh.png"])
