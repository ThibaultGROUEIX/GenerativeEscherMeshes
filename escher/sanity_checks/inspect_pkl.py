import pickle
from escher.rendering.render_mesh_matplotlib import render_mesh_matplotlib
import torch
import numpy as np
from omegaconf import OmegaConf


def inspect(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    constraint_data = data["constraint_data"]
    V = data["V"]
    T = data["T"]
    UV = data["UV"]
    bdry = data["bdry"]
    R = data["R"]
    # Get 2x2 matrix determinant
    # np.linalg.norm(R, axis=2)
    w_solver_input = data["w_solver_input"]
    solver = None
    # try:
    #     solver = data["solver"]
    # except:
    cli_conf = OmegaConf.from_cli()
    conf_file = "config.yaml"
    base_conf = OmegaConf.load(path.replace("results.pkl", conf_file))
    from escher.geometry.get_base_mesh import get_2d_square_mesh

    points, faces_npy, faces_split, mask = get_2d_square_mesh(base_conf.MESH_RESOLUTION, num_labels=1)

    import igl

    bdry = igl.boundary_loop(faces_npy)

    from escher.geometry.split_square_boundary import split_square_boundary

    square_sides = split_square_boundary(points, bdry)

    # generate nx2 list of edge pairs (i,j)
    adjacency_list = igl.adjacency_list(faces_npy)
    edge_pairs = []
    for r, i in zip(adjacency_list, range(len(adjacency_list))):
        for j in r:
            if i < j:
                edge_pairs.append((i, j))
    edge_pairs = np.asarray(edge_pairs)

    # the solver itself
    from escher.OTE.core.OTESolver import OTESolver
    from escher.OTE.tilings.Torus import TorusConstraints
    from escher.OTE.tilings.Cylinder import CylinderConstraints

    constraint_data = CylinderConstraints(points, square_sides)

    solver = OTESolver(edge_pairs, points, constraint_data)
    # Get vertices
    with torch.no_grad():
        V_2, _, _ = solver.solve(w_solver_input)
    # assert (V_2 - V).abs().max() < 1e-6, "V_2 and V are not the same!"
    from escher.geometry.sanity_checks import check_triangle_orientation

    # Render
    render_mesh_matplotlib(
        vertices_npy=V_2,
        faces=T,
        square_sides=square_sides,
        fnames=[
            path.replace(".pkl", "_rendered.pdf"),
        ],
    )
    check_triangle_orientation(V_2, T)

    print("Data has been inspected!")


if __name__ == "__main__":
    inspect("sanity_checks/_prompt=Achildrensbookillustrationofatulipamasterpiece_tiling=Cylinder/results.pkl")
