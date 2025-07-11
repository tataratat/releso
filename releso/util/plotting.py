"""Internal and auxiliary plotting functions might be removed.

This file contains some plotting functions these are very limited and should
were created for the seminar and master thesis of clemens fricke. Other
plotting function exist but are not part of the library.

This file is not tested due to it being very specific to the mixd/xns use case.
"""

from typing import List

import numpy as np
from matplotlib import pyplot as plt


def get_tricontour_solution(
    width: int,
    height: int,
    dpi: int,
    coordinates: np.ndarray,
    connectivity: np.ndarray,
    solution: np.ndarray,
    sol_len: int,
    limits_min: List[float],
    limits_max: List[float],
) -> np.ndarray:  # pragma: no cover
    """_summary_.

    Args:
        width (int): _description_
        height (int): _description_
        dpi (int): _description_
        coordinates (np.ndarray): _description_
        connectivity (np.ndarray): _description_
        solution (np.ndarray): _description_
        sol_len (int): _description_
        limits_min (List[float]): _description_
        limits_max (List[float]): _description_

    Returns:
        np.ndarray: _description_
    """
    arrays: List[np.ndarray] = []
    for i in range(sol_len):
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        if i == 2:  # pressure is special
            _ = plt.gca().tricontourf(
                coordinates[:, 0],
                coordinates[:, 1],
                np.clip(solution[:, i], limits_min[i], limits_max[i]) - 1,
                triangles=connectivity,
                cmap="Greys",
                vmin=limits_min[i],
                vmax=limits_max[i],
            )
        else:
            _ = plt.gca().tricontourf(
                coordinates[:, 0],
                coordinates[:, 1],
                solution[:, i],
                triangles=connectivity,
                cmap="Greys",
                vmin=limits_min[i],
                vmax=limits_max[i],
            )
        ax = plt.gca()
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.margins(tight=True)
        plt.axis("off")
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        arrays.append(data)
        plt.close()
    arr = np.array(arrays[0])
    for i in range(sol_len - 1):
        arr[:, :, i + 1] = arrays[i + 1][:, :, 0]
    return arr


# def create_open_knot_vector(
#         degrees: List[int], n_cps: List[int]) -> List[List[float]]:
#     """Create an open knot vector according to the given parameters.

#     Args:
#         degrees (List[int]): _description_
#         n_cps (List[int]): _description_

#     Raises:
#         ValueError: _description_

#     Returns:
#         List[List[float]]: _description_
#     """
#     knot_vectors = []

#     for degree, n_cp in zip(degrees, n_cps):
#         n_middle = (degree+n_cp+1)-(2*(1+degree))
#         if n_middle < 0:
#             raise ValueError(
#                 "The n_cp needs to be high enough to actually make an open "
#                 "knot vector. Which is not the case.")
#         start = [0 for _ in range(1+degree)]
#         end = [1 for _ in range(1+degree)]
#         middle = []
#         if n_middle > 0:
#             middle = np.linspace(0, 1, n_middle+2)[1:-1].tolist()
#         start.extend(middle)
#         start.extend(end)
#         knot_vectors.append(start)
#     return knot_vectors


# def create_gif_mp4_delete_images(
#         files: List[str], save_path: str, fps: int = 5,
#         delete_used: bool = True, cut=True):
#     """_summary_.

#     Args:
#         files (List[str]): _description_
#         save_path (str): _description_
#         fps (int, optional): _description_. Defaults to 5.
#         delete_used (bool, optional): _description_. Defaults to True.
#         cut (bool, optional): _description_. Defaults to True.
#     """
#     images = []
#     for file_name in files:
#         if cut:
#             images.append(imageio.imread(file_name)[700:1300, :])
#         else:
#             images.append(imageio.imread(file_name))
#     save_path = pathlib.Path(save_path)
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     imageio.mimsave(save_path, images, fps=fps)
#     imageio.mimsave(save_path.with_suffix(".mp4"), images, fps=fps)

#     if delete_used:
#         for file_name in set(files):
#             pathlib.Path(file_name).unlink()
#         files = []


# def normalize_mesh(mesh: gus.Mesh) -> gus.Mesh:
#     """Normalize the given mesh [0,1].

#     Args:
#         mesh (gus.Mesh): Mesh to normalize.

#     Returns:
#         gus.Mesh: Normalized mesh.
#     """
#     _mesh = deepcopy(mesh)
#     min_: np.ndarray = _mesh.vertices.min(axis=0)
#     _mesh.vertices -= min_
#     max_: np.ndarray = _mesh.vertices.max(axis=0)
#     _mesh.vertices /= max_
#     return _mesh


# # TODO this is gustaf specific
# def get_face_boundaries_and_vertices(
#         mesh: gus.Mesh,
#         optional=False) -> Tuple[List[List[List[float]]], pd.DataFrame]:
#     """_summary_.

#     Args:
#         mesh (gus.Mesh): _description_
#         optional (bool, optional): _description_. Defaults to False.

#     Returns:
#         Tuple[List[List[List[float]]], pd.DataFrame]: _description_
#     """
#     # create vertices dataframe
#     vertice_df = pd.DataFrame(mesh.vertices)

#     # create plot lines for face boundaries
#     face_boundaries = []
#     for face in mesh.faces:
#         # print(face)
#         x = [vertice_df.iloc[node][0] for node in face]
#         y = [vertice_df.iloc[node][1] for node in face]
#         x.append(x[0])
#         y.append(y[0])
#         face_boundaries.append([x, y])
#     if optional:
#         bounds = []
#         for boundary in mesh.edges[mesh.outlines]:
#             x = [vertice_df.iloc[node][0] for node in boundary]
#             y = [vertice_df.iloc[node][1] for node in boundary]
#             x.append(x[0])
#             y.append(y[0])
#             bounds.append([x, y])
#         return face_boundaries, vertice_df, bounds
#     return face_boundaries, vertice_df


# def plot_mesh(
#         mesh: gus.Mesh, save_name: str, no_axis: bool = False,
#         tight: bool = False):
#     """_summary_.

#     Args:
#         mesh (gus.Mesh): _description_
#         save_name (str): _description_
#         no_axis (bool, optional): _description_. Defaults to False.
#         tight (bool, optional): _description_. Defaults to False.
#     """
#     plt.rcParams["figure.figsize"] = (20, 20)
#     plt.rcParams.update({'font.size': 22})

#     # create plotlines for faceb oundaries
#     face_boundaries, vertice_df = get_face_boundaries_and_vertices(mesh)
#     # for face in mesh.faces:
#     #   print(face)
#     #   x = [vertice_df.iloc[node][0] for node in face]
#     #   y = [vertice_df.iloc[node][1] for node in face]
#     #   x.append(x[0])
#     #   y.append(y[0])
#     #   face_boundaries.append([x,y])

#     fig, ax = plt.subplots()
#     for face in face_boundaries:
#         ax.plot(face[0], face[1], c="black")
#     ax.scatter(vertice_df[0], vertice_df[1], c="black", s=10)
#     if no_axis:
#         plt.axis('off')
#     plt.grid(True)
#     if tight:
#         plt.tight_layout()
#     plt.savefig(save_name)
#     plt.close()


# def plot_spline(
#         spline: gus.nurbs.Spline, axis: Optional[plt.Axes] = None,
#         export_path: Optional[str] = None, close: bool = False,
#         control_point_marker: str = "*",
#         control_point_marker_size: int = 120,
#         control_point_color: str = "r", spline_path_color: str = "g",
#         spline_path_alpha: float = 0.2, num_points: int = 100,
#         spline_grid_plot_step: int = 1,
#         lims: List[List[float]] = None) -> Optional[plt.Axes]:
#     """Not Implemented.

#     Args:
#         spline (gus.nurbs.Spline): _description_
#         axis (Optional[plt.Axes], optional): _description_. Defaults to None.
#         export_path (Optional[str], optional): _description_.
#           Defaults to None.
#         close (bool, optional): _description_. Defaults to False.
#         control_point_marker (str, optional): _description_. Defaults to "*".
#         control_point_marker_size (int, optional):
#         _description_. Defaults to 120.
#         control_point_color (str, optional): _description_. Defaults to "r".
#         spline_path_color (str, optional): _description_. Defaults to "g".
#         spline_path_alpha (float, optional): _description_. Defaults to 0.2.
#         num_points (int, optional): _description_. Defaults to 100.
#         spline_grid_plot_step (int, optional): _description_. Defaults to 1.
#         lims (List[List[float]], optional): _description_. Defaults to None.

#     Raises:
#         NotImplemented: _description_

#     Returns:
#         Optional[plt.Axes]: _description_
#     """
#     raise NotImplemented
#     if not axis:
#         # max 2D projection dimension
#         fig, ax = plt.subplots()
#         if lims:
#             if len(lims) > 0:
#                 ax.set_xlim(lims[0])
#                 if len(lims) > 1:
#                     ax.set_ylim(lims[1])
#                     # if len(lims) == 3:
#                     #   ax.set_zlim(lims[1])

#     mins = np.array(spline.knot_vectors).min(axis=1)
#     maxs = np.array(spline.knot_vectors).max(axis=1)
#     lines = []
#     for min_, max_ in zip(mins, maxs):
#         lines.append(np.linspace(min_, max_, num_points))
#     items_ls = []
#     for items in itertools.product(*lines):
#         items_ls.append(items)

#     items_ls = spline.evaluate(items_ls)
#     # plot grid and lines needs evaluation
#     df_items = pd.DataFrame(items_ls)
#     # currently only 2D
#     # horizontal lines
#     for i in range(0, num_points, spline_grid_plot_step):
#         ax.plot(df_items[0][i::num_points], df_items[1][i::num_points],
#                 c=spline_path_color, alpha=spline_path_alpha)
#     # horizontal lines
#     for i in range(0, num_points, spline_grid_plot_step):
#         ax.plot(
#             df_items[0][num_points*i:num_points*(i+1)],
#             df_items[1][num_points * i:num_points*(i+1)],
#             c=spline_path_color, alpha=spline_path_alpha)
#     # ax.scatter(df_items[0], df_items[1])

#     # 1D case control_mesh in spline is differently configured so this has to
#     # be implemented differently
#     if spline.para_dim_ == 1:
#         cp = spline.control_points
#         df_cp = pd.DataFrame(cp)
#         ax.scatter(df_cp[0], df_cp[1], marker=control_point_marker,
#                    c=control_point_color, s=control_point_marker_size)
#         ax.plot(df_cp[0], df_cp[1], "--", c=control_point_color)
#         evaluated_knots = spline.evaluate(
#             [[item] for item in spline.knot_vectors[0]])
#         colors = ["lightgray", "darkgray"]
#         for knot in range(len(spline.knot_vectors[0])-1):
#             ax.axvspan(evaluated_knots[knot][0], evaluated_knots[knot+1]
#                        [0], facecolor=colors[knot % 2], alpha=0.2)
#     # create polygon or cube to show boundary deformations of spline
#     elif spline.para_dim_ > 1:
#         control_mesh = spline.control_mesh_()
#         face_boundaries, vertice_df = get_face_boundaries_and_vertices(
#             control_mesh)
#         b_min = control_mesh.vertices.min(axis=0)
#         b_max = control_mesh.vertices.max(axis=0)
#     # print(b_min)
#         lines = []
#         for i in range(spline.para_dim_):
#             lines.append(np.linspace(b_min[i], b_max[i], 100))
#         control_mesh.boundaries
#         boundary_lines = []
#         for i in range(2**spline.para_dim_):
#             pass
#         min_x_yline = [[b_min[0], y] for y in y_lin]
#         min_y_xline = [[x, b_min[1]] for x in x_lin]
#         max_x_yline = [[b_max[0], y] for y in y_lin]
#         max_y_xline = [[x, b_max[1]] for x in x_lin]

#         boundary = (
#               min_x_yline+max_y_xline+max_x_yline[::-1]+min_y_xline[::-1]
#         )
#         boundary = spline.evaluate(boundary)
#         pol = Polygon(boundary, alpha=0.4)
#         ax.add_patch(pol)
#         for face in face_boundaries:
#             ax.plot(face[0], face[1], c="r")
#         ax.scatter(vertice_df[0], vertice_df[1], c="red", marker="*", s=160)
#     if export_path:
#         pathlib.Path(export_path).parent.mkdir(exist_ok=True, parents=True)
#         plt.savefig(export_path)
#     if close:
#         plt.close()
