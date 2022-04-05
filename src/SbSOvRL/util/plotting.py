from typing import List, Optional, Tuple
import gustav as gus
import imageio as io
import pathlib
from numpy.core.function_base import linspace
from pandas.core.frame import DataFrame
# import vedo
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import imageio
import pandas as pd
from matplotlib.patches import Polygon
import itertools

# def get_axes(spline: gus.nurbs.Spline, limits: Optional[List[List[float]]] = None) -> vedo.Axes:
#   controll_points = spline.control_points
#   n_dim = len(controll_points[0])
#   # print(len(limits))
#   # print(len(limits[0]))
#   if limits is None:
#     limits = [[0,0] for i in range(n_dim)]
#     for point in controll_points:
#       for idx, coordinate in enumerate(point):
#         if coordinate < limits[idx][0]:
#           limits[idx][0] = coordinate
#         if coordinate > limits[idx][1]:
#           limits[idx][1] = coordinate

  
#   items = {}
#   for idx, span in enumerate(limits):
#     if idx == 0:
#       items["xrange"] = span
#     elif idx == 1:
#       items["yrange"] = span
#     elif idx == 2:
#       items["zrange"] = span
#   return vedo.Axes(**items, xyGrid=False, yzGrid=False, zxGrid=False,xtitle=" ",ytitle=" ",ztitle=" ")


def get_tricontour_solution(width: int, height: int, dpi: int, coordinates: np.ndarray, connectivity: np.ndarray, solution: np.ndarray, sol_len: int, limits_min: List[float], limits_max: List[float], ) -> np.ndarray:
  """_summary_

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
      fig = plt.figure(figsize=(width,height), dpi=dpi)
      if i == 2:
          mappable = plt.gca().tricontourf(coordinates[:,0],coordinates[:,1],np.clip(solution[:,i], limits_min[i], limits_max[i])-1, triangles=connectivity, cmap="Greys", vmin=limits_min[i], vmax=limits_max[i])
      else:
          mappable = plt.gca().tricontourf(coordinates[:,0],coordinates[:,1],solution[:,i], triangles=connectivity, cmap="Greys", vmin=limits_min[i], vmax=limits_max[i])
      ax=plt.gca()
      ax.set_xlim((0,1))
      ax.set_ylim((0,1))
      ax.margins(tight=True)
      plt.axis('off')
      plt.tight_layout(pad=0, w_pad=0, h_pad=0)
      fig.canvas.draw()
      # Now we can save it to a numpy array.
      data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      arrays.append(data)
      plt.close()
  arr = np.array(arrays[0])
  for i in range(sol_len-1):
      arr[:,:,i+1] = arrays[i+1][:,:,0]
  return arr

def create_open_knot_vector(degrees:List[int], n_cps:List[int])->List[List[float]]:
  knot_vectors = []

  for degree, n_cp in zip(degrees,n_cps):
    n_middle=(degree+n_cp+1)-(2*(1+degree))
    if n_middle < 0:
      raise ValueError("The n_cp needs to be high enough to actually make an open knot vector. Which is not the case.")
    start = [0 for _ in range(1+degree)]
    end = [1 for _ in range(1+degree)]
    middle = []
    if n_middle>0:
      middle = np.linspace(0,1,n_middle+2)[1:-1].tolist()
    start.extend(middle)
    start.extend(end)
    knot_vectors.append(start)
  return knot_vectors

def create_gif_mp4_delete_images(files: List[str], save_path: str, fps: int = 5, delete_used: bool = True, cut=True):
  images = []
  for file_name in files:
    if cut:
      images.append(imageio.imread(file_name)[700:1300,:])
    else:
      images.append(imageio.imread(file_name))
  save_path = pathlib.Path(save_path)
  save_path.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(save_path, images, fps=fps)
  imageio.mimsave(save_path.with_suffix(".mp4"), images)

  if delete_used:
    for file_name in set(files):
      pathlib.Path(file_name).unlink()
    files=[]


# def create_gif_side_spline():
#   d_images = []
#   for d in [1,2,3,4,5]:
#     controll_points = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 1]]
#     degrees = [d]
#     kn_v = create_open_knot_vector(degrees,len(controll_points))
#     basic_spline = gus.BSpline(degrees=degrees, knot_vectors=kn_v, control_points=controll_points)
#     ax = get_axes(basic_spline, limits=[[0,6],[0,1.5]])
#     asd = basic_spline.show(offscreen=True)
#     p_a  = vedo.show(asd, interactive=False, offscreen=True, size=[2000,2000],screensize=[2000,2000], axes = ax)
#     p_a.screenshot(f"side_b_spline_{d}.png")
#     d_images.append(f"side_b_spline_{d}.png")


#     image_files = []
#     controll_points[3][1] = 0
#     for i in linspace(0,1,20):
#       controll_points[6][1] = i
#       basic_spline = gus.BSpline(degrees=degrees, knot_vectors=kn_v, control_points=controll_points)
#       asd = basic_spline.show(offscreen=True)
#       p_a  = vedo.show(asd, interactive=False, offscreen=True, size=[2000,500], axes=ax)
#       image_files.append(f"img/basic_b_spline_{i}.png")
#       p_a.screenshot(image_files[-1])

#     create_gif_mp4_delete_images(files=image_files, save_path=f'output/b_spline_single_node_degree_{d}.gif')

#   create_gif_mp4_delete_images(files=d_images, save_path=f'output/b_spline_single_node_degrees.gif',fps=1,delete_used=False)

def normalize_mesh(mesh: gus.Mesh) -> gus.Mesh:
  _mesh = deepcopy(mesh)
  min_:np.ndarray = _mesh.vertices.min(axis=0)
  _mesh.vertices -= min_
  max_:np.ndarray = _mesh.vertices.max(axis=0)
  _mesh.vertices /= max_
  return _mesh


def get_face_boundaries_and_vertices(mesh: gus.Mesh) -> Tuple[List[List[List[float]]], pd.DataFrame]:
  # create vertice dataframe
  vertice_df = pd.DataFrame(mesh.vertices)

  # create plotlines for faceb oundaries
  face_boundaries = []
  for face in mesh.faces:
    # print(face)
    x = [vertice_df.iloc[node][0] for node in face]
    y = [vertice_df.iloc[node][1] for node in face]
    x.append(x[0])
    y.append(y[0])
    face_boundaries.append([x,y])

  return face_boundaries, vertice_df

def plot_mesh(mesh: gus.Mesh, save_name:str):
  plt.rcParams["figure.figsize"] = (20,20)
  plt.rcParams.update({'font.size':22})

  # create plotlines for faceb oundaries
  face_boundaries, vertice_df = get_face_boundaries_and_vertices(mesh)
  # for face in mesh.faces:
  #   print(face)
  #   x = [vertice_df.iloc[node][0] for node in face]
  #   y = [vertice_df.iloc[node][1] for node in face]
  #   x.append(x[0])
  #   y.append(y[0])
  #   face_boundaries.append([x,y])

  fig, ax = plt.subplots()
  for face in face_boundaries:
    ax.plot(face[0],face[1],c="black")
  ax.scatter(vertice_df[0], vertice_df[1], c="black", s=10)
  ax.grid(True)
  plt.savefig(save_name)
  plt.close()

def plot_spline(spline: gus.nurbs.Spline, axis: Optional[plt.Axes] = None, export_path: Optional[str] = None, close: bool = False, control_point_marker: str = "*", control_point_marker_size: int = 120,  control_point_color: str ="r", spline_path_color: str = "g", spline_path_alpha: float =0.2, num_points: int = 100, spline_grid_plot_step: int = 1, lims: List[List[float]] = None) -> Optional[plt.Axes]:
  raise NotImplemented
  if not axis:
    # max 2D projection dimension
    fig, ax = plt.subplots()
    if lims:
      if len(lims) > 0:
        ax.set_xlim(lims[0])
        if len(lims) > 1:
          ax.set_ylim(lims[1])
          # if len(lims) == 3:
          #   ax.set_zlim(lims[1])

  mins = np.array(spline.knot_vectors).min(axis=1)
  maxs = np.array(spline.knot_vectors).max(axis=1)
  lines=[]
  for min_, max_ in zip(mins, maxs):
      lines.append(np.linspace(min_,max_,num_points))
  items_ls = []
  for items in itertools.product(*lines):
      items_ls.append(items)

  items_ls = spline.evaluate(items_ls)
  # plot grid and lines needs evaluation 
  df_items = pd.DataFrame(items_ls)
  # currently only 2D
  # horizontal lines
  for i in range(0,num_points, spline_grid_plot_step):
      ax.plot(df_items[0][i::num_points], df_items[1][i::num_points], c=spline_path_color, alpha=spline_path_alpha)
  # horizontal lines
  for i in range(0,num_points, spline_grid_plot_step):
      ax.plot(df_items[0][num_points*i:num_points*(i+1)], df_items[1][num_points*i:num_points*(i+1)], c=spline_path_color, alpha=spline_path_alpha)
  # ax.scatter(df_items[0], df_items[1])
  
  if spline.para_dim_ == 1: # 1D case control_mesh in spline is differently condigured so this has to be implemented differently
    cp = spline.control_points
    df_cp = pd.DataFrame(cp)
    ax.scatter(df_cp[0], df_cp[1], marker=control_point_marker, c=control_point_color, s=control_point_marker_size)
    ax.plot(df_cp[0], df_cp[1], "--", c=control_point_color)
    evaluated_knots = spline.evaluate([[item] for item in spline.knot_vectors[0]])
    colors = ["lightgray", "darkgray"]
    for knot in range(len(spline.knot_vectors[0])-1):
      ax.axvspan(evaluated_knots[knot][0], evaluated_knots[knot+1][0], facecolor=colors[knot%2], alpha=0.2)
  elif spline.para_dim_ > 1: # create polygon or cube to show boundary deformations of spline
    control_mesh = spline.control_mesh_()
    face_boundaries, vertice_df = get_face_boundaries_and_vertices(control_mesh)
    b_min = control_mesh.vertices.min(axis=0)
    b_max = control_mesh.vertices.max(axis=0)
  # print(b_min)
    lines = []
    for i in range(spline.para_dim_):
      lines.append(np.linspace(b_min[i], b_max[i], 100))
    control_mesh.boundaries
    boundary_lines = []
    for i in range(2**spline.para_dim_):
      pass
    min_x_yline = [[b_min[0],y] for y in y_lin]
    min_y_xline = [[x,b_min[1]] for x in x_lin]
    max_x_yline = [[b_max[0],y] for y in y_lin]
    max_y_xline = [[x,b_max[1]] for x in x_lin]

    boundary = min_x_yline+max_y_xline+max_x_yline[::-1]+min_y_xline[::-1]
    boundary = spline.evaluate(boundary)
    pol = Polygon(boundary,alpha=0.4)
    ax.add_patch(pol)
    for face in face_boundaries:
      ax.plot(face[0],face[1],c="r")
    ax.scatter(vertice_df[0], vertice_df[1], c="red", marker="*", s=160)
  if export_path:
    pathlib.Path(export_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(export_path)
  if close:
    plt.close()

def create_gif_ffd():
  plt.rcParams["figure.figsize"] = (20,20)
  plt.rcParams.update({'font.size':22})
  _mesh = gus.load_mixd(2, mien="../../04-Data/ViscousFlowTJunction/mesh/coarse/mien", mxyz="../../04-Data/ViscousFlowTJunction/mesh/coarse/mxyz", quad=True)
  b_degree = [2,2]
  b_cp= [ [0,0], [0.5,0], [1,0], [0,0.5], [0.5,0.5], [1,0.5], [0,1], [0.5,1], [1,1]]
  b_kv = create_open_knot_vector(b_degree, [3,3])
  # basic_spline = gus.BSpline(degrees=b_degree, knot_vectors=b_kv, control_points=b_cp)
  # ax = get_axes(basic_spline, limits=[[0,6],[0,1.5]])
  # asd = basic_spline.show(offscreen=True)
  # p_a  = vedo.show(asd, interactive=False, offscreen=True, size=[2000,2000],screensize=[2000,2000], axes = ax)
  # p_a.screenshot(f"side_b_spline_{d}.png")
  # d_images.append(f"side_b_spline_{d}.png")


  image_files = []
  spline_files = []
  # down = np.linspace(0.5,0.2,5).tolist()
  # up = np.linspace(0.5,0.8,5).tolist()
  # coordinate = [4,1]
  # line = down[:-1]+down[::-1]+up[1:-1]+up[::-1]

  up = np.linspace(0,0.4,20).tolist()
  line = up[:-1]+up[::-1]
  coordinate = [1,1]

  for idx,i in enumerate(line):
    b_cp[coordinate[0]][coordinate[1]] = i
    b_spline = gus.BSpline(b_degree, b_kv, b_cp)
    deformed_mesh = normalize_mesh(_mesh)
    deformed_mesh.vertices = b_spline.evaluate(deformed_mesh.vertices)
    face_boundaries, vertice_df = get_face_boundaries_and_vertices(b_spline.control_mesh_())
    def_face_boundaries, def_vertice_df = get_face_boundaries_and_vertices(deformed_mesh)
    b_min = b_spline.control_mesh_().vertices.min(axis=0)
    b_max = b_spline.control_mesh_().vertices.max(axis=0)
    # print(b_min)

    x_lin = np.linspace(b_min[0], b_max[0], 100)
    y_lin = np.linspace(b_min[1], b_max[1], 100)
    min_x_yline = [[b_min[0],y] for y in y_lin]
    min_y_xline = [[x,b_min[1]] for x in x_lin]
    max_x_yline = [[b_max[0],y] for y in y_lin]
    max_y_xline = [[x,b_max[1]] for x in x_lin]

    boundary = min_x_yline+max_y_xline+max_x_yline[::-1]+min_y_xline[::-1]
    boundary = b_spline.evaluate(boundary)
    pol = Polygon(boundary,alpha=0.4)


    fig, ax = plt.subplots()
    for face in face_boundaries:
      ax.plot(face[0],face[1],c="r")
    ax.scatter(vertice_df[0], vertice_df[1], c="red", marker="*", s=160)
    spline_files.append(f"img/spline_{idx}.png")
    plt.savefig(spline_files[-1])
    for face in def_face_boundaries:
      ax.plot(face[0],face[1],c="black")
    ax.scatter(def_vertice_df[0], def_vertice_df[1], c="black", marker="*", s=80)
    ax.add_patch(pol)
    image_files.append(f"img/ffd_{idx}.png")
    plt.savefig(image_files[-1])
    plt.close()

  create_gif_mp4_delete_images(files=image_files, save_path=f'output/ffd.gif',cut=False)
  create_gif_mp4_delete_images(files=spline_files, save_path=f'output/spline.gif',cut=False)


if __name__ == "__main__":
  pass