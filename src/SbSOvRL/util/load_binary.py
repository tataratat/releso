import numpy as np
import os
from struct import unpack
from typing import Tuple

def read_mixd_double(filename : str, col : int, element_size: int = 8, element: str = ">d") -> np.ndarray:
  """Read a double array from a mixd file.

  Author: Daniel Wolff (wolff@avt.rwth-aachen.de)

  Args:
      filename (string): filename / location of the mixd file
      col (int): number of columns of the integer array 

  Returns:
      retValues (numpy.ndarray): Double array read from the provided file
  """
  with open(filename,"rb") as fHandle:
    size = os.stat(filename).st_size
    if (size%(col*element_size) != 0):
        raise RuntimeError("Not enough columns in the requested file!")

    rows = size//(element_size*col)
    retValues = np.zeros((rows,col))
    for i in range(rows):
        for j in range(col):
            data = fHandle.read(element_size)
            retValues[i,j] = unpack(element, data)[0]
  return retValues
  # _read_mixd_double()


def load_mixd(mesh_file : str, mesh_dim : int, solution_file : str, ndof : int) -> Tuple[np.ndarray, np.ndarray]:
  """Read in the mesh and the solution of an XNS simulation and store  
  them in the `coordinates` and `solution` attributes

  Author: Daniel Wolff (wolff@avt.rwth-aachen.de)

  Args:
      mesh_file (string): filename / location of the mesh file `mxyz`
      mesh_dim (int): Number of spatial dimensions of the mesh
      solution_file (string): filename / location of the file containing  
          the nodal solutions
      ndof (int): Number of degrees of freedom per node

  Returns:
      coordinates (numpy.ndarray): Array of points from the mixd mesh
      values (numpy.ndarray): Nodal solution values from the mixd file
  """
  coordinates = read_mixd_double(mesh_file, mesh_dim)
  solution = read_mixd_double(solution_file, ndof)
  return coordinates, solution