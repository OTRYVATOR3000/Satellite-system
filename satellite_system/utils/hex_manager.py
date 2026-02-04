import h3
from typing import List, Set, Tuple
from .singleton import singleton

@singleton
class HexManager:
  def __init__(self):
      """The class responsible for interacting with the h3 library
      """
      self.cells = {}

  def __get_all_cells(self, res : int) -> Set[str]:
      """Return a collection of H3 cells, all of resolution ''res''

      Args:
          res (int)

      Returns:
          Set[str]
      """
      return h3.uncompact_cells(h3.get_res0_cells(), res)

  def __get_all_centers(self, cells : Set[str]) -> List[Tuple[float, float, str]]:
      """Return the center points of all H3 cells in ''cells''

      Args:
          cells (Set[str])

      Returns:
          List[Tuple[float, float, str]]
      """
      centers = []

      for i in cells:
          centers.append((h3.cell_to_latlng(i)[0], h3.cell_to_latlng(i)[1], i))

      return centers

  def get_total_number(self, res : int) -> int:
      """Return the total number of cells for the given resolution

      Args:
          res (int)

      Returns:
          int
      """
      return h3.get_num_cells(res)

  def get_centers(self, resolution: int) -> List[Tuple[float, float, str]]:
      """Return the center points of all cells of resolution ''resrolution''

      Args:
          resolution (int)

      Raises:
          ValueError: Resolutions greater than 6 are not supported.

      Returns:
          List[Tuple[float, float]]
      """
      if (resolution >= 7):
          raise ValueError("A resolution of no more than 6 is supporting")

      if (resolution in self.cells):
          return self.cells[resolution]

      self.cells[resolution] = self.__get_all_centers(self.__get_all_cells(resolution))

      return self.cells[resolution]


  def get_square(self, cells : List[str]) -> float:
      """Return the total area of cells

      Args:
          cells (List[str])

      Returns:
          float
      """
      sum = 0

      for i in cells:
          sum += h3.cell_area(str(i), 'km^2')
      return sum