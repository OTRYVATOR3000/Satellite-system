import math
import pytest
from satellite_system.coordinator import Coordinator
coordinator = Coordinator()

EARTH_RADIUS_KM = 6371.0

def analytic_coverage(satel: int, altitude: float, angle: float) -> float:
  """Analytical assessment of Earth's satellite coverage.

  Args:
      satel (int): number of satellites
      altitude (float): orbit altitude, km
      fov_deg (float): half the viewing angle, degrees

  Returns:
      float: Earth coverage
  """
  R = EARTH_RADIUS_KM

  theta = math.radians(angle)

  betta = math.asin((R + altitude) * math.sin(theta) / R)

  return satel * 0.5 * (1 - math.cos(betta - theta))

def coverage(height, orb_inclin, longitude_asc, count_orb, count_sat, angle, res):
  """
  Comparison of project code coverage and test results
  """
  group_number = coordinator.add_group(f"{height} {orb_inclin} {longitude_asc} {count_orb} {count_sat} 3 0 2024-01-15 14:30:45.123456 {angle}".split())

  coverage_code = coordinator.calculate_coverage(f"{group_number} {res} 2024-01-15 14:30:45.123456".split())

  coverage_analytic = analytic_coverage(count_sat, height, angle)

  error = (abs(coverage_code - coverage_analytic) / coverage_analytic) * 100

  print(f"Coverage (code): {coverage_code * 100:.3g}%")
  print(f"Coverage (test): {coverage_analytic * 100:.3g}%")
  print(f"Error: {error}%")

  assert error <= 5

def test_coverage_1():
  coverage(400, 45, 45, 1, 4, 40, 4) # height (km), orbit inclination, longitude of the ascrding node,
                                     # count of orbits, count of satellites, half of the viewing angle,
                                     # resolution

def test_coverage_2():
  coverage(600, 20, 12, 1, 1, 50, 4)

def test_coverage_3():
  coverage(700, 70, 3, 1, 2, 30, 4)

def test_coverage_4():
  coverage(1000, 54, 124, 1, 5, 40, 4)

def test_coverage_5():
  coverage(800, 25, 350, 2, 2, 40, 4)
