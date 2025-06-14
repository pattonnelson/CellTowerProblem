import numpy as np
import math
from Tower import Tower
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
class Region():
    def __init__(self, xl: float, xh: float, 
                 yl: float, yh: float, 
                 spacing:Tuple[float, float],
                 **kwargs
                 ):
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh
        self.grid_spacing = spacing
        self.grid_size = (int((xh - xl) / self.grid_spacing[0]), 
                              int((yh - yl) / self.grid_spacing[1])
                          )
        self.towers = []
        self.kwargs = kwargs
        self.XYGrid = np.zeros((self.grid_size[0], self.grid_size[1]), dtype = self.kwargs.get('dtype', np.float32))
    
    def XYtoIJ(self, x, y) -> Tuple[int, int]:
        i = int((x - self.xl) / self.grid_spacing[0])
        j = int((y - self.yl) / self.grid_spacing[1])
        return (i, j)
    
    def IJtoXY(self, i, j) -> Tuple[float, float]:
        x = self.xl + i * self.grid_spacing[0]
        y = self.yl + j * self.grid_spacing[1]
        return (x, y)
    
    def getTowers(self) -> List[Tower]:
        return self.towers
    
    def update_signal_intensity(self, tower, AddOrRemove = "+"):
    # Create a meshgrid for the x and y coordinates
        if (AddOrRemove == "+"):
          self.XYGrid += tower.intensity
        else:
          self.XYGrid -= tower.intensity
    
    def remove_tower(self, tower):
        self.towers.remove(tower)
        self.update_signal_intensity(tower, "-")
    
    def add_tower(self, tower):
        self.towers.append(tower)
        #update XYGrid
        self.update_signal_intensity(tower, "+")

    def draw(self):
        #self.figure.clear()
        plt.xlim(self.xl, self.xh)
        plt.ylim(self.yl, self.yh)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Cell Tower Placement')

        for tower in self.towers:
            tower.r_threshold = tower.calculate_radius(tower.generation)
            tower.draw_tower()

        plt.grid(True, which='both', linestyle='--')

        plt.show()
    def show_intensity(self):
      plt.imshow(self.XYGrid.T, origin='lower', cmap='viridis', vmin = 0.)
      plt.colorbar(label='Signal Intensity')
      plt.xlabel('X')
      plt.ylabel('Y')
      plt.title('Signal Intensity Map')
      plt.show()
    
    def show_intensity3D(self):
      xx = np.arange(self.xl, self.xh, self.grid_spacing[0])
      yy = np.arange(self.yl, self.yh, self.grid_spacing[1])
      X, Y = np.meshgrid(xx, yy)
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.plot_surface(X, Y, self.XYGrid, cmap='viridis')
      ax.set_xlim(self.xl, self.xh)
      ax.set_ylim(self.yl, self.yh)
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Signal Intensity')
      plt.show()


    def calculate_deadzone(self):
      area_region = self.XYGrid.size #((self.xh - self.xl)/self.grid_size[0]) * (self.yh - self.yl / self.grid_size[1])
      area_deadzone = (np.count_nonzero(self.XYGrid < 1/math.e))/area_region
      return area_deadzone*100.0
    
    def compute_cost(self):
      return sum([tower.cost for tower in self.towers])
    
    # def calculate_deadzone(self):
    #   area_region = self.XYGrid.size #((self.xh - self.xl)/self.grid_size[0]) * (self.yh - self.yl / self.grid_size[1])
    #   area_deadzone = (np.count_nonzero(self.XYGrid > 1/math.e))/area_region
    #   return area_deadzone*100.0
    
    