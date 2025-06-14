import matplotlib.pyplot as plt
import math
import numpy as np

class Tower():
    """_summary_ : The class tower defines the location of the tower and the radius of the tower.
       _author_  : Patton Nelson & Karthik Suresh
    """
    def __init__(self, x, y, generation,r=-1.0,region=None,sigma = 5):
        self.x = x # ith position in a grid
        self.y = y # jth position in a grid
        self.generation = generation # the generation of the tower
        self.r_threshold = r
        self.region = region
        self.intensity = None
        self.sigma = sigma
        self.cost = max(100_000*(generation+1), 100_000*10**(generation-1)) if self.generation > 0 else 0.
        #self.cost = 100_000**(generation+1)
  
    def calculate_radius(self,generation = None):
        if generation == None:
            generation = self.generation
        sigma_val = self.sigma * (10 + generation * 5)
        #sigma_val = self.sigma * (10 * max(generation, 10e-6))
        return sigma_val
    
    def calculate_signal_intensity(self, region):
    # Create a meshgrid for the x and y coordinates
        if self.intensity is None:
            self.region = region
        x, y = np.meshgrid(np.arange(region.xl, region.xh, region.grid_spacing[0]), 
                        np.arange(region.yl, region.yh, region.grid_spacing[1]))
        if self.sigma is None:
          self.sigma = min(0.01*(region.xh-region.xl), 0.01*(region.yh-region.yl))
        # Calculate the distance from the tower to each cell in the grid
        distance = (x - self.x) ** 2 + (y - self.y) ** 2

        # Calculate the signal intensity for each cell
        if (self.generation < 0):
          self.intensity = 0
        else:
          intensity = np.exp(-1 * (distance / (self.sigma * (10 + self.generation * 5)) ** 2))
          #intensity = np.exp(-1 * (distance / (self.sigma * (10 * max(self.generation, 10e-6))) ** 2))
          self.intensity = intensity


    def draw_tower(self):
        plt.scatter(self.x, self.y, c='grey')
        coverage = plt.Circle((self.x, self.y), self.r_threshold, color='red', fill=False)
        plt.gca().add_artist(coverage)
    
    def show_tower_intensity(self):
        plt.imshow(self.intensity.T, origin='lower', cmap='viridis')
        plt.colorbar(label='Signal Intensity')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Signal Intensity Map for Tower {self.generation}')