# CellTowerProblem
An exploration of integrating neural networks into a multi-objective optimization problem. /n

This project builds upon a prior problem utilizing the NSGA-II multi-objective optimization algorithm in order to optimize cost and coverage in a simulated region with four generations of cell towers. Each generation of cell tower increases in radius of coverage while simulataneously increasing in cost. The simulated region is approximated to be 25 x 25 miles, with an average of 30 towers to be included. To improve efficiency of this problem I built a neural network using PyTorch to act as a surrogate model that can be passed into the NSGA-II algorithm rather than the simulation itself. This largely improves computational expenses so that the model can be used on a larger scale. /n

Further Work:
To improve upon this project, it would be ideal to implement the following:
- A 3D simulated region rather than 2D
- More precise calculations of cost
- Destructive frequencies that may cancel one another out instead of boost the overall system
- Active learning techniques to improve the training data

