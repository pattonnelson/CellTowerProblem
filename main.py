import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from customsampling import CustomMixedSampling
from pymoo.termination import get_termination
from Region import Region
from Tower import Tower
import matplotlib.pyplot as plt
import numpy as np
from tower_gnn_model import TowerGNN
from graph_builder import build_tower_graph
import torch


def compute_overlap_penalty(towers):
    penalty = 0.0
    for i in range(len(towers)):
        for j in range(i+1, len(towers)):
            d = np.sqrt((towers[i].x - towers[j].x)**2 + (towers[i].y - towers[j].y)**2)
            min_dist = towers[i].calculate_radius() + towers[j].calculate_radius()
            if d < min_dist:
                penalty += (min_dist - d)**2
    return penalty

# ======= REGION PARAMETERS (optional context) =======
regionparams = {"xl": 0., "xh": 1000., "yl": 0., "yh": 1000., "spacing": (10, 10)}

# ======= LOAD TRAINED GNN MODEL =======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TowerGNN(in_channels=8, hidden_channels=64, out_channels=2).to(device)
model.load_state_dict(torch.load("best_tower_gnn.pt", map_location=device))
model.eval()

# ======= DEFINE OPTIMIZATION PROBLEM =======
class MyProblem(ElementwiseProblem):
    def __init__(self, gnn_model, **kwargs):
        vars = {}
        for i in range(30):
            vars[f"x{i}"] = Real(bounds=(0, 1000))
            vars[f"y{i}"] = Real(bounds=(0, 1000))
            vars[f"g{i}"] = Integer(bounds=(0, 3))
        self.gnn_model = gnn_model
        super().__init__(vars=vars, n_obj=2, n_ieq_constr=0, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        towers = []
        for i in range(30):
            x = float(X[f"x{i}"])
            y = float(X[f"y{i}"])
            g = int(X[f"g{i}"])

            tower = Tower(x, y, g)
            tower.sigma = 5  # or whatever you're using
            towers.append(tower)

        data = build_tower_graph(towers)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long).to(device)
        data = data.to(device)
        with torch.no_grad():
            pred = self.gnn_model(data).cpu().numpy()
            pred_exp = np.expm1(pred)  # Reverses log1p
            overlap_penalty = compute_overlap_penalty(towers)  # see helper below
            pred_exp[0, 1] += overlap_penalty * 1e3  # Add to cost prediction
        out["F"] = pred_exp.tolist()  # [deadzone, cost]

# ======= CONFIGURE ALGORITHM =======
algorithm = MixedVariableGA(pop_size=100, survival=RankAndCrowdingSurvival())


# algorithm = NSGA2(
#     pop_size =  100,
#     n_offsprings = 10,
#     sampling = CustomMixedSampling,
#     eliminate_duplicates = True
# )

problem = MyProblem(gnn_model=model)

termination = get_termination("n_gen",300)

# ======= RUN OPTIMIZATION =======
res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    verbose=True
)

# ======= DISPLAY RESULTS =======
# from pymoo.visualization.scatter import Scatter
# Scatter().add(res.F, facecolor="none", edgecolor="blue").show()

result_dict = res.X[0]  # Extract the actual dictionary from the array

print("Tower Layout:")
for i in range(20):
    x = result_dict[f"x{i}"]
    y = result_dict[f"y{i}"]
    g = result_dict[f"g{i}"]
    print(f"Tower {i+1}: x={x:.1f}, y={y:.1f}, generation={g}")


# Visualize results


# --- Pull optimized values from res.X ---
result_dict = res.X[0]  # Unwrap from array


# === Predict with GNN on the same layout ===
towers = []
for i in range(30):
    x = result_dict[f"x{i}"]
    y = result_dict[f"y{i}"]
    g = result_dict[f"g{i}"]
    tower = Tower(x, y, g)
    tower.sigma = 5
    towers.append(tower)

data = build_tower_graph(towers)
data.batch = torch.zeros(data.x.size(0), dtype=torch.long).to(device)
data = data.to(device)

with torch.no_grad():
    pred = model(data).cpu().numpy()
    pred = np.expm1(pred)


# --- Initialize Region ---
region = Region(xl=0., xh=1000., yl=0., yh=1000., spacing=(10, 10))
# --- Add Towers to Region ---
for i in range(30):
    x = result_dict[f"x{i}"]
    y = result_dict[f"y{i}"]
    g = result_dict[f"g{i}"]
    
    tower = Tower(x, y, g)
    tower.sigma = 5 
    tower.generation = g
    tower.r_threshold = tower.calculate_radius(g)
    tower.calculate_signal_intensity(region)
    region.add_tower(tower)


def evaluate_layout(layout_dict, model, device):
    towers = []
    for i in range(30):
        x = layout_dict[f"x{i}"]
        y = layout_dict[f"y{i}"]
        g = layout_dict[f"g{i}"]
        t = Tower(x, y, g)
        t.sigma = 5
        towers.append(t)

    # --- GNN Prediction ---
    data = build_tower_graph(towers)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long).to(device)
    data = data.to(device)
    with torch.no_grad():
        pred = model(data).cpu().numpy()
        pred = np.expm1(pred)  # Undo log1p

    # --- Simulated Ground Truth ---
    region = Region(xl=0., xh=1000., yl=0., yh=1000., spacing=(10, 10))
    for t in towers:
        t.calculate_signal_intensity(region)
        region.add_tower(t)
    actual_deadzone = region.calculate_deadzone()
    actual_cost = region.compute_cost()

    return pred[0], [actual_deadzone, actual_cost]

# --- Draw tower layout ---
region.draw()

# --- Show intensity heatmap ---
region.show_intensity()

# --- Show 3D intensity surface plot (optional) ---
region.show_intensity3D()

# --- Show deadzone percentage ---
print(f"\nDeadzone: {region.calculate_deadzone():.2f}%")
print(f"\nTotal Cost: {region.compute_cost():,.0f}")

pred_deadzone, pred_cost = pred[0]
print(f"\nGNN Predicted Deadzone: {pred_deadzone:.2f}%")
print(f"\nGNN Predicted Cost:      ${pred_cost:,.0f}")

gnn_preds = []
true_vals = []

for layout in res.X:
    gnn, true = evaluate_layout(layout, model, device)
    gnn_preds.append(gnn)
    true_vals.append(true)

gnn_preds = np.array(gnn_preds)
true_vals = np.array(true_vals)

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(gnn_preds[:, 0], gnn_preds[:, 1], label="GNN Predictions", c='blue', alpha=0.6)
plt.scatter(true_vals[:, 0], true_vals[:, 1], label="Simulated Results", c='green', alpha=0.6)
plt.xlabel("Deadzone (%)")
plt.ylabel("Cost")
plt.title("Pareto Front: GNN vs Simulated")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
