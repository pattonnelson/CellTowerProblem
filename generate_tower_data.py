import numpy as np
import random
import math
import json
from tqdm import tqdm
from Tower import Tower
from Region import Region

# === Parameters ===
NUM_SAMPLES = 5000
TOWERS_PER_SAMPLE = 30
REGION_PARAMS = {"xl": 0., "xh": 1000, "yl": 0., "yh": 1000., "spacing": (10, 10)}
OUTPUT_FILE = "tower_training_data.json"

def is_valid_point(new_point, towers, min_distance):
    for tower in towers:
        if math.sqrt((new_point[0] - tower.x) ** 2 + (new_point[1] - tower.y) ** 2) < min_distance:
            return False
    return True


def generate_non_overlapping_towers(num_towers, region_bounds, sigma_func = 5):
    xl, xh, yl, yh = region_bounds
    towers = []
    attempts = 0
    max_attempts = num_towers * 100

    while len(towers) < num_towers and attempts < max_attempts:
        x = np.random.uniform(xl, xh)
        y = np.random.uniform(yl, yh)
        gen = np.random.randint(0, 4)

        tmp_tower = Tower(x, y, gen)
        tmp_tower.sigma = sigma_func
        radius = tmp_tower.calculate_radius(generation=gen)

        if is_valid_point((x, y), towers, 2 * radius):
            towers.append(tmp_tower)
        attempts += 1

    return towers


def build_dataset(num_samples):
    samples = []
    for _ in tqdm(range(num_samples), desc="Generating samples"):
        region = Region(**REGION_PARAMS)
        sigma_func = 5

        towers = generate_non_overlapping_towers(
            TOWERS_PER_SAMPLE,
            (REGION_PARAMS['xl'], REGION_PARAMS['xh'], REGION_PARAMS['yl'], REGION_PARAMS['yh']), 
            sigma_func)
        
        tower_records = []
        for tower in towers:
            tower.sigma = sigma_func
            tower.calculate_signal_intensity(region)
            region.add_tower(tower)
            tower_records.append((tower.x, tower.y, tower.generation))

        deadzone = region.calculate_deadzone()
        cost = region.compute_cost()
        samples.append({
            "towers": tower_records,
            "deadzone": deadzone,
            "cost": cost
        })
    return samples

if __name__ == "__main__":
    dataset = build_dataset(NUM_SAMPLES)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f)
    print(f"\nSaved {len(dataset)} samples to {OUTPUT_FILE}")
