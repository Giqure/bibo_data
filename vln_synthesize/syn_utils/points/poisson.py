import numpy as np
import carb

def sampleWithPoissonDisk(min_xy, max_xy, min_dist, num_target, max_attempts=30) -> np.ndarray:
    """Simple Poisson-disk sampling in 2D."""
    points = []
    rng = np.random.default_rng(42)

    for _ in range(num_target * max_attempts):
        if len(points) >= num_target:
            break
        candidate = rng.uniform(min_xy, max_xy)
        if len(points) == 0:
            points.append(candidate)
            continue
        dists = np.linalg.norm(np.array(points) - candidate, axis=1)
        if np.min(dists) >= min_dist:
            points.append(candidate)

    carb.log_info(f"Poisson-disk sampling: {len(points)}/{num_target} points (min_dist={min_dist})")
    return np.array(points)
