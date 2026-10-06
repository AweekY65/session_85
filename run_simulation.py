"""Demo: run the particle filter on a generated trajectory and save
results (trajectory + errors) to local CSV files. No visualization."""

import os

import numpy as np

from particle_filter import ParticleFilter, Simulator, LandmarkMap


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    landmark_map = LandmarkMap.random(16, xlim=(-10, 10), ylim=(-10, 10), seed=7)
    pf = ParticleFilter(
        n_particles=500,
        init_mean=(0.0, 0.0, 0.0),
        init_std=(0.5, 0.5, 0.3),
        motion_noise=(0.05, 0.02),
        resample_method="systematic",
        ess_threshold=0.5,
        seed=42,
    )
    sim = Simulator(landmark_map, range_std=0.3, bearing_std=0.1, seed=1)

    # Generated control sequence: straight, turn, straight.
    controls = [(1.0, 0.0)] * 40 + [(1.0, 0.4)] * 40 + [(1.0, 0.0)] * 40
    res = sim.run(pf, (0.0, 0.0, 0.0), controls=controls, dt=0.1)

    np.savez(
        os.path.join(out_dir, "results.npz"),
        landmarks=landmark_map.landmarks,
        truth=res["truth"],
        estimates=res["estimates"],
        pos_error=res["pos_error"],
        yaw_error=res["yaw_error"],
    )
    pf.save_state(os.path.join(out_dir, "final_state.pkl"))

    print(f"mean pos error : {res['pos_error'].mean():.4f} m")
    print(f"final pos error: {res['pos_error'][-1]:.4f} m")
    print(f"final yaw error: {res['yaw_error'][-1]:.4f} rad")
    print(f"results written to {out_dir}/")


if __name__ == "__main__":
    main()
