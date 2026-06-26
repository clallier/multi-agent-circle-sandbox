import os
import sys
import time
import subprocess
import tensorflow as tf
from PIL import Image
import cv2
import concurrent.futures

tf.compat.v1.disable_eager_execution()

# Add the experiments directory to the path so we can import train.py
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../code/maddpg/experiments")
    )
)

import train as maddpg_train
import maddpg.common.tf_util as U


# Ensure video and gif directories exist
os.makedirs("docs/videos", exist_ok=True)
os.makedirs("docs/gifs", exist_ok=True)

# Scenario configs: (scenario_id, episodes)
SCENARIOS = [
    (1, 5000),
    (2, 5000),
    (3, 5000),
    (4, 5000),
    (5, 25000),
    (6, 10000),
    (7, 20000),
    (8, 100000),
    (9, 100000),
    (10, 100000),
    (11, 20000),
]


class MockArgs:
    """Mock argument list mimicking train.py argument parser."""

    def __init__(self, scenario_name, load_dir=None):
        self.scenario = scenario_name
        self.max_episode_len = 100
        self.num_episodes = 5000
        self.num_adversaries = 0
        self.good_policy = "maddpg"
        self.adv_policy = "maddpg"
        self.lr = 1e-2
        self.gamma = 0.95
        self.batch_size = 1024
        self.num_units = 64
        self.exp_name = None
        self.save_dir = f"./test_{scenario_name}/"
        self.save_rate = 200
        self.load_dir = load_dir if load_dir else f"./test_{scenario_name}/best_model/"
        self.restore = True
        self.display = True


def train_scenario(scenario_id, episodes):
    """Invokes the training script to retrain the scenario policy."""
    scenario_name = f"circle_sandbox_{scenario_id}"
    print(f"==========================================")
    print(f"Retraining {scenario_name} for {episodes} episodes...")
    print(f"==========================================")

    # We specify KERAS_HOME environment variable to redirect caches
    env = os.environ.copy()
    env["KERAS_HOME"] = "./.keras"

    cmd = [
        "uv",
        "run",
        "python",
        "code/maddpg/experiments/train.py",
        "--scenario",
        scenario_name,
        "--max-episode-len",
        "100",
        "--num-episodes",
        str(episodes),
        "--save-rate",
        "200",
        "--save-dir",
        f"./test_{scenario_name}/",
    ]
    subprocess.run(cmd, env=env, check=True)


def record_scenario(scenario_id):
    """Loads the trained policy, plays 3 episodes, and records AVI + GIF."""
    scenario_name = f"circle_sandbox_{scenario_id}"
    print(f"Recording {scenario_name}...")

    # Reset default TF graph for clean state
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as session:
        arglist = MockArgs(scenario_name)
        env = maddpg_train.make_env(arglist.scenario, arglist)

        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = maddpg_train.get_trainers(env, 0, obs_shape_n, arglist)

        U.initialize()
        U.load_state(arglist.load_dir)

        frames = []

        # Run 3 episodes (each episode is 100 steps)
        for episode in range(3):
            obs_n = env.reset()
            for step in range(100):
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                obs_n, rew_n, done_n, info_n = env.step(action_n)

                # Render RGB frame
                frame_list = env.render(mode="rgb_array")
                if frame_list and len(frame_list) > 0:
                    frames.append(frame_list[0])

                time.sleep(0.01)

        env.close()

    if len(frames) == 0:
        print(f"Warning: No frames recorded for {scenario_name}!")
        return

    # 1. Save as AVI video (using OpenCV VideoWriter)
    height, width, _ = frames[0].shape
    video_path = f"docs/videos/experiment_{scenario_id}.avi"
    # MJPG codec works well and is cross-platform
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
    for frame in frames:
        # OpenCV expects BGR
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)
    video_writer.release()
    print(f"Saved AVI video to {video_path}")

    # 2. Save as GIF (using Pillow)
    gif_path = f"docs/gifs/experiment_{scenario_id}.gif"
    pil_frames = [Image.fromarray(f) for f in frames]
    # To keep GIF sizes reasonable, we can save every 2nd frame or keep full sequence
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,  # 100ms per frame = 10 FPS
        loop=0,
    )
    print(f"Saved GIF to {gif_path}")


def main():
    to_train = []
    for sid, eps in SCENARIOS:
        avi_exists = os.path.exists(f"docs/videos/experiment_{sid}.avi")
        gif_exists = os.path.exists(f"docs/gifs/experiment_{sid}.gif")
        if avi_exists and gif_exists:
            print(f"Scenario {sid} already trained and recorded. Skipping.")
        else:
            to_train.append((sid, eps))

    if not to_train:
        print("All remaining scenarios are already completed!")
        return

    print(
        f"Starting parallel training for scenarios: {[sid for sid, _ in to_train]}..."
    )

    trained_sids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(to_train)) as executor:
        futures = {
            executor.submit(train_scenario, sid, eps): sid for sid, eps in to_train
        }
        for future in concurrent.futures.as_completed(futures):
            sid = futures[future]
            try:
                future.result()
                print(f"Scenario {sid} training completed successfully.")
                trained_sids.append(sid)
            except Exception as exc:
                print(f"Scenario {sid} training generated an exception: {exc}")
                sys.exit(1)

    # Record them sequentially in the main thread to avoid TensorFlow/OpenGL thread safety issues
    print("Recording completed scenarios sequentially...")
    for sid in trained_sids:
        try:
            record_scenario(sid)
        except Exception as exc:
            print(f"Scenario {sid} recording generated an exception: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    # Redirect keras cache
    os.environ["KERAS_HOME"] = "./.keras"
    main()
