import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

SEED = 2209
TOTAL_TIMESTEPS = 10_000_000  
N_ENVS = 32 
OUTPUT_FILE = "best_policy_2209_improved.npy"


def train_and_extract():
    env = make_vec_env(
        "LunarLander-v3",
        n_envs=N_ENVS,
        seed=SEED,
        env_kwargs={"continuous": False},
    )

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),  
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=SEED,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,  
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        device="auto",
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    state_dict = model.policy.state_dict()

    # Extract weights
    w1 = state_dict["mlp_extractor.policy_net.0.weight"].cpu().numpy().T
    b1 = state_dict["mlp_extractor.policy_net.0.bias"].cpu().numpy()

    w2 = state_dict["mlp_extractor.policy_net.2.weight"].cpu().numpy().T
    b2 = state_dict["mlp_extractor.policy_net.2.bias"].cpu().numpy()

    w3 = state_dict["action_net.weight"].cpu().numpy().T
    b3 = state_dict["action_net.bias"].cpu().numpy()

    flat_params = np.concatenate(
        [
            w1.ravel(),
            b1.ravel(),
            w2.ravel(),
            b2.ravel(),
            w3.ravel(),
            b3.ravel(),
        ]
    ).astype(np.float32)

    np.save(OUTPUT_FILE, flat_params)
    print(f"Saved {flat_params.size} parameters to {OUTPUT_FILE}")


if __name__ == "__main__":
    train_and_extract()