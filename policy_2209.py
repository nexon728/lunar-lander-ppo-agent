import numpy as np

INPUT_SIZE = 8
HIDDEN_SIZE = 128  
OUTPUT_SIZE = 4

EXPECTED_PARAM_SIZE = (
    (INPUT_SIZE * HIDDEN_SIZE)
    + HIDDEN_SIZE
    + (HIDDEN_SIZE * HIDDEN_SIZE)
    + HIDDEN_SIZE
    + (HIDDEN_SIZE * OUTPUT_SIZE)
    + OUTPUT_SIZE
)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _unpack_params(params: np.ndarray):
    params = np.asarray(params, dtype=np.float32).reshape(-1)
    if params.size != EXPECTED_PARAM_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_PARAM_SIZE} parameters, but received {params.size}."
        )

    idx = 0

    w1 = params[idx : idx + INPUT_SIZE * HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
    idx += INPUT_SIZE * HIDDEN_SIZE
    b1 = params[idx : idx + HIDDEN_SIZE]
    idx += HIDDEN_SIZE

    w2 = params[idx : idx + HIDDEN_SIZE * HIDDEN_SIZE].reshape(HIDDEN_SIZE, HIDDEN_SIZE)
    idx += HIDDEN_SIZE * HIDDEN_SIZE
    b2 = params[idx : idx + HIDDEN_SIZE]
    idx += HIDDEN_SIZE

    w3 = params[idx : idx + HIDDEN_SIZE * OUTPUT_SIZE].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    idx += HIDDEN_SIZE * OUTPUT_SIZE
    b3 = params[idx : idx + OUTPUT_SIZE]

    return w1, b1, w2, b2, w3, b3


def policy_action(params, observation):
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)

    if obs.size != INPUT_SIZE:
        raise ValueError(f"Expected observation of length {INPUT_SIZE}, got {obs.size}")

    w1, b1, w2, b2, w3, b3 = _unpack_params(params)

    a1 = relu(obs @ w1 + b1)
    a2 = relu(a1 @ w2 + b2)
    logits = a2 @ w3 + b3

    return int(np.argmax(logits))