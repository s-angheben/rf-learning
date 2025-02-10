import datetime
import os
import pickle
import time
import json
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
import pgx
from pgx.experimental import auto_reset
from reinforce_v import config, devices, num_devices, env, forward


# optimizer = optax.adam(learning_rate=config.learning_rate)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(learning_rate=config.learning_rate),
)


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices
    epsilon = 0.1

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2, key3 = jax.random.split(key, 3)
        observation = state.observation

        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )

        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        action = jax.random.categorical(key1, logits, axis=-1)

        # epsilon-greedy
        #action = jax.lax.cond(
        #    jax.random.uniform(key1) < epsilon,
        #    lambda: jax.random.categorical(
        #        key2, state.legal_action_mask.astype(jnp.float32), axis=-1
        #    ),
        #    lambda: jax.random.categorical(key2, logits, axis=-1),
        #)

        actor = state.current_player
        keys = jax.random.split(key3, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, action, keys)
        discount = -1 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)

        return state, SelfplayOutput(
            obs=observation,
            action=action,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


class Sample(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # value_mask = jnp.cumsum(data.terminated, axis=0) < 1
    # value_mask = jnp.cumsum(~value_mask, axis=0) <= 1

    # Compute value target
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        action=data.action,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def loss_fn(model_params, model_state, samples: Sample):
    (logits, value), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )

    # Compute log probabilities of the policy
    log_probs = jax.nn.log_softmax(logits)
    selected_log_probs = jnp.take_along_axis(
        log_probs, samples.action[:, None], axis=-1
    ).squeeze(-1)

    # Compute the advantage: A(s_t, a_t) = G_t - V(s_t)
    advantage = samples.value_tgt - value
    # normalization
    advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 2e-8)

    # policy_loss = selected_log_probs * samples.value_tgt * samples.mask
    policy_loss = selected_log_probs * advantage * samples.mask
    policy_loss = -jnp.mean(policy_loss)

    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    entropy_loss = -jnp.mean(entropy * samples.mask)  # Maximize entropy

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)  # mask if the episode is truncated

    # loss = policy_loss + 0.1 * entropy_loss
    loss = policy_loss + value_loss + 0.1 * entropy_loss

    return loss, (
        model_state,
        policy_loss,
        value_loss,
    )


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss


if __name__ == "__main__":
    # Initialize model and opt_state
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_input = dummy_state.observation
    model = forward.init(jax.random.PRNGKey(0), dummy_input)  # (params, state)
    opt_state = optimizer.init(params=model[0])
    # replicates to all devices
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # Prepare checkpoint dir
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"{config.env_id}_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize logging file
    log_file = os.path.join(ckpt_dir, "training_log.json")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            json.dump([], f)  # Initialize an empty list for logs

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    rng_key = jax.random.PRNGKey(config.seed)
    while True:
        if iteration % config.eval_interval == 0:
            # Store checkpoints
            model_0, opt_state_0 = jax.tree_util.tree_map(
                lambda x: x[0], (model, opt_state)
            )
            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                    "config": config,
                    "rng_key": rng_key,
                    "model": jax.device_get(model_0),
                    "opt_state": jax.device_get(opt_state_0),
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "pgx.__version__": pgx.__version__,
                    "env_id": env.id,
                    "env_version": env.version,
                }
                pickle.dump(dic, f)

        # Print and log to file
        print(log)
        with open(log_file, "r") as f:
            logs = json.load(f)
        logs.append(log)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Selfplay
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(model, keys)
        samples: Sample = compute_loss_input(data)

        # Shuffle samples and make minibatches
        samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, *x.shape[3:])), samples
        )
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )

        # Training
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss = train(
                model, opt_state, minibatch
            )
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        )
