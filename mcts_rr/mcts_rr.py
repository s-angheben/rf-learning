import jax
import jax.numpy as jnp
import mctx
import pgx
from pgx._src.types import Array, PRNGKey

seed = 0

env_id: pgx.EnvId = "connect_four"
env = pgx.make(env_id)

rng_key = jax.random.PRNGKey(seed)
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))


def act_randomly(rng: PRNGKey, legal_action_mask: Array) -> Array:
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits=logits, axis=1)


def uniform_prior(state: pgx.State):
    logits = jnp.zeros_like(state.legal_action_mask, dtype=jnp.float32)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    return logits


def random_rollout(state, key):
    starting_player = state.current_player
    batch_size = len(starting_player)

    def cond_fn(loop_state):
        _, state, _ = loop_state
        return ~(state.terminated | state.truncated).all()

    def body_fn(loop_state):
        key, state, rewards = loop_state
        key, subkey = jax.random.split(key)
        action = act_randomly(subkey, state.legal_action_mask)
        state = step(state, action)
        rewards += state.rewards
        return key, state, rewards

    _, _, rewards = jax.lax.while_loop(
        cond_fn, body_fn, (key, state, jnp.zeros_like(state.rewards))
    )
    return rewards[jnp.arange(batch_size), starting_player]


def forward_fn(state: pgx.State, key):
    policy_out = uniform_prior(state)
    value_out = random_rollout(state, key)
    value_out = jnp.array(value_out)
    return policy_out, value_out


def recurrent_fn(params, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    current_player = state.current_player
    state = step(state, action)
    logits, value = forward_fn(state, rng_key)
    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


def run_mcts(num_simulations, state, rng_key: jnp.ndarray) -> jnp.ndarray:
    key1, key2 = jax.random.split(rng_key)

    (logits, value) = forward_fn(state, key1)

    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

    policy_output = mctx.gumbel_muzero_policy(
        params=None,
        rng_key=key2,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=5.0,
    )

    return policy_output

def play(state, key):
    po = run_mcts(32, state, key)

    logits = po.action_weights

    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    action = logits.argmax(axis=1)

    return action
