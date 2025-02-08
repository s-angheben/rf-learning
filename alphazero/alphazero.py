import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import pgx
from pydantic import BaseModel
from omegaconf import OmegaConf
from .network import AZNet
#from network import AZNet


devices = jax.local_devices()
num_devices = len(devices)


class Config(BaseModel):
    env_id: pgx.EnvId = "connect_four"
    seed: int = 0
    max_num_iters: int = 800
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 4096
    num_simulations: int = 128
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 16384
    learning_rate: float = 0.0005
    # eval params
    eval_interval: int = 10


conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)

env = pgx.make(config.env_id)


def forward_fn(x, is_eval=False):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
    )
    policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
    return policy_out, value_out


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    # model: params
    # state: embedding
    del rng_key
    model_params, model_state = model

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(
        model_params, model_state, state.observation, is_eval=True
    )
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

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

def play_model(model, state, key):
    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=config.num_channels,
            num_blocks=config.num_layers,
            resnet_v2=config.resnet_v2,
        )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out
    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    model_params, model_state = model

    (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=False
    )

    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    action = logits.argmax(axis=1)

    return action

def play_mcts_model(model, state, key):
    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=config.num_channels,
            num_blocks=config.num_layers,
            resnet_v2=config.resnet_v2,
        )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out
    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    model_params, model_state = model

    (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=False
    )

    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

    po = mctx.gumbel_muzero_policy(
        params=model,
        rng_key=key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=config.num_simulations,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )
    
    logits = po.action_weights

    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    action = logits.argmax(axis=1)

    return action

def run_mcts_model(model, state, key):
    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=config.num_channels,
            num_blocks=config.num_layers,
            resnet_v2=config.resnet_v2,
        )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out
    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    model_params, model_state = model

    (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=False
    )

    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

    po = mctx.gumbel_muzero_policy(
        params=model,
        rng_key=key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=config.num_simulations,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )
    
    return po.action_weights


def play_model_small(model, state, key):
    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=64,
            num_blocks=5,
            resnet_v2=config.resnet_v2,
        )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out
    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    model_params, model_state = model

    (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=False
    )

    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    action = logits.argmax(axis=1)

    return action

def play_mcts_model_small(model, state, key):
    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=64,
            num_blocks=5,
            resnet_v2=config.resnet_v2,
        )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out
    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
        # model: params
        # state: embedding
        del rng_key
        model_params, model_state = model
        
        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)
        
        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        # mask invalid actions
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        
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

    model_params, model_state = model

    (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=False
    )

    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

    po = mctx.gumbel_muzero_policy(
        params=model,
        rng_key=key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=config.num_simulations,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )
    
    logits = po.action_weights

    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    action = logits.argmax(axis=1)

    return action

