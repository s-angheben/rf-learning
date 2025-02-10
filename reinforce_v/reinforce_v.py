import haiku as hk
import jax
import pgx
from pydantic import BaseModel
from omegaconf import OmegaConf
import jax.numpy as jnp

from .network import AZNet
#from network import AZNet

devices = jax.local_devices()
num_devices = len(devices)


class Config(BaseModel):
    env_id: pgx.EnvId = "connect_four"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    num_channels: int = 64
    num_layers: int = 5
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 50
    # training params
    training_batch_size: int = 2048
    learning_rate: float = 1e-5
    # eval params
    eval_interval: int = 5


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

def play_model(model, state, key):
    del key
    model_params, model_state = model

    (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=False
    )
    
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    action = logits.argmax(axis=1)

    return action

