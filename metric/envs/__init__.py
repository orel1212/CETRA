from gym.envs.registration import register

register(
    id='Cetra-v1',
    entry_point='envs:CetraEnv',
)

from envs.cetra_env import CetraEnv
