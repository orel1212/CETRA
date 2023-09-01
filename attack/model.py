from chainer.initializers import LeCunNormal
from chainer.optimizer import GradientClipping
from chainer import links as L, functions as F
from chainerrl.optimizers.rmsprop_async import RMSpropAsync
from chainer.optimizers import Adam
from chainerrl.replay_buffer import EpisodicReplayBuffer
from chainerrl.distribution import SoftmaxDistribution
from chainerrl.action_value import DiscreteActionValue
from chainerrl.links import Sequence
from chainerrl.agents.acer import ACERSeparateModel
from chainerrl.agents import ACER
from chainer import serializers

from config import Config

config_obj = Config()

class CetraAgent:
    def __init__(self, train_mode=True, load_model=False, path_to_load='', input_size=5, output_size=7, hidden_size=32,
                 discount_factor=0.99, opt_alpha=0.99, entropy_beta=0.99, gradient_clipping=40, max_step=50,
                 replay_start=100):
        self.train_mode = train_mode  # training or testing
        self.input_size = input_size  # env observation_space shape(dim [0])
        self.hidden_size = hidden_size
        self.output_size = output_size  # env action_space shape
        self.discount_factor = discount_factor
        self.opt_alpha = opt_alpha
        self.entropy_beta = entropy_beta
        self.gradient_clipping = gradient_clipping
        self.max_step = max_step
        self.replay_start = replay_start
        self.agent = None
        self._create_agent(load_model, path_to_load)

    def _create_agent(self, load_model, path_to_load):
        model = ACERSeparateModel(
            pi=Sequence(
                L.Linear(self.input_size, self.hidden_size),
                F.relu,
                L.Linear(self.hidden_size, self.output_size, initialW=LeCunNormal(1e-3)),
                SoftmaxDistribution,
            ),
            q=Sequence(
                L.Linear(self.input_size, self.hidden_size),
                F.relu,
                L.Linear(self.hidden_size, self.output_size, initialW=LeCunNormal(1e-3)),
                DiscreteActionValue,
            )
        )

        optimizer = RMSpropAsync(lr=config_obj.model['learning_rate'], eps=config_obj.model['eps'], alpha=self.opt_alpha)
        optimizer.setup(model)
        optimizer.add_hook(GradientClipping(self.gradient_clipping))

        # optimizer = Adam(eps=0.02,alpha=self.opt_alpha)
        # optimizer.setup(model)
        # optimizer.add_hook(GradientClipping(self.gradient_clipping))

        if load_model == True:
            model_file = path_to_load / 'model.npz'
            optimizer_file = path_to_load / 'optimizer.npz'
            if model_file.exists() and optimizer_file.exists():
                serializers.load_npz(model_file, model)
                serializers.load_npz(optimizer_file, optimizer)
                print("Successfully loaded model!")
            else:
                print("Not found saved agent model...Exiting!")
                exit()

        created_agent = ACER(
            model=model,  # Model to train
            optimizer=optimizer,  # The optimizer
            gamma=self.discount_factor,  # Reward discount factor
            t_max=self.max_step,  # The model is updated after this many local steps
            replay_buffer=EpisodicReplayBuffer(config_obj.model['replay_buffer_size']),  # The replay buffer
            replay_start_size=self.replay_start,  # Replay buffer won't be used until it has at least this many episodes
            beta=self.entropy_beta,  # Entropy regularization parameter
        )

        self.agent = created_agent

    def set_act_deterministically_flag(self, flag=True):
        self.agent.act_deterministically = flag

    def act_and_train(self, state, reward):
        return self.agent.act_and_train(state, reward=reward)

    def act(self, state):
        return self.agent.act(state)

    def stop_episode_and_train(self, state, reward, done):
        self.agent.stop_episode_and_train(state, reward, done)

    def stop_episode(self):
        self.agent.stop_episode()

    def save(self, path):
        self.agent.save(path)

    def get_statistics(self):
        return self.agent.get_statistics()

    def get_model(self):
        return self.agent.model
