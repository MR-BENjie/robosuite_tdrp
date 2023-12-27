import argparse
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
import robosuite as suite
from rlkit.envs.wrappers import NormalizedBoxEnv
from robosuite.wrappers import GymWrapper
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument(
        '--env',
        type=str,
        default='Lift',
        help='Robosuite env to run on')

    parser.add_argument(
        '--robots',
        nargs="+",
        type=str,
        #default="Panda",
        default=['Panda', 'Panda'],
        help='Robot(s) to run with')

    parser.add_argument(
        '--expl_horizon',
        type=int,
        default=500,
        help='max num of time steps fo reach eval simulation')

    parser.add_argument(
        '--policy_freq',
        type=int,
        default=20,
        help='Policy frequency for environment(Hz)')

    parser.add_argument(
        '--controller',
        type=str,
        default="OSC_POSE",
        help='controller')

    parser.add_argument(
        '--reward_scale',
        type=float,
        default=1.0,
        help='max reward from single environment step'
    )

    parser.add_argument(
        '--hard_reset',
        action="store_true",
        help='If set,use shard resets for this env'
    )

    args = parser.parse_args()
    return args
def get_expl_env_kwargs(args):
    """
    Grabs the robosuite-specific arguments and converts them into an rlkit-compatible dict for exploration env
    """
    env_kwargs = dict(
        env_name=args.env,
        robots=args.robots,
        horizon=args.expl_horizon,
        control_freq=args.policy_freq,
        controller=args.controller,
        reward_scale=args.reward_scale,
        hard_reset=args.hard_reset,
        ignore_done=True,
    )

    # Lastly, return the dict
    return env_kwargs
def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  args = parse_args()
  sys.argv = sys.argv[:-2]

  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])

  import datetime
  now = datetime.datetime.now()
  formatted_date = now.strftime("%Y_%m_%d_%H_%M_%S")

  config = config.update({
      #'logdir': "./log/"+args.env+" "+str(args.robots)+" "+formatted_date,
      'run.train_ratio': 64,
      'run.log_every': 30,  # Seconds
      'batch_size': 16,
      'jax.prealloc': False,
      #'encoder.mlp_keys': '$^',
      #'decoder.mlp_keys': '$^',
      'encoder.mlp_keys': '.*',
      'decoder.mlp_keys': '.*',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
      # 'jax.platform': 'cpu',
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  import crafter
  from embodied.envs import from_gym


  expl_environment_kwargs = get_expl_env_kwargs(args)
  controller = expl_environment_kwargs.pop("controller")
  if controller in set(ALL_CONTROLLERS):
      # This is a default controller
      controller_config = load_controller_config(default_controller=controller)
  else:
      # This is a string to the custom controller
      controller_config = load_controller_config(custom_fpath=controller)
  # Create robosuite env and append to our list
  suite_env = suite.make(**expl_environment_kwargs,
                         has_renderer=False,
                         has_offscreen_renderer=False,
                         use_object_obs=True,
                         use_camera_obs=False,
                         reward_shaping=True,
                         controller_configs=controller_config,
                         )
  # Create robosuite envs
  env = NormalizedBoxEnv(GymWrapper(suite_env))

  env = from_gym.FromGym(env, obs_key='vector')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  main()
