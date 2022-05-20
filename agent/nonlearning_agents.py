import json
from collections import defaultdict
import os.path as osp
from socket import IP_ADD_MEMBERSHIP
import numpy as np
from habitat import Env, logger
from habitat.config.default import Config
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import tqdm, trange
import json 
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.nav.nav import DistanceToGoal, Success
import gzip
from PIL import Image

def draw_observation(dump_dir, rgb, depth):
    rgb_img = Image.fromarray(rgb)
    # depth_img = Image.fromarray(depth.squeeze(), mode='L')
    rgb_img.save(dump_dir+'_rgb.jpg')
    # depth_img.save(dump_dir+'_depth.jpg')

def evaluate_agent(config: Config) -> None:
    split = config.EVAL.SPLIT
    data_path = config.EVAL.DATA_PATH
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = 960
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = 1280
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.DATA_PATH = data_path
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    config.freeze()

    num_nan = 0

    env = Env(config=config.TASK_CONFIG)

    agent = ActionSimAgent(config.EVAL.NONLEARNING, env, split)

    stats = defaultdict(float)
    num_episodes = min(config.EVAL.EPISODE_COUNT, len(env.episodes))
    
    eval_dict = {} # record is_success of each episode
    for _ in trange(num_episodes):
        obs = env.reset()
        agent.reset()
    
        while not env.episode_over:
            action, ep_id = agent.act(obs)
            obs = env.step(action)
            # draw_observation(osp.join(config.EVAL.NONLEARNING.DUMP_DIR, f"{ep_id}-{agent.action_index}"), obs['rgb'], obs['depth'])
            draw_observation(osp.join(config.EVAL.NONLEARNING.DUMP_DIR, f"{agent.action_index}"), obs['rgb'], obs['depth'])
        
        import ipdb;ipdb.set_trace() # breakpoint 54
        is_success = env.task.measurements.measures[Success.cls_uuid].get_metric()
        try:
            eval_dict[agent.ep_results['episode_id']] = {
                'episode': agent.ep_results['episode_id'], 
                'traj_id': agent.ep_results['ep_id'],
                'is_success': is_success>0.5}
        except:
            pass
 
        for m, v in env.get_metrics().items():
            stats[m] += v

    stats = {k: v / num_episodes for k, v in stats.items()}

    logger.info(f"Averaged benchmark for {config.EVAL.NONLEARNING.AGENT}:")
    for stat_key in stats.keys():
        logger.info("{}: {:.3f}".format(stat_key, stats[stat_key]))

    print("DONE !!")
    
    
class ActionSimAgent(Agent):

    def __init__(self, config, env, split, annt_format="data/mln3d/annt/{split}/{split}.json.gz"):
      
        with open(config.RESULT_PATH, "r") as f:
            self.data= {str(v['episode_id']):v for v in json.load(f)}
        
        with gzip.open(annt_format.format(split=split), 'r') as f:
            eps_data = json.load(f)["episodes"]

        self.eps_data = {}
        for v in eps_data:
            scene_id = v['scene_id'].split('/')[-1].split('.')[0]
            self.eps_data[str(v['episode_id'])] = {
                'trajectory_id': v['trajectory_id'],
                'scene_id': scene_id,
                'start_position': v['start_position'],
                'start_rotation': v['start_rotation'],
                'goal_position': v['goals'][0]['position'],
                'goal_radius': v['goals'][0]['radius'],
                'instruction': v['instruction']['instruction_text'],
            }

        self.actions = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        
        self.shortest_path_follower = ShortestPathFollower(
            env._sim, goal_radius=0.2, return_one_hot=False, 
            stop_on_error=True
        )
        self.env = env

    def reset(self):
        self.action_index = -1 
        self.current_ep_id = None

    def act(self, observations):
        episode_id = self.env.current_episode.episode_id
        if self.current_ep_id is None or episode_id != self.current_ep_id:
            self.current_ep_id =  episode_id
            # try:
            action_seq  = self.data[episode_id]['action_seq']
            raw_annt = self.eps_data[episode_id]
            print("\n"+"=" * 50)
            print(f"Scene: {raw_annt['scene_id']} Episode {episode_id}", end=' ')
            if self.data[episode_id]['is_correct']:
                print('\033[92m' + "CORRECT" + '\033[0m')
            else:
                print('\033[91m' + "WRONG" + '\033[0m')
            print("Instruction: ", raw_annt['instruction'])
            print("")
            print(action_seq)
            print()
            print()
            self.action_seq = action_seq
            self.action_index = 0
            # except:
            #     print(f"[WARNING] epid {episode_id} not predicted")
            #     return {"action": self.actions[0]}, episode_id

        action = {"action": self.actions[self.action_seq[self.action_index]]}
        self.action_index+=1
        return action, episode_id