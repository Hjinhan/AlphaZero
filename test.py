
import os
import pickle
import time
  
from GoEnv.environment import GoEnv
from self_play import MCTS
from configure import Config
from model import AlphaZeroNetwork

def simulate_game(black_agent, white_agent,config):  # two agent play
    go_env = GoEnv(config)
    game_state, done = go_env.reset()
    BLACK = 1
    WHITE = 2
    agents = {
        BLACK: black_agent,
        WHITE: white_agent,
    }
    while not done:
        next_player = go_env.getPlayer(game_state)
        next_action = agents[next_player].select_action(game_state)
        game_state, done = go_env.step(game_state,next_action)
    winner = go_env.getWinner(game_state)
    return winner


def fab_agents(model_paths, config, go_env):
    models = []
    agents = []
    num_agents = len(model_paths)
    print("\nnum_agents:\n",num_agents)
    for _ in range(num_agents):
        model = AlphaZeroNetwork(config).to(config.device)
        models.append(model)
    
    for i, path in enumerate(model_paths):
        if os.path.exists(path):
            print("agent{}_path is exists.\n".format(i))
            with open(path, "rb") as f:
                    model_weights = pickle.load(f)
                    models[i].set_weights( model_weights["weights"])

    for i in range(num_agents):
        agent = MCTS(config, go_env, models[i])
        agents.append(agent)

    return agents
 
if __name__ == '__main__':
    config = Config()
    go_env = GoEnv(config)
    num_games = 1
    print(config.device)
    model_paths = ["./save_weight/best_policy_1200.model", "./save_weight/best_policy_1400.model"]
    agents = fab_agents(model_paths, config, go_env)
                       
    # test time
    start = time.time()
    simulate_game(agents[0],agents[1],config)
    end = time.time()
    print("run time:%.4fs" % (end - start))