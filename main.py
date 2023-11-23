import sys
from CartPole.DQN_Agent_Class.DQNAgentClass import DQNAgent
from CartPole.Let_User_Play.Let_User_Play import let_user_play

def run_factory(mode='test'):
    agent = DQNAgent()
    if mode == 'train':
        agent.run()
    elif mode == 'test':
        agent.test()
    elif mode == 'play':
        let_user_play()
    else:
        print("Invalid mode. Please choose 'train', 'play' or 'run'.")

mode = sys.argv[1]
run_factory(mode=mode)
