from make_env import make_env

def test():
    env = make_env("group_spread")
    env.reset()
    for i in range(200):
        env.render('')
        actions = [space.sample() for space in env.action_space]
        env.step(actions)
    env.close()

if __name__ == "__main__":
    test()