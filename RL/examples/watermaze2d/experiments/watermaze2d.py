from .object import *
import sys
import os
# from register import register

path = os.path.abspath(os.path.dirname(__file__))
with open(path + '/env.list', 'r') as a:
    f = a.readlines()
env_list = {}
for i in range(len(f)):
    b = f[i].split('@')
    env_list[b[0]] = eval(b[1])

class Watermaze2dEnv(WaterMaze2d):
    """
    This environment is a memory test. The agent starts in a small room
    where it sees an object. It then has to go through a narrow hallway
    which ends in a split. At each end of the split there is an object,
    one of which is the same as the object in the starting room. The
    agent has to remember the initial object, and go to the matching
    object at split.
    """

    def __init__(
        self,
        seed,
        mode,
        size=8,
        random_length=False,
        agent_view_size = 3,
        cfg=None
    ):
        self.cfg = cfg
        self.random_length = random_length
        self.mode = mode
        self.size = size
        self.max_step=10*size**2 # 
        self.stay = 0
        self.random_list = env_list[str(size)]

        self.corner_list = [[2, size-3, Square_yellow], [2, 2, Square_green], [size-3, size-3, Tri_green], [size-3, 2, Tri_yellow]]

        self.num = 0

        super().__init__(
            seed=seed,
            grid_size=size,
            max_steps=self.max_step, # size=20
            # Set this to True for maximum speed
            see_through_walls=False,
            agent_view_size = agent_view_size
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls

        self.grid.wall_rect(0, 0, width, height)

        if self.cfg.watermaze2d_mode == 'easy':
            for i in range(width):
                pos1 = self.random_list[i]
                self.grid.set(*pos1, Square_yellow()) 
        elif self.cfg.watermaze2d_mode == 'hard':
            for i in range(4):
                pos1 = self.corner_list[i]
                self.grid.set(*pos1[:2], pos1[2]()) 
        else:
            print('Error')
            sys.exit()

        # for i in range(1,6):            # 注释
        #     self.grid.set(i,3,Wall())
        
        # 智能体
        self.agent_pos = (self._rand_int(1,self.size - 1),self._rand_int(1,self.size - 1))#(self._rand_int(1, 9), 3)
        self.agent_dir = 0

        # 目标
        pos0 = (self._rand_int(1,self.size - 1),self._rand_int(1,self.size - 1))
        while self.agent_pos == pos0:
            pos0 = (self._rand_int(1,self.size - 1),self._rand_int(1,self.size - 1))
        # self.grid.set(*pos0, Ball_green()) # 6,3

        # 干扰
        # self.grid.set(6,3,self._rand_elem([Ball_blue, Ball_green])())

        # Choose the target objects
        self.success_pos = pos0

        self.mission = 'go to the matching object at the end of the hallway'
    
    def reset(self):
        self.stay = 0
        obs = WaterMaze2d.reset(self)

        return obs

    def step(self, action):
        if tuple(self.agent_pos) == self.success_pos:
            action = WaterMaze2d.Actions.stay
            self.stay += 1
            if self.stay >= 5:
                self.stay = 0
                self.agent_pos = (self._rand_int(1,self.size - 1),self._rand_int(1,self.size - 1))
                self.agent_dir = 0
            # print(self.stay, self.agent_pos)
        obs, reward, done, info = WaterMaze2d.step(self, action)

        if tuple(self.agent_pos) == self.success_pos:
            reward = 1
            action == WaterMaze2d.Actions.stay
            self.num += 1
        else:
            reward = 0
        
        if self.step_count == 1e6 or self.num == 20:
            done = True
            self.num = 0

        return obs, reward, done, info

class Watermaze2d(Watermaze2dEnv):
    def __init__(self, mode, seed=None):
        super().__init__(mode=mode, seed=seed, size=20)

if __name__ == "__main__":
    env = Watermaze2d(seed=1)
    env.render()
    input()