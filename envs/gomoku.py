import numpy as np
import gym

class GomokuEnv(gym.Env):
    def __init__(self, board_size, num_win, num_player=2):
        self._board_size = board_size
        self._num_win = num_win
        self._num_player = num_player

        self.reset()

    @property
    def cur_player(self):
        return self._turn % self._num_player

    @property
    def num_player(self):
        return self._num_player

    def reset(self):
        self._board = - np.ones( (self._board_size, self._board_size), dtype=int )
        self._turn = 0
        init_state = self._board.copy()
        init_state.setflags(write=False)
        return init_state

    def step(self, action):
        x, y = action
        player = self._turn % self._num_player

        if not self._available(x, y):
            print("Illegal Move for player {}".format(player))
            raise ValueError

        self._board[x, y] = player
        self._turn += 1

        full = self._turn >= self._board_size**2
        win = self._checkForWin(x, y)

        done = full or win
        reward = int(win)
        state = self._board.copy()
        state.setflags(write=False)
        return state, reward, done, {}

    def render(self, mode='human'):
        board_string = str(self._board)
        board_string = board_string.replace('[', ' ').replace(']', ' ').replace('-1', '  ')
        board_string = board_string.replace('0', '-').replace('1', '+')
        print(render)

    def _valid(self, x, y):
        def _in_range(v):
            return (0 <= v) and (v < self._board_size)
        return _in_range(x) and _in_range(y)

    def _available(self, x, y):
        return self._valid(x, y) and self._board[x][y] == -1

    def _checkLines(self, pos, delta, player):
        x, y = pos
        dx, dy = delta

        count = 0
        while(self._valid(x, y)):
            if self._board[x][y] == player:
                x += dx
                y += dy
                count += 1
            else:
                break
        return count - 1

    def _checkForWin(self, x, y):
        dirs = ( (1, 0), (0, 1), (1, 1), (1, -1) )

        player = self._board[x][y]
        for (dx, dy) in dirs:
            sum_1 = self._checkLines( (x, y), ( dx,  dy), player )
            sum_2 = self._checkLines( (x, y), (-dx, -dy), player )
            if (sum_1 + sum_2) >= (self._num_win - 1):
                return True
        return False

