import numpy as np
import gym

import numba as nb
@nb.jit(nb.boolean(nb.i4[:, :], nb.i4, nb.i4))
def valid(board, x, y):
    board_size = board.shape[0]
    def _in_range(v):
        return (0 <= v) and (v < board_size)
    return _in_range(x) and _in_range(y)

@nb.jit(nb.boolean(nb.i4[:, :], nb.i4, nb.i4))
def available(board, x, y):
    return valid(board, x, y) and board[x][y] == -1

@nb.jit(nb.boolean(nb.i4[:, :], nb.i4, nb.i4, nb.i4))
def checkForWin(board, x, y, num_win):
    def _checkLines(pos, delta, player):
        x, y = pos
        dx, dy = delta

        count = 0
        while(valid(board, x, y)):
            if board[x][y] == player:
                x += dx
                y += dy
                count += 1
            else:
                break
        return count - 1

    dirs = ( (1, 0), (0, 1), (1, 1), (1, -1) )

    player = board[x][y]
    for (dx, dy) in dirs:
        sum_1 = _checkLines( (x, y), ( dx,  dy), player )
        sum_2 = _checkLines( (x, y), (-dx, -dy), player )
        if (sum_1 + sum_2) >= (num_win - 1):
            return True
    return False

class GomokuState(object):
    def __init__(self, board):
        self.board = board

    def __hash__(self):
        return hash(self.board.tobytes())

    def __eq__(self, other):
        return (self.board == other.board).all()

    @property
    def valid_actions(self):
        return list(zip(*np.where(self.board==-1)))

class GomokuEnv(gym.Env):
    def __init__(self, board_size, num_win, num_player=2):
        self._board_size = board_size
        self._num_win = num_win
        self._num_player = num_player

        self._board = - np.ones((self._board_size, self._board_size), dtype=np.int32)
        self._turn = None
        self.reset()

    @property
    def cur_player(self):
        return self._turn % self._num_player

    @property
    def num_player(self):
        return self._num_player

    def reset(self):
        self._board.fill(-1)
        self._turn = 0
        return GomokuState(self._board.copy())

    def step(self, action):
        x, y = action
        player = self._turn % self._num_player

        if not available(self._board, x, y):
            print("Illegal Move for player {}".format(player))
            print("Board:")
            print(self._board)
            print("Action:", action)
            raise ValueError

        self._board[x, y] = player
        self._turn += 1

        full = self._turn >= self._board_size**2
        win = checkForWin(self._board, x, y, num_win=self._num_win)

        done = full or win
        reward = int(win)
        return GomokuState(self._board.copy()), reward, done, {}

    def render(self, mode='human'):
        board_string = str(self._board)
        board_string = board_string.replace('[', ' ').replace(']', ' ').replace('-1', '  ')
        board_string = board_string.replace('0', '-').replace('1', '+')
