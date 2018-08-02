class PolicyGradientPlayer(object):
    def __init__(self, ctrl_fns):
        self._ctrl_fns = ctrl_fns

    def move(self, state):
        return self._ctrl_fns[state.cur_player](state)
