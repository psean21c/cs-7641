from hiive.visualization.mdpviz.next_state import NextState
from hiive.visualization.mdpviz.reward import Reward


class TransitionProbabilities:
    """Container for transition probabilities."""

    def __init__(self, mdp_spec):
        self.next_states = {}
        self.rewards = {}
        for state in mdp_spec.states:
            for action in mdp_spec.actions:
                next_states = NextState.get_choices(mdp_spec.state_outcomes[state, action])
                if not state.terminal_state and not next_states:
                    raise ValueError('No next states specified for non-terminal (%s, %s)!' % (state, action))
                if state.terminal_state and next_states:
                    raise ValueError('Next states %s specified for terminal (%s, %s)!' % (next_states, state, action))
                self.next_states[state, action] = next_states

                rewards = mdp_spec.reward_outcomes[state, action]
                if state.terminal_state and rewards:
                    raise ValueError('Rewards %s specified for terminal (%s, %s)!' % (next_states, state, action))
                self.rewards[state, action] = Reward.get_choices(rewards)

    def __repr__(self):
        return 'TransitionProbabilities(%s)' % self.__dict__
