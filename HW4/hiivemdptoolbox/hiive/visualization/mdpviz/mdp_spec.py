import typing
from collections import defaultdict

import networkx as nx
import numpy as np

from hiive.visualization.mdpviz.mdp_discrete_env import MDPDiscreteEnv
from hiive.visualization.mdpviz.mdp_env import MDPEnv
from hiive.visualization.mdpviz.transition import Transition
from hiive.visualization.mdpviz.state import State
from hiive.visualization.mdpviz.transition_probabilities import TransitionProbabilities
from hiive.visualization.mdpviz.action import Action
from hiive.visualization.mdpviz.next_state import NextState
from hiive.visualization.mdpviz.reward import Reward
from hiive.visualization.mdpviz.outcome import Outcome


class MDPSpec(object):
    def __init__(self, verbose=False):
        self._states = {}
        self._actions = {}
        self.states = []
        self.actions = []
        self.state_outcomes: typing.Dict[tuple, typing.List[NextState]] = defaultdict(list)
        self.reward_outcomes: typing.Dict[tuple, typing.List[Reward]] = defaultdict(list)
        self.gamma = 1.0
        self._node_attribute_dictionary = {}
        self._edge_attribute_dictionary = {}
        self.verbose = verbose

    def reset(self):
        self._node_attribute_dictionary = {}
        self._edge_attribute_dictionary = {}
        self._states = {}
        self._actions = {}
        self.states = []
        self.actions = []
        self.state_outcomes: typing.Dict[tuple, typing.List[NextState]] = defaultdict(list)
        self.reward_outcomes: typing.Dict[tuple, typing.List[Reward]] = defaultdict(list)

    def has_state(self, state_name):
        return state_name in [str(s) for s in self._states]

    def get_state(self, state_name):
        if self.has_state(state_name):
            return self._states[state_name]
        return None

    def state(self, name=None, index=None, terminal_state=False, extra_data=None):
        if not name:
            if not terminal_state:
                name = 'S%s' % self.num_states
            else:
                name = 'T%s' % self.num_states
        if name not in self.states:
            index = self.num_states if index is None else index
            new_state = State(name, index=index, terminal_state=terminal_state, extra_data=extra_data)
            self._states[name] = new_state
            self.states.append(new_state)
        return self._states[name]

    def action(self, name=None, extra_data=None):
        if not name:
            name = 'A%s' % self.num_actions

        if name not in self.actions:
            new_action = Action(name=name, index=self.num_actions, extra_data=extra_data)
            self._actions[name] = new_action
            self.actions.append(new_action)
        return self._actions[name]

    def transition(self, state: State, action: Action, outcome: Outcome):
        """Specify either a next state or a reward as `outcome` for a transition."""

        if isinstance(outcome, NextState):
            self.state_outcomes[state, action].append(outcome)
        elif isinstance(outcome, Reward):
            self.reward_outcomes[state, action].append(outcome)
        else:
            raise NotImplementedError()

    @property
    def num_states(self):
        return len(self._states)

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def is_deterministic(self):
        for state in self.states:
            for action in self.actions:
                if len(self.reward_outcomes[state, action]) > 1:
                    return False
                if len(self.state_outcomes[state, action]) > 1:
                    return False
        return True

    def __repr__(self):
        return 'Mdp(states=%s, actions=%s, state_outcomes=%s, reward_outcomes=%s)' % (
            self.states, self.actions, dict(self.state_outcomes), dict(self.reward_outcomes))

    def set_edge_attributes(self, u, v, a, **kwargs):
        key = (u, v, a)
        update_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if key not in self._edge_attribute_dictionary:
            self._edge_attribute_dictionary[key] = {}
        self._edge_attribute_dictionary[key].update(update_kwargs)

        del_kwargs = {k: v for k, v in kwargs.items() if v is None}
        for k in del_kwargs:
            self._edge_attribute_dictionary.pop(k, None)

        if len(self._edge_attribute_dictionary[key]) == 0:
            self._edge_attribute_dictionary.pop(key, None)

    def set_node_attributes(self, n, **kwargs):
        update_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if n not in self._node_attribute_dictionary:
            self._node_attribute_dictionary[n] = {}
        self._node_attribute_dictionary[n].update(update_kwargs)

        del_kwargs = {k: v for k, v in kwargs.items() if v is None}
        for k in del_kwargs:
            self._node_attribute_dictionary.pop(k, None)

        if len(self._node_attribute_dictionary[n]) == 0:
            self._node_attribute_dictionary.pop(n, None)

    def to_graph(self, highlight_state: State = None, highlight_action: Action = None,
                 highlight_next_state: State = None):
        transitions = TransitionProbabilities(self)

        graph = nx.MultiDiGraph()
        self._node_attribute_dictionary = {}
        self._edge_attribute_dictionary = {}

        for state in self.states:
            fillcolor = 'yellow' if highlight_state == state else 'red' if highlight_next_state == state else '#E0E0E0'
            self.set_node_attributes(n=state,
                                     shape='doubleoctagon' if state.terminal_state else 'circle',
                                     label=state.name,
                                     fontname='consolas',
                                     fillcolor=fillcolor,
                                     style='filled',
                                     type='state')
            # for s2 in self.states:
            #    self.set_edge_attributes(u=state, v=s2, label=f'{state.name} ->{s2.name}')

        t_index = 0
        for state in self.states:
            if not state.terminal_state:
                for action in self.actions:
                    reward_probs = transitions.rewards[state, action].items()
                    expected_reward = sum(reward * prob for reward, prob in reward_probs)

                    action_label = f'{action.name}\n({expected_reward:+.2f})'

                    next_states = transitions.next_states[state, action].items()
                    action_color = action.index + 1
                    # color = f'/set28/{action_color}'
                    fontcolor = f'/set19/{action_color}'
                    color = f'/set19/{action_color}'
                    if len(next_states) == 1:
                        next_state, _ = list(next_states)[0]
                        self.set_edge_attributes(u=state, v=next_state, a=action,
                                                 type='state_to_state',
                                                 color=color,
                                                 fontsize=8,
                                                 decorate=False,
                                                 fontname='consolas',
                                                 labelfloat=False,
                                                 fontcolor=fontcolor,
                                                 label=action_label)
                    else:
                        transition = Transition(action, state, t_index)
                        t_index += 1
                        self.set_edge_attributes(u=state, v=transition, a=action,
                                                 type='state_to_transition',
                                                 color=color,
                                                 fontcolor=fontcolor,
                                                 decorate=False,
                                                 fontname='consolas',
                                                 fontsize=8,
                                                 labelfloat=False,
                                                 label=action_label)
                        # transition_label = f'{state.name, action.name}'
                        self.set_node_attributes(n=transition, type='transition',  # label=transition_label,
                                                 fillcolor=color,
                                                 color=color,
                                                 style='filled, bold', shape='point')
                        # rewards = [reward for reward, _ in reward_probs]
                        for i, nsp in enumerate(next_states):
                            next_state, prob = nsp
                            if not prob:
                                continue
                            # color = f'/set28/{action_color}'
                            # color = f'/pastel19/{action_color}'
                            # fontcolor = f'/set19/{action_color}'
                            # reward = transition.reward
                            transition_label = f'{(prob * 100):3.2f}%'  #\n({rewards[i]:+.2f})'
                            self.set_edge_attributes(u=transition, v=next_state, a=action,
                                                     label=transition_label,
                                                     color=color,
                                                     decorate=False,
                                                     fontname='consolas',
                                                     fontsize=6.5,
                                                     labelfloat=False,
                                                     fontcolor=fontcolor,
                                                     type='transition_to_state')

                        if state == highlight_state and action == highlight_action:
                            self.set_node_attributes(n=transition, style='bold')
                            self.set_edge_attributes(u=transition, v=next_state,  a=action,
                                                     style='bold', color='green')
                            if highlight_next_state:
                                self.set_edge_attributes(u=transition, v=highlight_next_state, a=action,
                                                         style='bold',
                                                         color='red', data='e4')
                        """
                        else:
                            next_state, _ = list(next_states)[0]
                            # if state == highlight_state and action == highlight_action:
                            self.set_node_attributes(n=(next_state, action), label=action_label, style='bold', color='red')

                        """
        # build nodes
        # graph.node.clear()
        for n, node_attributes in self._node_attribute_dictionary.items():
            graph.add_node(node_for_adding=n, **node_attributes)
            if self.verbose:
                print(f'Adding node: {n}, nodes={len(graph.nodes)}')
        if self.verbose:
            print()

        # build edges
        # graph.edge.clear()
        for edge_key, edge_attributes in self._edge_attribute_dictionary.items():
            u, v, _ = edge_key
            graph.add_edge(u_for_edge=u, v_for_edge=v, **edge_attributes)
            if self.verbose:
                print(f'Adding edge: u={u}, v={v}, edges={len(graph.edges)}, attributes={edge_attributes}')

        if self.verbose:
            print()
        # for some reason, adding edges clears out the node dictionary.
        # so we set it now.
        """
        for n, node_attributes in self._node_attribute_dictionary.items():
            graph.node[n] = node_attributes
            if self.verbose:
                print(f'Setting node attributes for [{n}]: {node_attributes}')
        if self.verbose:
            print()
        """

        return graph

    def get_node_attributes(self, graph, state):
        if isinstance(graph.nodes, dict):
            attributes = graph.nodes[state]
        else:
            attributes = graph.nodes(state)[0][1]
        return attributes

    def to_env(self):
        return MDPEnv(self)

    def to_discrete_env(self):
        return MDPDiscreteEnv(self)

    def validate(self):
        # For now, just validate by trying to compute the transitions.
        # It will raise errors if anything is wrong.
        TransitionProbabilities(self)
        return self

    def get_transition_and_reward_arrays(self, p_default=0.5):
        """Generate the fire management transition and reward matrices.

        The output arrays from this function are valid input to the mdptoolbox.mdp
        classes.

        Let ``S`` = number of states, and ``A`` = number of actions.

        Parameters
        ----------
        p_default : float
            The class-independent probability of the population staying in its
            current population abundance class.

        Returns
        -------
        out : tuple
            ``out[0]`` contains the transition probability matrices P and
            ``out[1]`` contains the reward vector R. P is an  ``A`` × ``S`` × ``S``
            numpy array and R is a numpy vector of length ``S``.

        """
        assert 0 <= p_default <= 1, "'p_default' must be between 0 and 1"

        n_actions = len(self.actions)
        n_states = len(self.states)

        # The transition probability array
        transition_probabilities = np.zeros((n_actions, n_states, n_states))
        # The reward vector
        rewards = np.zeros((n_states, n_actions))
        # Loop over all states
        for state in self.states:
            s = state.index
            # Loop over all actions
            w = 0.0
            total_transition_weight = 0
            for action in self.actions:
                a = action.index
                reward_info = self.reward_outcomes[(state, action)]
                r = np.sum([rwi.outcome * rwi.weight for rwi in reward_info])
                w += np.sum([rwi.weight for rwi in reward_info])
                rewards[s, a] = r
                # Assign the transition probabilities for this state, action pair
                if state.terminal_state:
                    pass
                    transition_probabilities[a][s][s] = 1.0
                    total_transition_weight += 1.0
                else:
                    transitions = self.state_outcomes[(state, action)]
                    total_transition_weight += np.sum([so.weight for so in transitions])
                    for transition in transitions:
                        state_next = transition.outcome
                        transition_probabilities[a][s][state_next.index] = transition.weight
                # transition_probabilities[a, s, ] /= total_transition_weight
                ttp = np.sum(transition_probabilities[a, s, :])
                if ttp > 0:
                    transition_probabilities[a, s, :] /= np.sum(transition_probabilities[a, s, :])
            if w > 0:
                rewards[s, :] /= w

        return transition_probabilities, rewards
