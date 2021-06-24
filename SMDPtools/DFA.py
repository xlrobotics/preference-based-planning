__author__ = 'Jie Fu, jfu2@wpi.edu'

from types import *


class ExceptionFSM(Exception):
    """This is the FSM Exception class."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class Action:
    def __init__(self, v, boolv=True):
        self.v = v
        self.boolv = boolv

    def conjunction(self, other):
        # return (self.display(), other.display())
        return self.display() + "&&" + other.display()

    def negation(self):
        return "!" + self.v

    def display(self):
        if self.boolv:
            return self.v
        else:
            return "!" + self.v


class DFA:
    """This is a deterministic Finite State Automaton (DFA).
    """

    def __init__(self, initial_state=None, alphabet=None, transitions=dict([]), final_states=None, memory=None):
        self.state_transitions = {}
        self.final_states = set([])
        self.state_transitions = transitions
        if alphabet == None:
            self.alphabet = []
        else:
            self.alphabet = alphabet
        self.initial_state = initial_state
        self.states = [initial_state]  # the list of states in the machine.
        self.sink_states = set([])
        self.effTS = {}
        self.invEffTS = {}
        self.transition_tree = {}
        self.state_info = {}
        self.g_unsafe = ''
        self.final_transitions = []
        self.pref_labels = {}
        self.pref_trans = {}
        self.inv_pref_trans = {}
        self.inv_pref_labels = {}
        # self.actions = []

    def pref_labeling(self, q, pref_node):
        if q not in self.pref_labels:
            self.pref_labels[q] = pref_node

    def add_pref_trans(self, a, b): # a->b, b is more preferred
        if a not in self.pref_trans:
            self.pref_trans[a] = []

        if b not in self.pref_trans[a]:
            self.pref_trans[a].append(b)

    def clear(self):
        self.state_transitions = {}
        self.final_states = set([])
        self.alphabet == []
        self.initial_state = []
        self.states = []  # the list of states in the machine.
        self.sink_states = set([])
        self.effTS = {}
        self.invEffTS = {}
        self.transition_tree = {}
        self.state_info = {}
        self.g_unsafe = ''

    def reset(self):

        """This sets the current_state to the initial_state and sets
        input_symbol to None. The initial state was set by the constructor
         __init__(). """

        self.current_state = self.initial_state
        self.input_symbol = None

    def add_transition(self, input_symbol, state, next_state=None):
        if next_state is None:
            next_state = state
        else:
            if state != next_state and next_state in self.final_states:
                if input_symbol not in self.final_transitions:
                    self.final_transitions.append(input_symbol)
            self.state_transitions[(input_symbol, state)] = next_state
            if state != next_state:
                if state not in self.transition_tree:
                    self.transition_tree[state] = []
                    self.state_info[state] = {'safe': [], 'unsafe': []}

                if next_state not in self.transition_tree[state]:
                    self.transition_tree[state].append(next_state)
                    if next_state in self.sink_states:
                        # print "insink"
                        self.state_info[state]['unsafe'].append(input_symbol)
                    else:
                        self.state_info[state]['safe'].append(input_symbol)

                if tuple([state, next_state]) not in self.invEffTS:
                    self.invEffTS[state, next_state] = []
                if input_symbol not in self.invEffTS[state, next_state]:
                    self.invEffTS[state, next_state].append(input_symbol)
                if input_symbol not in self.effTS:
                    self.effTS[input_symbol] = []
                self.effTS[input_symbol].append([state, next_state])

        if next_state in self.states:
            pass
        else:
            self.states.append(next_state)
        if state in self.states:
            pass
        else:
            self.states.append(state)

        if input_symbol in self.alphabet:
            pass
        else:
            self.alphabet.append(input_symbol)

    def prune_eff_transition(self):
        delList = []
        for key in self.effTS:
            for state in self.effTS[key]:
                if state[1] in self.sink_states:
                    delList.append(key)
                    break
        for key in delList:
            del self.effTS[key]
            # self.effTS[key].remove(state)

    def get_transition(self, input_symbol, state):

        """This returns a list of next states given an input_symbol and state.
        """

        if (input_symbol, state) in self.state_transitions:
            return self.state_transitions[(input_symbol, state)]
        else:
            return None

    def predecessor(self, s):  ## isn't this successor?
        """
        list a set of predecessor for state s.
        """
        transFrom = set([])
        for a in self.alphabet:
            if (a, s) in self.state_transitions:
                transFrom.add((s, a, self.get_transition(a, s)))
        return transFrom

    def accessible(self):
        """
        list a set of reachable states from the initial state. Used for pruning.
        """
        reachable, index = [self.initial_state], 0
        while index < len(reachable):
            state, index = reachable[index], index + 1
            for (s0, a, s1) in self.transitionsFrom(state):
                if s1 not in reachable:
                    reachable.append(s1)
        states = []
        for s in reachable:
            states.append(s)
        transitions = dict([])
        for ((a, s0), s1) in self.state_transitions.items():
            if s0 in states and s1 in states:
                transitions[(a, s0)] = s1
        self.states = states
        self.state_transitions = transitions
        return

    def Trim(self):
        # remove states that are not reachable from the initial state.
        reachable, index = [self.initial_state], 0
        while index < len(reachable):
            state, index = reachable[index], index + 1
            for (s0, a, s1) in self.transitionsFrom(state):
                if s1 not in reachable:
                    reachable.append(s1)
        endable, index = list(self.final_states), 0
        while index < len(endable):
            state, index = endable[index], index + 1
            for ((a, s0), s1) in self.state_transitions.items():
                if s1 == state and s0 not in endable:
                    endable.append(s0)
        states = []
        for s in reachable:
            if s in endable:
                states.append(s)
        if not states:
            print("NO states after trimming. Null FSA.")
            return None
        transitions = dict([])
        for ((a, s0), s1) in self.state_transitions.items():
            if s0 in states and s1 in states:
                transitions[(a, s0)] = s1
        self.states = states
        self.state_transitions = transitions
        return

    def toDot(self, filename):
        f = open(filename, 'w')
        f.write("digraph G { rankdir = LR\n")
        i = 0
        indexed_states = {}
        for q in self.states:
            indexed_states[q] = i
            if self.final_states:
                if q in self.final_states:
                    f.write("\t" + str(i) + "[label=\"" + str(q) + "\",shape=doublecircle]\n")
                else:
                    f.write("\t" + str(i) + "[label=\"" + str(q) + "\",shape=circle]\n")
            else:
                f.write("\t" + str(i) + "[label=\"" + str(q) + "\",shape=circle]\n")
            i += 1
        for ((a, s0), s1) in self.state_transitions.items():
            f.write("\t" + str(indexed_states[s0]) + "->" + str(indexed_states[s1]) + "[label = \"" + str(a) + "\"]\n")
        f.write("}")
        f.close()

    def set_final(self, state):
        self.final_states.add(state)

    def set_sink(self, state):
        self.sink_states.add(state)
        self.set_final(state)

    # TODO add a string parser for the input test, e.g recognize that 'g' satisfies '!y && !r'
    def parser(self, action):
        return


class DRA(DFA, object):
    """A child class of DFA --- determinisitic Rabin automaton
    """

    def __init__(self, initial_state=None, alphabet=None, transitions=dict([]), rabin_acc=None, memory=None):
        # The rabin_acc is a list of rabin pairs rabin_acc=[(J_i, K_i), i =0,...,N]
        # Each K_i, J_i is a set of states.
        # J_i is visited only finitely often
        # K_i has to be visited infinitely often.
        super(DRA, self).__init__(initial_state, alphabet, transitions)
        self.acc = rabin_acc

    def add_rabin_acc(self, rabin_acc):
        self.acc = rabin_acc


class DRA2(DFA, object):
    """A child class of DFA --- determinisitic Rabin automaton
    """

    def __init__(self, initial_state=None, alphabet=None, transitions=dict([]), rabin_acc=None, memory=None):
        # The rabin_acc is a list of rabin pairs rabin_acc=[(J_i, K_i), i =0,...,N]
        # Each K_i, J_i is a set of states.
        # J_i is visited only finitely often
        # K_i has to be visited infinitely often.
        super(DRA2, self).__init__(initial_state, alphabet, transitions)
        self.acc = rabin_acc

    def add_rabin_acc(self, rabin_acc):
        self.acc = rabin_acc


if __name__ == '__main__':
    # construct a DRA, which is a complete automaton.
    dra = DRA(0, ['1', '2', '3', '4', 'E'])  # we use 'E' to stand for everything else other than 1,2,3,4.
    dra.add_transition('2', 0, 0)
    dra.add_transition('3', 0, 0)
    dra.add_transition('E', 0, 0)

    dra.add_transition('1', 0, 1)
    dra.add_transition('1', 1, 2)
    dra.add_transition('3', 1, 2)
    dra.add_transition('E', 1, 2)

    dra.add_transition('2', 1, 3)

    dra.add_transition('1', 2, 2)
    dra.add_transition('3', 2, 2)
    dra.add_transition('E', 2, 2)

    dra.add_transition('2', 2, 3)

    dra.add_transition('1', 3, 3)
    dra.add_transition('2', 3, 3)
    dra.add_transition('E', 3, 3)

    dra.add_transition('3', 3, 0)

    dra.add_transition('4', 0, 4)
    dra.add_transition('4', 1, 4)
    dra.add_transition('4', 2, 4)
    dra.add_transition('4', 3, 4)
    dra.add_transition('4', 4, 4)
    dra.add_transition('1', 4, 4)
    dra.add_transition('2', 4, 4)
    dra.add_transition('3', 4, 4)
    dra.add_transition('E', 4, 4)

    J0 = {4}
    K0 = {1}
    rabin_acc = [(J0, K0)]
    dra.add_rabin_acc(rabin_acc)

    # print dra.states
    # print dra.state_transitions
    # print dra.predecessor(1)

    # print dra.get_transition()
    dra.toDot(__file__ + '_test')

    # dra.Trim()

    # dra.toDot(__file__ + '_test_after_trim')
    # print dra.