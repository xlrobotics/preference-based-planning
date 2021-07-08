"""
File: automata.py
Project: ggsolver_nx (https://github.com/abhibp1993/ggsolver/tree/nx-solvers)

Description: TODO.
"""

import copy
import itertools
import networkx as nx
from abc import ABC, abstractmethod
from itertools import chain, combinations


ACC_REACHABILITY = "Reachability"
ACC_BUCHI = "Buchi"
ACC_PREFERENCE = "Preference Graph"
ACC = [ACC_BUCHI, ACC_REACHABILITY, ACC_PREFERENCE]


def powerset(iterable):
    """ Returns powerset of given set. """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class BaseAutomaton(ABC):
    EXPLICIT = "explicit"
    SYMBOLIC = "symbolic"

    def __init__(self):
        self._graph = None
        self._alphabet = set()
        self._delta = None
        self._pre = None
        self._post = None
        self._init_st = None
        self._final = None
        self._acc = None
        self._mode = None
        self._is_initialized = False
        self.properties = dict()

    @abstractmethod
    def _validate_graph(self, graph):
        pass

    @abstractmethod
    def construct_explicit(self, graph, init_st, final, acc):
        pass

    @abstractmethod
    def construct_symbolic(self, states, alphabet, delta, pre, post, init_st, final, acc):
        pass

    @abstractmethod
    def delta(self, u, f):
        pass

    @abstractmethod
    def pre(self, v):
        pass

    @abstractmethod
    def post(self, u):
        pass

    @property
    def graph(self):
        if self._is_initialized and self._mode == self.EXPLICIT:
            return self._graph
        raise ValueError("Automaton is NOT initialized or mode is NOT explicit.")

    @property
    def states(self):
        if self._is_initialized:
            return self._graph.nodes()
        raise ValueError("Automaton is NOT initialized.")

    @property
    def alphabet(self):
        if self._is_initialized:
            return self._alphabet
        raise ValueError("Automaton is NOT initialized.")

    @property
    def init_st(self):
        if self._is_initialized:
            return self._init_st
        raise ValueError("Automaton NOT initialized.")

    @property
    def final(self):
        if self._is_initialized:
            return self._final
        raise ValueError("Automaton is NOT initialized.")

    @property
    def acc(self):
        if self._is_initialized:
            return self._acc
        raise ValueError("Automaton is NOT initialized.")


class Dfa(BaseAutomaton):
    def __init__(self):
        _final: set
        super(Dfa, self).__init__()
        self._acc = ACC_REACHABILITY

    def __str__(self):
        output = f"""This DFA has {len(self.states)}.
Alphabet: {self.alphabet}
Starting state: {self.init_st}
Accepting states: {self.final}
States: {self.states}
Transition function:"""
        for src in self.states:
            for sigma in powerset(self.alphabet):
                dst = self.delta(src, sigma)
                output += "\n\t" + f"{src} -- {set(sigma)} --> {dst}" if len(sigma) > 0 else \
                    "\n\t" + f"{src} -- {{}} --> {dst}"
        return output

    def _validate_graph(self, graph):
        """
        Checks that:
        (1) The accepting-state set is a subset of the state set.
        (2) The start-state is a member of the state set.
        (3) Every transition returns a member of the state set.
        """
        # TODO. Implement function.
        return True

    def construct_symbolic(self, states, alphabet, delta, pre, post, init_st, final, acc=ACC_REACHABILITY):
        if self._is_initialized:
            raise RuntimeError("Cannot initialize an already initialized game.")

        assert isinstance(final, set)

        self._graph = nx.MultiDiGraph()
        self._graph.add_nodes_from(states)
        self._alphabet = alphabet
        self._delta = delta
        self._pre = pre
        self._post = post
        self._init_st = init_st
        self._final = final
        self._mode = self.SYMBOLIC
        self._is_initialized = True

    def construct_explicit(self, graph, init_st, final, acc=ACC_REACHABILITY):
        if self._is_initialized:
            raise RuntimeError("Cannot initialize an already initialized game.")

        def _delta(state, true_atoms):
            # Construct evaluation map
            eval_map = dict()
            for atom in self.alphabet:
                if atom in true_atoms:
                    eval_map[atom] = True
                else:
                    eval_map[atom] = False

            # Get successors
            succ = [v for v, f in self.post(state) if f.evaluate(eval_map)]
            if len(succ) == 1:
                return succ[0]
            elif len(succ) == 0:
                return ValueError("Automaton transition function is incomplete.")
            else:
                return ValueError("Automaton is not deterministic.")

        def _pre(state):
            return {(e[0], e[2]["formula"]) for e in self.graph.in_edges(state, data=True)}

        def _post(state):
            return {(e[1], e[2]["formula"]) for e in self.graph.out_edges(state, data=True)}

        if not self._validate_graph(graph):
            raise ValueError("Graph validation failed.")

        assert isinstance(final, set)

        alphabet = set()
        for _, _, data in graph.edges(data=True):
            alphabet.update(data["formula"].alphabet)

        self._graph = graph
        self._alphabet = alphabet
        self._delta = _delta
        self._pre = _pre
        self._post = _post
        self._init_st = init_st
        self._final = final
        self._mode = self.EXPLICIT
        self._is_initialized = True

    def delta(self, u, true_atoms):
        # assert set(true_atoms).issubset(self.alphabet), f"true_atoms must be subset of alphabet."
        if not self._is_initialized:
            raise ValueError(f"Cannot access delta function of 'uninitialized' DFA.")
        return self._delta(u, true_atoms)

    def pre(self, v):
        if not self._is_initialized:
            raise ValueError(f"Cannot access pre function of 'uninitialized' DFA.")
        return self._pre(v)

    def post(self, u):
        if not self._is_initialized:
            raise ValueError(f"Cannot access post function of 'uninitialized' DFA.")
        return self._post(u)


class Nba(BaseAutomaton):
    def __init__(self):
        _final: set
        super(Nba, self).__init__()
        self._acc = ACC_BUCHI

    def __str__(self):
        output = f"""This NBA has {len(self.states)} states.
Alphabet: {self.alphabet}
Starting state: {self.init_st}
Accepting states: {self.final}
States: {self.states}
Transition function:"""
        for src in self.states:
            for sigma in powerset(self.alphabet):
                dst = self.delta(src, sigma)
                output += "\n\t" + f"{src} -- {set(sigma)} --> {dst}" if len(sigma) > 0 else \
                    "\n\t" + f"{src} -- {{}} --> {dst}"
        return output

    def _validate_graph(self, graph):
        """
        Checks that:
        (1) The accepting-state set is a subset of the state set.
        (2) The start-state is a member of the state set.
        (3) Every transition returns a member of the state set.
        """
        # TODO. Implement function.
        return True

    def construct_symbolic(self, states, alphabet, delta, pre, post, init_st, final, acc):
        if self._is_initialized:
            raise RuntimeError("Cannot initialize an already initialized game.")

        assert isinstance(final, set)

        self._graph = nx.MultiDiGraph()
        self._graph.add_nodes_from(states)
        self._alphabet = alphabet
        self._delta = delta
        self._pre = pre
        self._post = post
        self._init_st = init_st
        self._final = final
        self._mode = self.SYMBOLIC
        self._is_initialized = True

    def construct_explicit(self, graph, init_st, final, acc=ACC_BUCHI):
        if self._is_initialized:
            raise RuntimeError("Cannot initialize an already initialized game.")

        def _delta(state, true_atoms):
            # Construct evaluation map
            eval_map = dict()
            for atom in self.alphabet:
                if atom in true_atoms:
                    eval_map[atom] = True
                else:
                    eval_map[atom] = False

            # Get successors
            succ = [v for v, f in self.post(state) if f.evaluate(eval_map)]
            if len(succ) > 0:
                return set(succ)
            else:   # len(succ) == 0:
                return ValueError("Automaton transition function is incomplete.")

        def _pre(state):
            return {(e[0], e[2]["formula"]) for e in self.graph.in_edges(state, data=True)}

        def _post(state):
            return {(e[1], e[2]["formula"]) for e in self.graph.out_edges(state, data=True)}

        if not self._validate_graph(graph):
            raise ValueError("Graph validation failed.")

        assert isinstance(final, set)

        alphabet = set()
        for _, _, data in graph.edges(data=True):
            alphabet.update(data["formula"].alphabet)

        self._graph = graph
        self._alphabet = alphabet
        self._delta = _delta
        self._pre = _pre
        self._post = _post
        self._init_st = init_st
        self._final = final
        self._mode = self.EXPLICIT
        self._is_initialized = True

    def delta(self, u, true_atoms):
        assert set(true_atoms).issubset(self.alphabet), f"true_atoms must be subset of alphabet."
        if not self._is_initialized:
            raise ValueError(f"Cannot access delta function of 'uninitialized' DFA.")
        return self._delta(u, true_atoms)

    def pre(self, v):
        if not self._is_initialized:
            raise ValueError(f"Cannot access pre function of 'uninitialized' DFA.")
        return self._pre(v)

    def post(self, u):
        if not self._is_initialized:
            raise ValueError(f"Cannot access post function of 'uninitialized' DFA.")
        return self._post(u)


class PrefDfa(BaseAutomaton):
    FROM_DFA = "From DFA"

    def __init__(self):
        _final: set
        super(PrefDfa, self).__init__()
        self._acc = ACC_PREFERENCE

    def __str__(self):
        output = f"""This Pref-DFA has {len(self.states)} states.
Alphabet: {self.alphabet}
Starting state: {self.init_st}
Accepting states: {self.final}
States: {self.states}
Transition function:"""
        for src in self.states:
            for sigma in powerset(self.alphabet):
                dst = self.delta(src, sigma)
                output += "\n\t" + f"{src} -- {set(sigma)} --> {dst}" if len(sigma) > 0 else \
                    "\n\t" + f"{src} -- {{}} --> {dst}"
        output += "\nPreference Graph: "
        output += "\nStates: "
        for state in self.final.nodes(data=True):
            output += f"\n\t{state}"
        output += "\nEdges: "
        for edge in self.final.edges(data=True):
            output += f"\n\t{edge}"
        return output

    def _validate_graph(self, graph):
        """
        Checks that:
        (1) The accepting-state set is a subset of the state set.
        (2) The start-state is a member of the state set.
        (3) Every transition returns a member of the state set.
        """
        # TODO. Implement function.
        return True

    def construct_from_dfa(self, dfa, init_st, final, acc=ACC_PREFERENCE):
        if self._is_initialized:
            raise RuntimeError("Cannot initialize an already initialized game.")

        assert isinstance(final, (nx.DiGraph, nx.MultiDiGraph)), "final must be a nx.DiGraph or nx.MultiDiGraph object."

        self._graph = nx.MultiDiGraph()
        self._graph.add_nodes_from(dfa.states)
        self._alphabet = dfa.alphabet
        self._delta = dfa.delta
        self._pre = dfa.pre
        self._post = dfa.post
        self._init_st = init_st
        self._final = final
        self._mode = self.FROM_DFA
        self._is_initialized = True

    def construct_symbolic(self, states, alphabet, delta, pre, post, init_st, final, acc=ACC_PREFERENCE):
        if self._is_initialized:
            raise RuntimeError("Cannot initialize an already initialized game.")

        assert isinstance(final, (nx.DiGraph, nx.MultiDiGraph)), "final must be a nx.DiGraph or nx.MultiDiGraph object."

        self._graph = nx.MultiDiGraph()
        self._graph.add_nodes_from(states)
        self._alphabet = alphabet
        self._delta = delta
        self._pre = pre
        self._post = post
        self._init_st = init_st
        self._final = final
        self._mode = self.SYMBOLIC
        self._is_initialized = True

    def construct_explicit(self, graph, init_st, final, acc=ACC_PREFERENCE):
        if self._is_initialized:
            raise RuntimeError("Cannot initialize an already initialized game.")

        def _delta(state, true_atoms):
            # Construct evaluation map
            eval_map = dict()
            for atom in self.alphabet:
                if atom in true_atoms:
                    eval_map[atom] = True
                else:
                    eval_map[atom] = False

            # Get successors
            succ = [v for v, f in self.post(state) if f.evaluate(eval_map)]
            if len(succ) > 0:
                return set(succ)
            else:   # len(succ) == 0:
                return ValueError("Automaton transition function is incomplete.")

        def _pre(state):
            return {(e[0], e[2]["formula"]) for e in self.graph.in_edges(state, data=True)}

        def _post(state):
            return {(e[1], e[2]["formula"]) for e in self.graph.out_edges(state, data=True)}

        if not self._validate_graph(graph):
            raise ValueError("Graph validation failed.")

        assert isinstance(final, (nx.DiGraph, nx.MultiDiGraph))

        alphabet = set()
        for _, _, data in graph.edges(data=True):
            alphabet.update(data["formula"].alphabet)

        self._graph = graph
        self._alphabet = alphabet
        self._delta = _delta
        self._pre = _pre
        self._post = _post
        self._init_st = init_st
        self._final = final
        self._mode = self.EXPLICIT
        self._is_initialized = True

    def delta(self, u, true_atoms):
        assert set(true_atoms).issubset(self.alphabet), f"true_atoms must be subset of alphabet."
        if not self._is_initialized:
            raise ValueError(f"Cannot access delta function of 'uninitialized' DFA.")
        return self._delta(u, true_atoms)

    def pre(self, v):
        if not self._is_initialized:
            raise ValueError(f"Cannot access pre function of 'uninitialized' DFA.")
        return self._pre(v)

    def post(self, u):
        if not self._is_initialized:
            raise ValueError(f"Cannot access post function of 'uninitialized' DFA.")
        return self._post(u)


def cross_product(aut1, aut2):
    if isinstance(aut1, Dfa) and isinstance(aut2, Dfa):
        return cross_product_dfa_dfa(aut1, aut2)
    elif isinstance(aut1, PrefDfa) and isinstance(aut2, PrefDfa):
        return cross_product_prefdfa_prefdfa(aut1, aut2)
    else:
        raise ValueError(f"Cross product of type aut1: {type(aut1)} and aut2: {type(aut2)} is not implemented.")


def cross_product_dfa_dfa(dfa1, dfa2):
    if dfa1.alphabet != dfa2.alphabet:
        raise ValueError("Cross product of DFA's with different alphabet is not implemented.")

    states = itertools.product(dfa1.states, dfa2.states)
    final = set(itertools.product(dfa1.final, dfa2.final))
    init_st = (dfa1.init_st, dfa2.init_st)
    alphabet = copy.deepcopy(dfa1.alphabet)

    def delta(state, true_atoms):
        nstate1 = dfa1.delta(state[0], true_atoms)
        nstate2 = dfa2.delta(state[1], true_atoms)
        return nstate1, nstate2

    def post(state):
        next_states = set()
        for sigma in powerset(alphabet):
            nstate1 = dfa1.delta(state[0], sigma)
            nstate2 = dfa2.delta(state[1], sigma)
            next_states.add(((nstate1, nstate2), sigma))
        return next_states

    def pre(state):
        pre_states = set()
        pre1 = list(dfa1.pre(state[0]))
        pre2 = list(dfa2.pre(state[1]))

        for sigma in powerset(alphabet):
            eval1 = [f.evaluate({sym: sym in sigma for sym in alphabet}) for _, f in pre1]
            eval2 = [f.evaluate({sym: sym in sigma for sym in alphabet}) for _, f in pre2]

            # FIXME (IndexError): Logic is incorrect. pre1, pre2 may have different lengths.
            for idx1 in range(len(eval1)):
                for idx2 in range(len(eval2)):
                    if eval1[idx1] is True and eval2[idx2] is True:
                        pre_states.add(((pre1[idx1][0], pre2[idx2][0]), sigma))

        return pre_states

    prod_dfa = Dfa()
    prod_dfa.construct_symbolic(states=states, alphabet=alphabet,
                                delta=delta, pre=pre, post=post,
                                init_st=init_st, final=final)

    return prod_dfa


def cross_product_prefdfa_prefdfa(dfa1, dfa2):
    if dfa1.alphabet != dfa2.alphabet:
        raise ValueError("Cross product of DFA's with different alphabet is not implemented.")

    states = itertools.product(dfa1.states, dfa2.states)
    init_st = (dfa1.init_st, dfa2.init_st)
    alphabet = copy.deepcopy(dfa1.alphabet)

    def delta(state, true_atoms):
        nstate1 = dfa1.delta(state[0], true_atoms)
        nstate2 = dfa2.delta(state[1], true_atoms)
        return nstate1, nstate2

    def post(state):
        next_states = set()
        for sigma in powerset(alphabet):
            nstate1 = dfa1.delta(state[0], sigma)
            nstate2 = dfa2.delta(state[1], sigma)
            next_states.add(((nstate1, nstate2), sigma))
        return next_states

    def pre(state):
        pre_states = set()
        pre1 = list(dfa1.pre(state[0]))
        pre2 = list(dfa2.pre(state[1]))

        for sigma in powerset(alphabet):
            eval1 = [f.evaluate(sigma) for _, f in pre1]
            eval2 = [f.evaluate(sigma) for _, f in pre2]

            for idx in range(len(eval1)):
                if eval1[idx] == eval2[idx] == True:
                    pre_states.add(((pre1[idx][0], pre2[idx][0]), sigma))

        return pre_states

    prod_dfa = PrefDfa()
    prod_dfa.construct_symbolic(states=states, alphabet=alphabet,
                                delta=delta, pre=pre, post=post,
                                init_st=init_st, final=nx.MultiDiGraph())

    return prod_dfa
