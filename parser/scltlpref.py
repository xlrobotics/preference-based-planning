"""
File: scltlpref.py
Project: ggsolver_nx (https://github.com/abhibp1993/ggsolver/tree/nx-solvers)

Description: TODO.
"""

import itertools
import networkx as nx
import spot

from pathlib import Path
from lark import Lark, Transformer, Tree, Visitor
from formula import BaseFormula, CUR_DIR
from scltl import ScLTLFormula
from automata import PrefDfa, cross_product


class ScLTLPrefParser:
    """PrefScLTL Parser class."""

    def __init__(self):
        """Initialize."""
        self._parser = Lark(open(str(Path(CUR_DIR, "scltlpref.lark"))), parser="lalr")

    def __call__(self, text):
        """Call."""
        parse_tree = self._parser.parse(text)
        return parse_tree


class PrefOr:
    def __init__(self, lformula, rformula) -> None:
        self.lformula = lformula
        self.rformula = rformula

    def __str__(self) -> str:
        return f"({self.lformula}) | ({self.rformula})"

    def translate(self):
        # Construct component DFAs
        pref_dfa1 = self.lformula.translate()
        pref_dfa2 = self.rformula.translate()

        # Force alphabet to be equal in order to ensure DFA product is well-defined.
        alphabet = set.union(self.lformula.alphabet, self.rformula.alphabet)
        pref_dfa1._alphabet = alphabet
        pref_dfa2._alphabet = alphabet

        # Construct product DFA
        product_dfa = cross_product(pref_dfa1, pref_dfa2)

        # Construct preference graph
        pref_product_graph = or_product_pref_graph(pref_dfa1.final, pref_dfa2.final)

        # Construct preference DFA
        pref_dfa = PrefDfa()
        pref_dfa.construct_from_dfa(dfa=product_dfa, init_st=product_dfa.init_st, final=pref_product_graph)

        return pref_dfa

    @property
    def alphabet(self):
        return set.union(self.lformula.alphabet, self.rformula.alphabet)


class PrefAnd:
    def __init__(self, lformula, rformula) -> None:
        self.lformula = lformula
        self.rformula = rformula

    def __str__(self) -> str:
        return f"({self.lformula}) & ({self.rformula})"

    def translate(self):
        # Construct component DFAs
        pref_dfa1 = self.lformula.translate()
        pref_dfa2 = self.rformula.translate()

        # Force alphabet to be equal in order to ensure DFA product is well-defined.
        alphabet = set.union(self.lformula.alphabet, self.rformula.alphabet)
        pref_dfa1._alphabet = alphabet
        pref_dfa2._alphabet = alphabet

        # Construct product DFA
        product_dfa = cross_product(pref_dfa1, pref_dfa2)

        # Construct preference graph
        pref_product_graph = and_product_pref_graph(pref_dfa1.final, pref_dfa2.final)

        # Construct preference DFA
        pref_dfa = PrefDfa()
        pref_dfa.construct_from_dfa(dfa=product_dfa, init_st=product_dfa.init_st, final=pref_product_graph)

        return pref_dfa

    @property
    def alphabet(self):
        return set.union(self.lformula.alphabet, self.rformula.alphabet)


class StrictPreference:
    def __init__(self, lformula, rformula):
        self.lformula = lformula
        self.rformula = rformula

    def __str__(self) -> str:
        return f"{self.lformula} > {self.rformula}"

    def translate(self):
        # Construct component DFAs
        dfa1 = self.lformula.translate()
        dfa2 = self.rformula.translate()

        # Force alphabet to be equal in order to ensure DFA product is well-defined.
        alphabet = set.union(self.lformula.alphabet, self.rformula.alphabet)
        dfa1._alphabet = alphabet
        dfa2._alphabet = alphabet

        # Construct product DFA
        product_dfa = cross_product(dfa1, dfa2)

        # Construct preference graph
        pref_graph = nx.MultiDiGraph()
        x1_states = set(itertools.product(dfa1.final, set(dfa2.states) - set(dfa2.final)))
        x2_states = set(itertools.product(set(dfa1.states) - set(dfa1.final), dfa2.final))
        x3_states = set(product_dfa.states) - set.union(x1_states, x2_states)

        pref_graph.add_nodes_from([
            ("X1", {"states": x1_states}),
            ("X2", {"states": x2_states}),
            ("X3", {"states": x3_states})
        ])
        pref_graph.add_edges_from([
            ("X2", "X1")
        ])

        # Construct preference DFA
        pref_dfa = PrefDfa()
        pref_dfa.construct_from_dfa(dfa=product_dfa, init_st=(dfa1.init_st, dfa2.init_st), final=pref_graph)

        # print()
        # print("DEBUG: StrictPreference.translate")
        # print(f"DFA ({self.lformula})")
        # print(dfa1)
        # print(f"DFA ({self.rformula})")
        # print(dfa2)
        # print(f"Pref-DFA ({self})")
        # print(pref_dfa)

        return pref_dfa

    @property
    def alphabet(self):
        return set.union(self.lformula.alphabet, self.rformula.alphabet)


class WeakPreference:
    def __init__(self, lformula, rformula):
        self.lformula = lformula
        self.rformula = rformula

    def __str__(self) -> str:
        return f"{self.lformula} >= {self.rformula}"

    def translate(self):
        # Construct component DFAs
        dfa1 = self.lformula.translate()
        dfa2 = self.rformula.translate()

        # Force alphabet to be equal in order to ensure DFA product is well-defined.
        alphabet = set.union(self.lformula.alphabet, self.rformula.alphabet)
        dfa1._alphabet = alphabet
        dfa2._alphabet = alphabet

        # Construct product DFA
        product_dfa = cross_product(dfa1, dfa2)

        # Construct preference graph
        pref_graph = nx.MultiDiGraph()
        x1_states = set(itertools.product(dfa1.final, dfa2.states))
        x2_states = set(itertools.product(set(dfa1.states) - set(dfa1.final), dfa2.final))
        x3_states = set(product_dfa.states) - set.union(x1_states, x2_states)

        pref_graph.add_nodes_from([
            ("X1", {"states": x1_states}),
            ("X2", {"states": x2_states}),
            ("X3", {"states": x3_states})
        ])
        pref_graph.add_edges_from([
            ("X2", "X1")
        ])

        # Construct preference DFA
        pref_dfa = PrefDfa()
        pref_dfa.construct_from_dfa(dfa=product_dfa, init_st=(dfa1.init_st, dfa2.init_st), final=pref_graph)

        return pref_dfa

    @property
    def alphabet(self):
        return set.union(self.lformula.alphabet, self.rformula.alphabet)


class ScLTLPrefTransformer(Transformer):
    def __init__(self):
        super(ScLTLPrefTransformer, self).__init__()
        self._alphabet = None

    def __call__(self, tree, alphabet=None):
        self._alphabet = alphabet
        return self.transform(tree)

    def start(self, args):
        """Entry point."""
        # print(f"(t) start: {args}")
        if type(args[0]) in [StrictPreference, WeakPreference, PrefAnd, PrefOr]:
            return args[0]
        elif type(args[0]) == Tree:
            f_str = TOSTRING_ScLTL(args[0])
            return ScLTLFormula(f_str)
        else:
            raise ValueError("Trouble")

    def prefltl_strictpref(self, args):
        """ Parse strict preferences. """
        lformula = TOSTRING_ScLTL(args[0])
        rformula = TOSTRING_ScLTL(args[2])
        return StrictPreference(ScLTLFormula(lformula, self._alphabet), ScLTLFormula(rformula, self._alphabet))

    def prefltl_nonstrictpref(self, args):
        """ Parse strict preferences. """
        lformula = TOSTRING_ScLTL(args[0])
        rformula = TOSTRING_ScLTL(args[2])
        return WeakPreference(ScLTLFormula(lformula, self._alphabet), ScLTLFormula(rformula, self._alphabet))

    def pref_or(self, args):
        return PrefOr(args[0].children[1], args[2].children[1])

    def pref_and(self, args):
        return PrefAnd(args[0].children[1], args[2].children[1])


class ScLTLPrefSubstitutor(Transformer):

    def __init__(self):
        super(ScLTLPrefSubstitutor, self).__init__()
        self.eval_map = None

    def __call__(self, tree, eval_map):
        return self.substitute(tree, eval_map)

    def substitute(self, tree, eval_map):
        self.eval_map = eval_map
        return super(ScLTLPrefSubstitutor, self).transform(tree)

    def start(self, args):
        """Entry point."""
        return str(args[0])

    def ltl_formula(self, args):
        """Parse LTLf formula."""
        assert len(args) == 1
        return str(args[0])

    def ltl_equivalence(self, args):
        """Parse LTLf Equivalence."""
        if len(args) == 1:
            return str(str(args[0]))
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") <-> (".join(subformulas) + ")"
        else:
            raise ValueError("Parsing Error")

    def ltl_implication(self, args):
        """Parse LTLf Implication."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") -> (".join(subformulas) + ")"
        else:
            raise ValueError("Parsing Error")

    def ltl_or(self, args):
        """Parse LTLf Or."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") | (".join(subformulas) + ")"
        else:
            raise ValueError("Parsing Error")

    def ltl_and(self, args):
        """Parse LTLf And."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") & (".join(subformulas) + ")"
        else:
            raise ValueError("Parsing Error")

    def ltl_until(self, args):
        """Parse LTLf Until."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") U (".join(subformulas) + ")"
        else:
            raise ValueError("Parsing Error")

    def ltl_release(self, args):
        """Parse LTLf Release."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") R (".join(subformulas) + ")"
        else:
            raise ValueError("Parsing Error")

    def ltl_always(self, args):
        """Parse LTLf Always."""
        if len(args) == 1:
            return str(args[0])
        else:
            f = str(args[-1])
            for _ in args[:-1]:
                f = f"G({f})"
            return f

    def ltl_eventually(self, args):
        """Parse LTLf Eventually."""
        if len(args) == 1:
            return str(args[0])
        else:
            f = str(args[-1])
            for _ in args[:-1]:
                f = f"F({f})"
            return f

    def ltl_next(self, args):
        """Parse LTLf Next."""
        if len(args) == 1:
            return str(args[0])
        else:
            f = str(args[-1])
            for _ in args[:-1]:
                f = f"X({f})"
            return f

    def ltl_not(self, args):
        """Parse LTLf Not."""
        if len(args) == 1:
            return str(args[0])
        else:
            f = str(args[-1])
            for _ in args[:-1]:
                f = f"!({f})"
            return f

    def ltl_wrapped(self, args):
        """Parse LTLf wrapped formula."""
        if len(args) == 1:
            return str(args[0])
        elif len(args) == 3:
            _, formula, _ = args
            return str(formula)
        else:
            raise ValueError("Parsing Error")

    def ltl_atom(self, args):
        """Parse LTLf Atom."""
        assert len(args) == 1
        return str(args[0])

    def ltl_ap(self, args):
        """Parse LTLf Atom."""
        assert len(args) == 1
        token = str(args[0])
        try:
            print(str(self.eval_map[token]))
            return str(self.eval_map[token])
        except KeyError:
            return token

    def ltl_true(self, args):
        """Parse LTLf True."""
        return "true"

    def ltl_false(self, args):
        """Parse LTLf False."""
        return "false"


class ScLTLPrefEvaluator(Transformer):
    def __call__(self, tree, f_str, eval_map):
        return self.evaluate(tree, f_str, eval_map)

    def evaluate(self, tree, f_str, eval_map):
        # TODO: Answer the following questions.
        #   If ScLTLPref formula does not include preference operators, evaluation is identical to ScLTL.
        #   Otherwise, when is ScLTLPref formula identically 'true' or 'false'?
        try:
            f_mp_class = str(spot.mp_class(f_str)).lower()
            if "b" in f_mp_class or "g" in f_mp_class:
                val = SUBSTITUTE_ScLTLPref(tree, eval_map)
                f = spot.formula(val)
                if f.is_tt():
                    return True
                elif f.is_ff():
                    return False
                else:
                    raise ValueError(f"ScLTLPref formula '{f_str}' could not be evaluated. "
                                     f"Did you forget to substitute all atoms?")

        except SyntaxError:
            raise NotImplementedError("Theoretical questions need addressing.")


class ScLTLPrefAtomGlobber(Visitor):
    def __init__(self):
        self._alphabet = set()

    def __call__(self, tree):
        self._alphabet = set()
        super(ScLTLPrefAtomGlobber, self).visit(tree)
        return self._alphabet

    def add_alphabet(self, sym):
        self._alphabet.add(sym)

    def ltl_ap(self, args):
        """ Glob Atom."""
        self.add_alphabet(args.children[0].value)


class ScLTLPrefToStringVisitor(Visitor):
    def __init__(self):
        self.strf = ""

    def __call__(self, tree):
        self.strf = ""
        self.visit(tree)
        return self.strf

    def ltl_ap(self, tree):
        # print(f"(v) ltl_ap: {tree}")
        self.strf += str(tree.children[0])
        return tree.children[0]

    def ltl_eventually(self, tree):
        # print(f"(v) ltl_eventually: {tree}")
        self.strf = f"F({self.strf})"
        return tree.children[0]

    def ltl_always(self, tree):
        # print(f"(v) ltl_always: {tree}")
        self.strf = f"G({self.strf})"
        return tree.children[0]

    def ltl_next(self, tree):
        # print(f"(v) ltl_next: {tree}")
        self.strf = f"X({self.strf})"
        return tree.children[0]

    def ltl_not(self, tree):
        # print(f"(v) ltl_not: {tree}")
        self.strf = f"!{self.strf}"
        return tree.children[0]

    def ltl_wrapped(self, tree):
        # print(f"(v) ltl_wrapped: {tree}")
        self.strf = f"({self.strf})"
        return tree.children[1]

    def ltl_or(self, tree):
        # print(f"(v) ltl_or: {tree}")
        lvisitor = ScLTLPrefToStringVisitor()
        rvisitor = ScLTLPrefToStringVisitor()
        lvisitor.visit(tree.children[0])
        rvisitor.visit(tree.children[2])
        self.strf = f"({lvisitor.strf}) | ({rvisitor.strf})"
        return tree.children[1]

    def ltl_and(self, tree):
        # print(f"(v) ltl_wrapped: {tree}")
        lvisitor = ScLTLPrefToStringVisitor()
        rvisitor = ScLTLPrefToStringVisitor()
        lvisitor.visit(tree.children[0])
        rvisitor.visit(tree.children[2])
        self.strf = f"({lvisitor.strf}) & ({rvisitor.strf})"
        return tree.children[1]

    def ltl_until(self, tree):
        # print(f"(v) ltl_wrapped: {tree}")
        lvisitor = ScLTLPrefToStringVisitor()
        rvisitor = ScLTLPrefToStringVisitor()
        lvisitor.visit(tree.children[0])
        rvisitor.visit(tree.children[2])
        self.strf = f"({lvisitor.strf}) U ({rvisitor.strf})"
        return tree.children[1]

    def ltl_true(self, tree):
        self.strf = "true"
        return tree.children[0]

    def ltl_false(self, tree):
        self.strf = "false"
        return tree.children[0]

    def prefltl_strictpref(self, tree):
        """ Parse strict preferences. """
        lvisitor = ScLTLPrefToStringVisitor()
        rvisitor = ScLTLPrefToStringVisitor()
        lvisitor.visit(tree.children[0])
        rvisitor.visit(tree.children[2])
        self.strf = f"({lvisitor.strf}) > ({rvisitor.strf})"
        return tree.children[1]

    def prefltl_nonstrictpref(self, tree):
        """ Parse strict preferences. """
        lvisitor = ScLTLPrefToStringVisitor()
        rvisitor = ScLTLPrefToStringVisitor()
        lvisitor.visit(tree.children[0])
        rvisitor.visit(tree.children[2])
        self.strf = f"({lvisitor.strf}) >= ({rvisitor.strf})"
        return tree.children[1]

    def pref_or(self, tree):
        lvisitor = ScLTLPrefToStringVisitor()
        rvisitor = ScLTLPrefToStringVisitor()
        lvisitor.visit(tree.children[0])
        rvisitor.visit(tree.children[2])
        self.strf = f"({lvisitor.strf}) | ({rvisitor.strf})"
        return tree.children[1]

    def pref_and(self, tree):
        lvisitor = ScLTLPrefToStringVisitor()
        rvisitor = ScLTLPrefToStringVisitor()
        lvisitor.visit(tree.children[0])
        rvisitor.visit(tree.children[2])
        self.strf = f"({lvisitor.strf}) & ({rvisitor.strf})"
        return tree.children[1]


PARSE_ScLTLPref = ScLTLPrefParser()
TRANSFORM_ScLTLPref = ScLTLPrefTransformer()
SUBSTITUTE_ScLTLPref = ScLTLPrefSubstitutor()
EVALUATE_ScLTLPref = ScLTLPrefEvaluator()
ATOMGLOB_ScLTLPref = ScLTLPrefAtomGlobber()
TOSTRING_ScLTL = ScLTLPrefToStringVisitor()


class ScLTLPrefFormula(BaseFormula):
    def __init__(self, f_str, alphabet=None):
        super(ScLTLPrefFormula, self).__init__(f_str, alphabet)
        self.parse_tree = PARSE_ScLTLPref(f_str)
        self.formula = TRANSFORM_ScLTLPref(self.parse_tree, alphabet)

    def __hash__(self):
        return hash(self.f_str)

    def __eq__(self, other):
        return self.formula == other

    def __str__(self):
        return str(self.formula)

    def translate(self):
        return self.formula.translate()

    def substitute(self, subs_map):
        return ScLTLPrefFormula(SUBSTITUTE_ScLTLPref(self.parse_tree, subs_map))

    def evaluate(self, eval_map):
        return EVALUATE_ScLTLPref(self.parse_tree, self.f_str, eval_map)

    @property
    def alphabet(self):
        if self._alphabet is not None:
            return self._alphabet

        return ATOMGLOB_ScLTLPref(self.parse_tree)


def and_product_pref_graph(pref_graph1, pref_graph2):
    states = set(itertools.product(pref_graph1.nodes(), pref_graph2.nodes()))
    edges = set()
    for (x1, x2), (y1, y2) in itertools.product(states, states):
        if (pref_graph1.has_edge(x1, y1) and pref_graph2.has_edge(x2, y2)) or \
                (pref_graph1.has_edge(x1, y1) and x2 == y2) or \
                (x1 == y1 and pref_graph2.has_edge(x2, y2)):
            edges.add(((x1, x2), (y1, y2)))

    prod_graph = nx.MultiDiGraph()
    prod_graph.add_nodes_from(states)
    prod_graph.add_edges_from(edges)
    return prod_graph


def or_product_pref_graph(pref_graph1, pref_graph2):
    states = set(itertools.product(pref_graph1.nodes(), pref_graph2.nodes()))
    edges = set()
    for (x1, x2), (y1, y2) in itertools.product(states, states):
        if (pref_graph1.has_edge(x1, y1) or pref_graph2.has_edge(x2, y2)) and \
                (x1 not in nx.algorithms.dag.descendants(pref_graph1, y1)) and \
                (x2 not in nx.algorithms.dag.descendants(pref_graph1, y2)):
            edges.add(((x1, x2), (y1, y2)))

    prod_graph = nx.MultiDiGraph()
    prod_graph.add_nodes_from(states)
    prod_graph.add_edges_from(edges)
    return prod_graph


if __name__ == '__main__':
    # parser = Lark(grammar, parser='lalr')
    # parser = ScLTLPrefParser()
    while True:
        try:
            s = input("prefltl > ")
        except EOFError:
            break
        if not s:
            continue
        if s == "exit":
            break
        result = ScLTLPrefFormula(s)
        print()
        print(f"Formula: {result} of type {result.__class__.__name__}")
        aut = result.translate()
        print(aut)
