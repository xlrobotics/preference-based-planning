"""
File: pl.py
Project: ggsolver_nx (https://github.com/abhibp1993/ggsolver/tree/nx-solvers)

Description: TODO.
"""

import networkx as nx
import spot

from pathlib import Path
from lark import Lark, Transformer, Visitor
from formula import BaseFormula, CUR_DIR
from automata import Dfa


class PLParser:
    """
    Parser for Propositional Logic formula.
    """

    def __init__(self):
        """Initialize."""
        self._parser = Lark(open(str(Path(CUR_DIR, "pl.lark"))), parser="lalr")

    def __call__(self, text):
        """ Parse the given text. """
        parse_tree = self._parser.parse(text)
        return parse_tree
        # spot_formula = spot.formula(text)

        # if str(spot.mp_class(spot_formula)).lower() in ["b"]:
        #     return spot_formula, tree
        #
        # raise ValueError(f"Given formula {text} is not an scLTL formula. "
        #                  f"It belongs to '{spot.mp_class(spot_formula, 'v')}' class.")


class PLSubstitutor(Transformer):
    """PL Transformer."""

    def __init__(self):
        super(PLSubstitutor, self).__init__()
        self.eval_map = None

    def __call__(self, tree, eval_map):
        return self.substitute(tree, eval_map)

    def substitute(self, tree, eval_map):
        self.eval_map = eval_map
        return super(PLSubstitutor, self).transform(tree)

    def start(self, args):
        """Entry point."""
        return str(args[0])

    def propositional_formula(self, args):
        """Parse Propositional formula."""
        assert len(args) == 1
        return str(args[0])

    def prop_equivalence(self, args):
        """Parse Propositional Equivalence."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") <-> (".join(subformulas) + ")"
        else:
            raise ValueError("parsing error")

    def prop_implication(self, args):
        """Parse Propositional Implication."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") -> (".join(subformulas) + ")"
        else:
            raise ValueError("parsing error")

    def prop_or(self, args):
        """Parse Propositional Or."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") | (".join(subformulas) + ")"
        else:
            raise ValueError("parsing error")

    def prop_and(self, args):
        """Parse Propositional And."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "("+ ") & (".join(subformulas) + ")"
        else:
            raise ValueError("Propositional AND didn't work out.")

    def prop_not(self, args):
        """Parse Propositional Not."""
        if len(args) == 1:
            return str(args[0])
        else:
            f = str(args[-1])
            for _ in args[:-1]:
                f = f"!({f})"
            return f

    def prop_wrapped(self, args):
        """Parse Propositional wrapped formula."""
        if len(args) == 1:
            return str(args[0])
        elif len(args) == 3:
            _, f, _ = args
            return str(f)
        else:
            raise ValueError("parsing error")

    def prop_atom(self, args):
        """Parse Propositional Atom."""
        assert len(args) == 1
        return str(args[0])

    def prop_true(self, args):
        """Parse Propositional True."""
        assert len(args) == 1
        return "true"

    def prop_false(self, args):
        """Parse Propositional False."""
        assert len(args) == 1
        return "false"

    def atom(self, args):
        """Parse Atom."""
        assert len(args) == 1
        try:
            return str(self.eval_map[args[0]])
        except KeyError:
            return args[0]


class PLEvaluator(Transformer):
    """PL Transformer."""
    def __call__(self, tree, f_str, eval_map):
        return self.evaluate(tree, f_str, eval_map)

    def evaluate(self, tree, f_str, eval_map):
        if str(spot.mp_class(f_str)).lower() in ["b", "bottom"]:
            val = SUBSTITUTE_PL(tree, eval_map)
            f = spot.formula(val)
            if f.is_tt():
                return True
            elif f.is_ff():
                return False

        raise ValueError(f"PL formula '{f_str}' could not be evaluated. "
                         f"Did you forget to substitute all atoms?\n"
                         f"eval_map given is {eval_map}")


class PLAtomGlobber(Visitor):
    def __init__(self):
        self._alphabet = set()

    def __call__(self, tree):
        self._alphabet = set()
        super(PLAtomGlobber, self).visit(tree)
        return self._alphabet

    def add_alphabet(self, sym):
        self._alphabet.add(sym)

    def atom(self, args):
        """ Glob Atom."""
        self.add_alphabet(args.children[0].value)


PARSE_PL = PLParser()
SUBSTITUTE_PL = PLSubstitutor()
EVALUATE_PL = PLEvaluator()
ATOMGLOB_PL = PLAtomGlobber()


class PLFormula(BaseFormula):
    """
    Represents a propositional logic formula.

    Internally, `spot` module is used to translate the PL formula to automaton.
    """
    def __init__(self, f_str, alphabet=None):
        super(PLFormula, self).__init__(f_str, alphabet)
        self.parse_tree = PARSE_PL(f_str)

    def __hash__(self):
        return hash(str(spot.formula(self.f_str)))

    def __eq__(self, other):
        if isinstance(other, BaseFormula):
            return spot.formula(self.f_str) == spot.formula(other)
        elif isinstance(other, str):
            return spot.formula(self.f_str) == spot.formula(other)
        raise AssertionError(f"PLFormula.__eq__: type(other)={type(other)}.")

    def translate(self):
        """
        Constructs a DFA accepting language defined by PL formula.
        :return: Dfa object.
        """
        # Generate spot automaton
        aut = spot.translate(self.f_str, "BA", "High", "SBAcc", "Complete")
        bdict = aut.get_dict()

        # Convert spot automaton to Dfa object
        #   We will use explicit construction: define a graph and use it to construct Dfa object.
        graph = nx.MultiDiGraph()
        init_st = int(aut.get_init_state_number())
        final = set()
        for src in range(0, aut.num_states()):
            graph.add_node(int(src))
            for edge in aut.out(src):
                f_lbl_str = str(spot.bdd_format_formula(bdict, edge.cond))
                if f_lbl_str == '1':
                    f_lbl_str = "true"
                elif f_lbl_str == '0':
                    f_lbl_str = "false"
                f_lbl = PLFormula(f_lbl_str)
                graph.add_edge(int(edge.src), int(edge.dst), formula=f_lbl)

                # Final state is the source of accepting edge.
                #   See: `G(p1 | p2 | F!p0)` in spot app by toggling `force transition-based` option.
                #   Observe edge from 0 -> 1.
                if edge.acc.count() > 0:
                    final.add(int(edge.src))

        dfa = Dfa()
        dfa.construct_explicit(graph=graph, init_st=init_st, final=final)
        return dfa

    def mp_class(self, verbose=False):
        return spot.mp_class(spot.formula(self.f_str), "v" if verbose else "")

    def substitute(self, subs_map):
        return PLFormula(SUBSTITUTE_PL(self.parse_tree, subs_map))

    def evaluate(self, eval_map):
        return EVALUATE_PL(self.parse_tree, self.f_str, eval_map)

    @property
    def alphabet(self):
        if self._alphabet is not None:
            return self._alphabet

        return ATOMGLOB_PL(self.parse_tree)
