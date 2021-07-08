"""
File: scltl.py
Project: ggsolver_nx (https://github.com/abhibp1993/ggsolver/tree/nx-solvers)

Description: TODO.
"""

import networkx as nx
import spot

from pathlib import Path
from lark import Lark, Transformer, Visitor
from formula import BaseFormula, CUR_DIR
from pl import PLFormula
from automata import Dfa


class ScLTLParser:
    """ LTL Parser class. """

    def __init__(self):
        """Initialize."""
        self._parser = Lark(open(str(Path(CUR_DIR, "scltl.lark"))), parser="lalr")

    def __call__(self, text):
        """Call."""
        parse_tree = self._parser.parse(text)
        return parse_tree


class ScLTLSubstitutor(Transformer):

    def __init__(self):
        super(ScLTLSubstitutor, self).__init__()
        self.eval_map = None

    def __call__(self, tree, eval_map):
        return self.substitute(tree, eval_map)

    def substitute(self, tree, eval_map):
        self.eval_map = eval_map
        return super(ScLTLSubstitutor, self).transform(tree)

    def start(self, args):
        """Entry point."""
        return str(args[0])

    def scltl_formula(self, args):
        """Parse LTLf formula."""
        assert len(args) == 1
        return str(args[0])

    def scltl_equivalence(self, args):
        """Parse LTLf Equivalence."""
        if len(args) == 1:
            return str(str(args[0]))
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") <-> (".join(subformulas) + ")"
        else:
            raise ValueError("Parsing Error")

    def scltl_implication(self, args):
        """Parse LTLf Implication."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") -> (".join(subformulas) + ")"
        else:
            raise ValueError("Parsing Error")

    def scltl_or(self, args):
        """Parse LTLf Or."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") | (".join(subformulas) + ")"
        else:
            raise ValueError("Parsing Error")

    def scltl_and(self, args):
        """Parse LTLf And."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") & (".join(subformulas) + ")"
        else:
            raise ValueError("Parsing Error")

    def scltl_until(self, args):
        """Parse LTLf Until."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") U (".join(subformulas) + ")"
        else:
            raise ValueError("Parsing Error")

    def scltl_release(self, args):
        """Parse LTLf Release."""
        if len(args) == 1:
            return str(args[0])
        elif (len(args) - 1) % 2 == 0:
            subformulas = [str(f) for f in args[::2]]
            return "(" + ") R (".join(subformulas) + ")"
        else:
            raise ValueError("Parsing Error")

    def scltl_always(self, args):
        """Parse LTLf Always."""
        if len(args) == 1:
            return str(args[0])
        else:
            f = str(args[-1])
            for _ in args[:-1]:
                f = f"G({f})"
            return f

    def scltl_eventually(self, args):
        """Parse LTLf Eventually."""
        if len(args) == 1:
            return str(args[0])
        else:
            f = str(args[-1])
            for _ in args[:-1]:
                f = f"F({f})"
            return f

    def scltl_next(self, args):
        """Parse LTLf Next."""
        if len(args) == 1:
            return str(args[0])
        else:
            f = str(args[-1])
            for _ in args[:-1]:
                f = f"X({f})"
            return f

    def scltl_not(self, args):
        """Parse LTLf Not."""
        if len(args) == 1:
            return str(args[0])
        else:
            f = str(args[-1])
            for _ in args[:-1]:
                f = f"!({f})"
            return f

    def scltl_wrapped(self, args):
        """Parse LTLf wrapped formula."""
        if len(args) == 1:
            return str(args[0])
        elif len(args) == 3:
            _, formula, _ = args
            return str(formula)
        else:
            raise ValueError("Parsing Error")

    def scltl_atom(self, args):
        """Parse LTLf Atom."""
        assert len(args) == 1
        return str(args[0])

    def scltl_true(self, args):
        """Parse LTLf True."""
        return "true"

    def scltl_false(self, args):
        """Parse LTLf False."""
        return "false"

    def scltl_symbol(self, args):
        """Parse LTLf Symbol."""
        assert len(args) == 1
        token = str(args[0])
        try:
            print(str(self.eval_map[token]))
            return str(self.eval_map[token])
        except KeyError:
            return token

        # return LTLfAtomic(symbol)


class ScLTLEvaluator(Transformer):
    def __call__(self, tree, f_str, eval_map):
        return self.evaluate(tree, f_str, eval_map)

    def evaluate(self, tree, f_str, eval_map):
        if str(spot.mp_class(f_str)).lower() in ["b", "g"]:
            val = SUBSTITUTE_ScLTL(tree, eval_map)
            f = spot.formula(val)
            if f.is_tt():
                return True
            elif f.is_ff():
                return False

        raise ValueError(f"ScLTL formula '{f_str}' could not be evaluated. "
                         f"Did you forget to substitute all atoms?")


class ScLTLAtomGlobber(Visitor):
    def __init__(self):
        self._alphabet = set()

    def __call__(self, tree):
        self._alphabet = set()
        super(ScLTLAtomGlobber, self).visit(tree)
        return self._alphabet

    def add_alphabet(self, sym):
        self._alphabet.add(sym)

    def scltl_symbol(self, args):
        """ Glob Atom."""
        self.add_alphabet(args.children[0].value)


PARSE_ScLTL = ScLTLParser()
SUBSTITUTE_ScLTL = ScLTLSubstitutor()
EVALUATE_ScLTL = ScLTLEvaluator()
ATOMGLOB_ScLTL = ScLTLAtomGlobber()


class ScLTLFormula(BaseFormula):
    def __init__(self, f_str, alphabet=None):
        super(ScLTLFormula, self).__init__(f_str, alphabet)
        self.parse_tree = PARSE_ScLTL(f_str)

    def __hash__(self):
        return hash(str(spot.formula(self.f_str)))

    def __eq__(self, other):
        if isinstance(other, BaseFormula):
            return spot.formula(self.f_str) == spot.formula(other)
        elif isinstance(other, str):
            return spot.formula(self.f_str) == spot.formula(other)
        raise AssertionError(f"PLFormula.__eq__: type(other)={type(other)}.")

    def __str__(self):
        return str(spot.formula(self.f_str))

    def translate(self):
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
        return ScLTLFormula(SUBSTITUTE_ScLTL(self.parse_tree, subs_map))

    def evaluate(self, eval_map):
        return EVALUATE_ScLTL(self.parse_tree, self.f_str, eval_map)

    @property
    def alphabet(self):
        if self._alphabet is not None:
            return self._alphabet

        return ATOMGLOB_ScLTL(self.parse_tree)


if __name__ == "__main__":
    parser = ScLTLParser()
    while True:
        try:
            s = input("scltl > ")
        except EOFError:
            break

        if s == "exit":
            break

        if not s:
            continue

        formula, t = parser(s)
        print(f"spot.formula({s}) = {formula}.")

        # eval_map = {"a": "true", "b": "false"}
        # evaluator = ScLTLSubstitutor()
        # eStr = evaluator.transform(t, eval_map)
        # print(f"eStr = {eStr}")
        # f, _ = parser(eStr)
        # print(f)
        #
        # e = ScLTLEvaluator()
        # print("e.transform", t, formula, eval_map)
        # print(e.transform(t, formula, eval_map))
        formula = ScLTLFormula(s)
        print(formula.translate())

