from lark import Lark, Transformer, Visitor, Tree
from lark.tree import Tree

try:
    import spot
except ImportError as e:
    spot = None

# Parser for scLTL + Preference operators (>, >=, ~). 
#   scLTL restricts negation to only APs.
#   Negation of preferences is not allowed. 
#   Preferences are binary ops defined over two scLTL formulas.


grammar =  \
"""
start: prefltl_formula


?prefltl_formula:           pref_formula 
            |               ltl_formula


?ltl_formula:               ltl_or
?ltl_or:                    ltl_and (OR ltl_and)*
?ltl_and:                   ltl_until (AND ltl_until)*
?ltl_until:                 ltl_unaryop (UNTIL ltl_unaryop)*


?ltl_unaryop:               ltl_always
            |               ltl_eventually
            |               ltl_next
            |               ltl_weak_next
            |               ltl_not
            |               ltl_wrapped


?ltl_always:                ALWAYS ltl_formula
?ltl_eventually:            EVENTUALLY ltl_formula
?ltl_next:                  NEXT ltl_formula
?ltl_weak_next:             WEAK_NEXT ltl_formula
?ltl_not:                   NOT ltl_atom
?ltl_wrapped:               ltl_atom
            |               LSEPARATOR ltl_formula RSEPARATOR
?ltl_atom:                  ltl_ap
            |               ltl_true
            |               ltl_false


ltl_ap:                     AP_NAME
ltl_true:                   TRUE  
ltl_false:                  FALSE


?pref_formula:              pref_or
?pref_or:                   pref_and (OR pref_and)*
?pref_and:                  pref_base (AND pref_base)*

?pref_base:                 prefltl_strictpref
            |               prefltl_nonstrictpref
            |               prefltl_indifference
            |               prefltl_wrapped

?prefltl_wrapped:           LSEPARATOR pref_formula RSEPARATOR
?prefltl_strictpref:        ltl_formula STRICTPREF ltl_formula
?prefltl_nonstrictpref:     ltl_formula NONSTRICTPREF ltl_formula
?prefltl_indifference:      ltl_formula INDIFFERENCE ltl_formula


AP_NAME:                    /[a-z][a-z0-9_]*/
TRUE.2:                     /(?i:true)/ | "tt"
FALSE.2:                    /(?i:false)/ | "ff"
LSEPARATOR:                 "("
RSEPARATOR:                 ")"
UNTIL.2:                    /U(?=[^a-z]|$)/
RELEASE.2:                  /R(?=[^a-z]|$)/
ALWAYS.2:                   /G(?=[^a-z]|$)/
EVENTUALLY.2:               /F(?=[^a-z]|$)/
NEXT.2:                     /X(?=[^a-z]|$)/
WEAK_NEXT.2:                /WX(?=[^a-z]|$)/
NOT:                        "!" 
OR:                         "||" | "|"
AND:                        "&&" | "&"
STRICTPREF:                 ">" 
NONSTRICTPREF:              ">="
INDIFFERENCE:               "~" 


%ignore /\s+/
"""


class PrefScLTLVisitor(Visitor):
    def __init__(self):
        self.strf = ""

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
        self.strf = f"!({self.strf})" 
        return tree.children[0]
    
    def ltl_wrapped(self, tree):
        # print(f"(v) ltl_wrapped: {tree}")
        self.strf = f"({self.strf})" 
        return tree.children[1]
    
    def ltl_or(self, tree):
        # print(f"(v) ltl_or: {tree}")
        lvisitor = PrefScLTLVisitor()
        rvisitor = PrefScLTLVisitor()
        lvisitor.visit(tree.children[0])
        rvisitor.visit(tree.children[2])
        self.strf = f"({lvisitor.strf}) | ({rvisitor.strf})" 
        return tree.children[1]
    
    def ltl_and(self, tree):
        # print(f"(v) ltl_wrapped: {tree}")
        lvisitor = PrefScLTLVisitor()
        rvisitor = PrefScLTLVisitor()
        lvisitor.visit(tree.children[0])
        rvisitor.visit(tree.children[2])
        self.strf = f"({lvisitor.strf}) & ({rvisitor.strf})" 
        return tree.children[1]
    
    def ltl_until(self, tree):
        # print(f"(v) ltl_wrapped: {tree}")
        lvisitor = PrefScLTLVisitor()
        rvisitor = PrefScLTLVisitor()
        lvisitor.visit(tree.children[0])
        rvisitor.visit(tree.children[2])
        self.strf = f"({lvisitor.strf}) U ({rvisitor.strf})" 
        return tree.children[1]

    
class StrictPreference:
    def __init__(self, lformula, rformula):
        self.lformula = lformula
        self.rformula = rformula

    def __str__(self) -> str:
        return f"{self.lformula} > {self.rformula}"


class NonStrictPreference:
    def __init__(self, lformula, rformula):
        self.lformula = lformula
        self.rformula = rformula

    def __str__(self) -> str:
        return f"{self.lformula} >= {self.rformula}"


class IndifferentPreference:
    def __init__(self, lformula, rformula):
        self.lformula = lformula
        self.rformula = rformula

    def __str__(self) -> str:
        return f"{self.lformula} ~ {self.rformula}"


class PrefOr:
    def __init__(self, lpref, rpref) -> None:
        self.lpref = lpref
        self.rpref = rpref


class PrefAnd:
    def __init__(self, lpref, rpref) -> None:
        self.lpref = lpref
        self.rpref = rpref


class ScLTLFormula:
    def __init__(self, scltlformula) -> None:
        self.formula = spot.formula(scltlformula)
        if not spot.mp_class(self.formula, "v") in ["guarantee", "guarantee safety", "bottom"]:
            raise TypeError(f"spot.mp_class(scltlformula) = '{spot.mp_class(self.formula, 'v')}'. Expected 'guarantee'.")
    
    def __str__(self) -> str:
        return str(self.formula)


class PrefScLTLTransformer(Transformer):

    def start(self, args):
        """Entry point."""
        # print(f"(t) start: {args}")
        if type(args[0]) in [StrictPreference, NonStrictPreference, IndifferentPreference, PrefAnd, PrefOr]:
            return args[0]
        elif type(args[0]) == Tree:
            visitor = PrefScLTLVisitor()
            _ = visitor.visit(args[0])
            return ScLTLFormula(visitor.strf)
        else:
            raise ValueError("Trouble")

    def prefltl_strictpref(self, args):
        """ Parse strict preferences. """
        lvisitor = PrefScLTLVisitor()
        rvisitor = PrefScLTLVisitor()

        lhs = lvisitor.visit(args[0])
        rhs = rvisitor.visit(args[2])
        
        # print(f"(t) prefltl_strictpref: {lvisitor.strf} > {rvisitor.strf}")
        return StrictPreference(ScLTLFormula(lvisitor.strf), ScLTLFormula(rvisitor.strf))

    def prefltl_nonstrictpref(self, args):
        """ Parse strict preferences. """
        lvisitor = PrefScLTLVisitor()
        rvisitor = PrefScLTLVisitor()

        lhs = lvisitor.visit(args[0])
        rhs = rvisitor.visit(args[2])
        
        # print(f"(t) prefltl_strictpref: {lvisitor.strf} >= {rvisitor.strf}")
        return NonStrictPreference(ScLTLFormula(lvisitor.strf), ScLTLFormula(rvisitor.strf))
    
    def prefltl_indifference(self, args):
        """ Parse strict preferences. """
        lvisitor = PrefScLTLVisitor()
        rvisitor = PrefScLTLVisitor()

        lhs = lvisitor.visit(args[0])
        rhs = rvisitor.visit(args[2])
        
        # print(f"(t) prefltl_strictpref: {lvisitor.strf} ~ {rvisitor.strf}")
        return IndifferentPreference(ScLTLFormula(lvisitor.strf), ScLTLFormula(rvisitor.strf))
    
    def pref_or(self, args):
        ltransformer = PrefScLTLTransformer()
        rtransformer = PrefScLTLTransformer()

        lhs = ltransformer.transform(args[0])
        rhs = rtransformer.transform(args[2])
        
        # print(f"(t) prefltl_strictpref: {lhs} ~ {rhs}")
        return PrefOr(lhs, rhs)

    def pref_and(self, args):
        ltransformer = PrefScLTLTransformer()
        rtransformer = PrefScLTLTransformer()

        lhs = ltransformer.transform(args[0])
        rhs = rtransformer.transform(args[2])
        
        # print(f"(t) prefltl_strictpref: {lhs} ~ {rhs}")
        return PrefAnd(lhs, rhs)


class PrefScLTLParser:
    """PrefScLTL Parser class."""

    def __init__(self):
        """Initialize."""
        self._transformer = PrefScLTLTransformer()
        self._parser = Lark(grammar, parser="lalr")

    def __call__(self, text):
        """Call."""
        tree = self._parser.parse(text)
        formula = self._transformer.transform(tree)
        return formula



if __name__ == "__main__":
    # parser = Lark(grammar, parser='lalr')
    parser = PrefScLTLParser()
    while True:
        try:
            s = input("prefltl > ")
        except EOFError:
            break
        if not s:
            continue
        if s == "exit":
            break
        result = parser(s)
        print(f"Formula: {result} of type {result.__class__.__name__}")