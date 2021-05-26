from lark import Lark

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
?ltl_atom:                  ltl_symbol
            |               ltl_true
            |               ltl_false


ltl_symbol:                 AP_NAME
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
?prefltl_strictpref:        ltl_formula (STRICTPREF ltl_formula)+
?prefltl_nonstrictpref:     ltl_formula (NONSTRICTPREF ltl_formula)+
?prefltl_indifference:      ltl_formula (INDIFFERENCE ltl_formula)+


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

parser = Lark(grammar, parser='lalr')

if __name__ == "__main__":
    # parser = PLParser()
    while True:
        try:
            s = input("prefltl > ")
        except EOFError:
            break
        if not s:
            continue
        result = parser.parse(s)
        print(result)