"""
Provides functionality to translate LTL formula to Automaton.
- Supports scLTL formulas only. 
- 
"""

import spot
import networkx as nx


def translate(f):
    # Check if formula is scLTL formula.
    if not spot.mp_class(f, 'v') in ["guarantee", "b", "bottom"]:
       raise ValueError(f"Input 'f' must be of bottom or guarantee type. Given f='{f}' is '{spot.mp_class(f, 'v')}' type.") 

    # Translate to spot automaton
    spotaut =  spot.translate(f, "BA", "High", "SBAcc", "Complete")
    bdict = spotaut.get_dict()

    # Convert spot.automaton to networkx digraph
    aut = nx.MultiDiGraph()
    aut.graph["acceptance"] = str(spotaut.get_acceptance())
    aut.graph["numAccSets"] = int(spotaut.num_sets())
    aut.graph["numStates"] = int(spotaut.num_states())
    aut.graph["initStates"] = [int(spotaut.get_init_state_number())]
    aut.graph["apNames"] = [str(ap) for ap in spotaut.ap()]
    aut.graph["formula"] = str(spotaut.get_name())
    aut.graph["isDeterministic"] = bool(spotaut.prop_universal() and spotaut.is_existential())
    aut.graph["isTerminal"] = bool(spotaut.prop_terminal())
    aut.graph["hasStateBasedAcc"] = bool(spotaut.prop_state_acc())

    states = []
    edges = []
    for src in range(0, spotaut.num_states()):
        aut.add_node(int(src))
        for edge in spotaut.out(src):
            e = aut.add_edge(int(edge.src), int(edge.dst), label=str(spot.bdd_format_formula(bdict, edge.cond)))
            aut.nodes[int(src)]["isAcc"] = not (edge.acc is None)
                
    return aut


