"""
File: formula.py
Project: ggsolver_nx (https://github.com/abhibp1993/ggsolver/tree/nx-solvers)

Description: TODO.
"""

import os
import inspect
from abc import ABC, abstractmethod

# Global Variable
CUR_DIR = os.path.dirname(inspect.getfile(inspect.currentframe()))


class BaseFormula(ABC):
    def __init__(self, f_str, alphabet=None):
        self.f_str = f_str
        self.parse_tree = None
        self.str_traverser = None
        self._alphabet = alphabet

    def __str__(self):
        return str(self.f_str)

    @abstractmethod
    def translate(self):
        pass

    @abstractmethod
    def substitute(self, subs_map):
        pass

    @abstractmethod
    def evaluate(self, eval_map):
        pass

    @abstractmethod
    def alphabet(self):
        pass



