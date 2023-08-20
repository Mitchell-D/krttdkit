""" Abstract class that collects attributes of arbitrary array recipes.  """

from typing import Callable

class Recipe:
    def __init__(self, args:list, func, name:str="", desc:tuple=None,
                 ref:str=""):
        """
        Abstract class for a scalar array or RGB recipe based on text or
        integer labels.

        :@param args: List of integer or string labels for data ingredients
            for this Recipe
        :@param func: function with positional arguments corresponding to
            the datasets labeled by args.
        :@param name: Readable text name of this Recipe
        :@param desc: Arbitrary description attribute; usually I use this
            for a tuple string description of each argument.
        :@param ref: DOI or link reference to source of Recipe
        """
        if not func.__code__.co_argcount==len(args):
            raise ValueError("Recipe functions must have the same number " + \
                    "of arguments as args in the provided list.")
        self.args = args
        self.func = func
        self.name = name
        self.desc = desc
        self.ref = ref
