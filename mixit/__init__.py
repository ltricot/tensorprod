from warnings import warn
from  operator import add
from functools import reduce
import importlib

from ..utils import find_totest, find_todocument, find_todo
from . import interfaces


modules = ['diverse', 'saving', 'training', 'testing']

for mod in modules:
    globals()[mod] = importlib.import_module('.' + mod, package='tensorprod.mixit')

# warn if undone documenting or testing
for find_marked in [find_totest, find_todocument]:
    for module in [globals()[m] for m in modules]:
        marked = find_marked(module)
        if marked:
            warn('Following objects in {} are marked with {}: {}'.format(
                module.__name__, find_marked.__name__, ', '.join(marked.keys())),
                ImportWarning,
            )

for module in [globals()[m] for m in modules]:
    todos = find_todo(module)
    if todos:
        warn('Following objects in {} are marked with todo comments:\n'.format(module.__name__) +\
            reduce(add, ('{}: {}\n'.format(k, v) for k, v in todos.items())),
            ImportWarning,
        )
