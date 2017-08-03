from functools import partial


# mark obj with attr
def _mark(obj, attr, value=True):
    setattr(obj, attr, value)
    return obj

totest = partial(_mark, attr='__totest__')
todocument = partial(_mark, attr='__todocument__')
todo = lambda comment: partial(_mark, attr='__todo__', value=comment)

def finder(attr):
    # find objects in a module who have attr
    def find(module):
        modvars = vars(module)
        return {name: getattr(modvars[name], attr) for name in dir(module)\
            if hasattr(modvars[name], attr)}
    find.__name__ = attr
    return find

find_totest = finder('__totest__')
find_todocument = finder('__todocument__')
find_todo = finder('__todo__')
