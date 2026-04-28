
import ast
import inspect
import builtins

def get_called_functions(func):
    """
    Finds all the functions used in the func function and returns a list of their names and a dict with the names as
    keys and the functions' docstrings as values (if a function doesn't have a docstring the value is None).
    :param func: The func to look into
    :return: a list of functions' names and a dict of their docstrings
    """
    source = inspect.getsource(func)
    tree = ast.parse(source)

    namespace = {**vars(builtins), **func.__globals__}

    called = []       # list of display names like "dsl.some_func"
    resolved = {}     # display name -> callable object

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name):
            # Plain call: some_func()
            name = node.func.id
            display = name
            obj = namespace.get(name)

        elif isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if isinstance(node.func.value, ast.Name):
                # Qualified call: dsl.some_func()
                owner_name = node.func.value.id
                owner = namespace.get(owner_name)
                display = f"{owner_name}.{attr}"
                obj = getattr(owner, attr, None) if owner is not None else None
            else:
                # Chained call: something.complex.foo() — just use attr name
                display = attr
                obj = None
        else:
            continue

        if display not in resolved:
            called.append(display)
            resolved[display] = obj

    docstrings = {
        name: inspect.getdoc(obj) if callable(obj) else None
        for name, obj in resolved.items()
    }

    return called, docstrings