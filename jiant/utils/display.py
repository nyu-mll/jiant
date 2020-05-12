import json
import tqdm as _tqdm


def tqdm(iterable=None, desc=None, total=None, initial=0):
    if is_notebook():
        return _tqdm.tqdm_notebook(iterable=iterable, desc=desc, total=total, initial=initial,)
    else:
        return _tqdm.tqdm(iterable=iterable, desc=desc, total=total, initial=initial,)


def trange(*args, desc=None, total=None):
    return tqdm(range(*args), desc=desc, total=total)


def maybe_tqdm(iterable=None, desc=None, total=None, initial=0, verbose=True):
    if verbose:
        return tqdm(iterable=iterable, desc=desc, total=total, initial=initial,)
    else:
        return iterable


def maybe_trange(*args, verbose, **kwargs):
    return maybe_tqdm(range(*args), verbose=verbose, **kwargs)


def show_json(obj, do_print=True):
    string = json.dumps(obj, indent=2)
    if do_print:
        print(string)
    else:
        return string


def is_notebook():
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
