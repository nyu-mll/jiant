"""Global task registry.

Import this to use @register_task when defining new tasks,
then access the registry as registry.REGISTRY.
"""

REGISTRY = {}  # Do not edit manually!

def register_task(name, rel_path, **kw):
    '''Decorator to register a task.

    Use this instead of adding to NAME2INFO in preprocess.py

    If kw is not None, this will be passed as additional args when the Task is
    constructed in preprocess.py.

    Usage:
    @register_task('mytask', 'my-task/data', **extra_kw)
    class MyTask(SingleClassificationTask):
        ...
    '''
    def _wrap(cls):
        entry = (cls, rel_path, kw) if kw else (cls, rel_path)
        REGISTRY[name] = entry
        return cls
    return _wrap


