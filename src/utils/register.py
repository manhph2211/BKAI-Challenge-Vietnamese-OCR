import sys
from collections import defaultdict
from copy import deepcopy


_module_to_models = defaultdict(set)
_model_to_module = {}  # mapping of model names to module names
_model_entrypoints = {}  # mapping of model name to entrypoint fns
_model_has_pretrained = set()  # set of model names that have pretrained weight url present
_model_default_cfgs = dict()


def register_model(fn):
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # Add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # Add entries to register dict/sets
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    has_pretrained = False
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        has_pretrained = 'url' in mod.default_cfgs[model_name] and 'http' in mod.default_cfgs[model_name]
        _model_default_cfgs[model_name] = deepcopy(mod.default_cfgs[model_name])

    if has_pretrained:
        _model_has_pretrained.add(model_name)

    return fn


def is_model(model_name):

    return model_name in _model_entrypoints


def is_model_in_modules(model_name, module_names):

    return any(model_name in _module_to_models[n] for n in module_names)


def model_entrypoints(model_name):

    return _model_entrypoints[model_name]