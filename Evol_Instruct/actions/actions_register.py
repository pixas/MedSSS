ACTIONS_REGISTRY = []

def register(action_name, func_type="base"):
    def decorator(func):
        ACTIONS_REGISTRY.append((func, func_type, action_name))
        return func
    return decorator