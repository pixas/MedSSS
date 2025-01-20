
tree_registry = {}

def register_tree(name):
    """装饰器，用于注册类"""
    def decorator(cls):
        tree_registry[name] = cls
        return cls
    return decorator