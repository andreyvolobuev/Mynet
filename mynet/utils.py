def debatch(func):
    def wrapper(*args):
        if isinstance(args[-1][0], list):
            return [wrapper(*args[:-1], i) for i in args[-1]]
        return func(*args)
    return wrapper
