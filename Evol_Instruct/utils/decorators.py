import functools

def retry_on_exception(max_retries=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        print(f"Function failed after {max_retries} retries. Returning empty dictionary.")
                        return {}
                    print(f"Exception occurred: {e}. Retrying {retries}/{max_retries}...")
        return wrapper
    return decorator