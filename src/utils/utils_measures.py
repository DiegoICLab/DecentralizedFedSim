import sys

def get_length_bytes(obj, seen=None):
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    total_size = sys.getsizeof(obj)
    
    if isinstance(obj, dict):
        total_size += sum(get_length_bytes(k, seen) + get_length_bytes(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        total_size += sum(get_length_bytes(i, seen) for i in obj)
    
    return total_size