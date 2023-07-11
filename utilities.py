
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def __instancecheck__(mcs, instance):
        if instance.__class__ is mcs:
            return True
        else:
            return isinstance(instance.__class__, mcs)


def is_member_in_all(x, lists):
    for s in lists:
        if x not in s:
            return False
    return True


def get_intersection(lists):
    sz = len(lists)
    if sz == 0:
        return []
    first = lists[0]
    rest = lists[1:]
    sz1 = len(first)
    res = []
    k = 0
    while k < sz1:
        if is_member_in_all(first[k], rest):
            res.append(first[k])
        k = k + 1
    return res


