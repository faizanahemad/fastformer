from multiprocessing.reduction import ForkingPickler, AbstractReducer
import pickle


class ForkingPickler4(ForkingPickler):
    def __init__(self, *args):
        args = list(args)
        if len(args) > 1:
            args[1] = pickle.HIGHEST_PROTOCOL
        else:
            args.append(pickle.HIGHEST_PROTOCOL)
        super().__init__(*args)

    @classmethod
    def dumps(cls, obj, protocol=pickle.HIGHEST_PROTOCOL):
        return ForkingPickler.dumps(obj, pickle.HIGHEST_PROTOCOL)

    loads = pickle.loads


def dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL):
    ForkingPickler4(file, protocol).dump(obj)


class Pickle4Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler4
    register = ForkingPickler4.register
    dump = dump
