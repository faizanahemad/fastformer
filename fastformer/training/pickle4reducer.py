from multiprocessing.reduction import ForkingPickler, AbstractReducer
import pickle
import copyreg
import io


class ForkingPickler4(pickle.Pickler):
    _extra_reducers = {}
    _copyreg_dispatch_table = copyreg.dispatch_table

    def __init__(self, *args):
        args = list(args)
        if len(args) > 1:
            args[1] = pickle.HIGHEST_PROTOCOL
        else:
            args.append(pickle.HIGHEST_PROTOCOL)
        super().__init__(*args)
        self.dispatch_table = self._copyreg_dispatch_table.copy()
        self.dispatch_table.update(self._extra_reducers)

    @classmethod
    def register(cls, type, reduce):
        '''Register a reduce function for a type.'''
        cls._extra_reducers[type] = reduce

    @classmethod
    def dumps(cls, obj, protocol=pickle.HIGHEST_PROTOCOL):
        buf = io.BytesIO()
        cls(buf, protocol).dump(obj)
        return buf.getbuffer()

    loads = pickle.loads


def dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL):
    ForkingPickler4(file, protocol).dump(obj)


class Pickle4Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler4
    register = ForkingPickler4.register
    dump = dump
