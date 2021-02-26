from multiprocessing.reduction import ForkingPickler, AbstractReducer
import pickle
import copyreg
import io


class ForkingPickler4(pickle.Pickler):
    _extra_reducers = {}
    _copyreg_dispatch_table = copyreg.dispatch_table

    def __init__(self, file, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=None):
        print(type(file), type(protocol), file)
        super().__init__(file, protocol, buffer_callback=buffer_callback)
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
    print(type(obj), type(file), obj, file)
    ForkingPickler4(file, pickle.HIGHEST_PROTOCOL).dump(obj)


class Pickle4Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler4
    register = ForkingPickler4.register
    dump = dump
