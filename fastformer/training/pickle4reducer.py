from multiprocessing.reduction import ForkingPickler, AbstractReducer
import pickle
import copyreg
import io
import dill


class ForkingPickler4(ForkingPickler):

    def __init__(self, file, protocol=pickle.HIGHEST_PROTOCOL):
        # print(type(file), type(protocol), file)
        buf = io.BytesIO()
        dill.dump(file, buf)
        file = buf
        assert hasattr(file, "write")
        super().__init__(file, protocol)

    @classmethod
    def dumps(cls, obj, protocol=pickle.HIGHEST_PROTOCOL):
        buf = io.BytesIO()
        cls(buf, protocol).dump(obj)
        return buf.getbuffer()
    loads = dill.loads


def dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL):
    # print(type(obj), type(file), obj, file)
    ForkingPickler4(file, pickle.HIGHEST_PROTOCOL).dump(obj)


class Pickle4Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler4
    register = ForkingPickler4.register
    dump = dump
