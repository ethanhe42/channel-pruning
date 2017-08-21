import multiprocessing as mp
import numpy as np

class Worker():
    def __init__(self):
        self.mp_queue = mp.Queue()
        self._target = None

    def set_target(self, target):
        self._target = target

    def do(self, target=None, **mp_kwargs):
        if target is None:
            assert self._target is not None, "please provide target"
            # cache target
            target = self._target
        else:
            self._target = target
        
        def job(queue, **kwargs):
            ret = target(**kwargs)
            assert isinstance(ret, dict)
            queue.put(ret)

        mp_kwargs['queue'] = self.mp_queue
        has_q = False
        for var_name, var_value in mp_kwargs.items():
            if var_name == 'q':
                has_q = True
                q = var_value
                break

        p = mp.Process(target=job, kwargs=mp_kwargs)
        p.start()
        if has_q:
            points_dict = dict()
            while True:
                item = q.get()
                if isinstance(item, str) and item == 'end':
                    break
                else:
                    print("got ", item[0])
                    points_dict[item[0]] = item[1]
            
            ret = []
            ret.append(points_dict)
            for var_name, var_value in mp_kwargs.items():
                if var_name == 'convs':
                    name_len = len(var_value)
                    break
            
            for i in range(name_len):
                arr = []
                while True:
                    item = q.get()
                    if isinstance(item, str) and item == 'end':
                        break
                    else:
                        arr.append(item)

                print("joined", i, '/', name_len, ' length ', len(arr))
                ret.append(np.array(arr))

        result = self.mp_queue.get()
        p.join()
        if has_q:
            return ret
        return result

def test_worker():
    def a(queue, **kwargs):
        for i,j in kwargs.items():
            print(i, j)
        print(queue)
    a(b=3,d=4, queue=6)

if __name__ == "__main__":
    test_worker()