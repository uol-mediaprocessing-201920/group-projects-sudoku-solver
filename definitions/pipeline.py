import threading
from typing import List

class SingleThreadExecutor():
    def __init__(self):
        self.queue = []
        self.queue_semaphore = threading.Semaphore(0)
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def submit(self, work):
        self.queue.append(work)
        self.queue_semaphore.release()

    def loop(self):
        while True:
            self.queue_semaphore.acquire(blocking=True)
            work = self.queue.pop(0)
            work()


class Stage:
    def __init__(self):
        self.semaphore = threading.Semaphore(0)
        self.executor = SingleThreadExecutor()
        self.executor.submit(self.setup)
        self.executor.submit(lambda: self.semaphore.release())

    def setup(self):
        pass

    def compute(self, data):
        raise NotImplementedError()


class Pipeline:
    def __init__(self, stages: List[Stage]):
        self.stages = stages

    def feed(self, data):
        self.__spawn_stage(0, data)

    def __spawn_stage(self, stage_index: int, data):
        stage = self.stages[stage_index]
        if not stage.semaphore.acquire(blocking=False):
            return False
        def work():
            new_data = stage.compute(data)
            stage.semaphore.release()
            if new_data is not None:
                self.__spawn_stage(stage_index + 1, new_data)
        stage.executor.submit(work)
        return True
