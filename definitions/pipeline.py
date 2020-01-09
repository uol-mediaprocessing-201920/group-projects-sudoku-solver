import threading
from typing import List


class Stage:
    def __init__(self):
        self.semaphore = threading.Semaphore(1)

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

        def thread_function():
            new_data = stage.compute(data)
            stage.semaphore.release()
            if new_data is not None:
                self.__spawn_stage(stage_index + 1, new_data)

        thread = threading.Thread(target=thread_function, daemon=True)
        thread.start()
        return True
