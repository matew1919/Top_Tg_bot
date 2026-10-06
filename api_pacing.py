"""One account-wide request clock for replies and semantic memory."""
import threading
import time


class ApiPacer:
    def __init__(self, interval=1.1):
        self.interval = interval
        self._condition = threading.Condition()
        self._next_start = 0.0
        self._foreground_until = 0.0

    def touch(self):
        with self._condition:
            self._foreground_until = time.monotonic() + 15
            self._condition.notify_all()

    def run(self, operation, *args, background=False, **kwargs):
        with self._condition:
            while True:
                deadline = max(self._next_start, self._foreground_until if background else 0)
                delay = deadline - time.monotonic()
                if delay <= 0:
                    self._next_start = time.monotonic() + self.interval
                    break
                self._condition.wait(delay)
        return operation(*args, **kwargs)

    def idle_delay(self):
        with self._condition:
            return max(0, self._foreground_until - time.monotonic())
