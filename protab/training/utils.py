class SimpleCounter:
    def __init__(self, start: int = 0) -> None:
        self.value = start

    def __call__(self):
        self.value += 1
        return self.value

    def __repr__(self):
        return str(self.value)

    def __int__(self):
        return self.value

    def __add__(self, other):
        return self.value + other
