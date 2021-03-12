import numpy as np


class Immovable:
    def __init__(self, position, traversable, transparent):
        self.position = (
            position
            if isinstance(position, np.ndarray)
            else np.array(position)
        )
        self.traversable = traversable
        self.transparent = transparent

    def render(self):
        """
        To be implemented by subclasses.
        """
        raise NotImplementedError