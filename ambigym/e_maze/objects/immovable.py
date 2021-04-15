import numpy as np


class Immovable:
    """
    Base class to specify the physics of immovable objects in the world.
    """

    def __init__(self, position, traversable, transparent):
        """
        Construct an immovable object.
        
        Keyword arguments:
        position -- permanent position of the immovable
        traversable -- True if movables can freely move through
        transparent -- False if this object casts shadows
        """
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

        How to render the movable.
        """
        raise NotImplementedError