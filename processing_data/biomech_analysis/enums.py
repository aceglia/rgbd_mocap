import enum


class FilteringMethod(enum.Enum):
    """
    Enum class for filtering methods.
    """

    NONE = 0
    Kalman = 1
    MovingAverage = 3
    OffLine = 4
