
import dataclasses
import numpy as np 
from typing import Optional, Tuple, TypeVar
from imitation.data.types import (
    Trajectory,
    Transitions,
    _rews_validation
)
T = TypeVar("T")
@dataclasses.dataclass(frozen=True, eq=False)
class RecurrentTrajectory(Trajectory):
    hidden_states: np.ndarray 


@dataclasses.dataclass(frozen=True, eq=False)
class RecurrentTrajectoryWithRew(RecurrentTrajectory):

    rews: np.ndarray

    infos: Optional[np.ndarray]

    
    def __post_init__(self):
        super().__post_init__()
        _rews_validation(self.rews, self.acts)

@dataclasses.dataclass(frozen=True)
class RecurrentTransitions(Transitions):
    hidden_states: np.ndarray 

@dataclasses.dataclass(frozen=True)
class RecurrentTransitionsWithRew(RecurrentTransitions):

    rews: np.ndarray

    def __post_init__(self):
        super().__post_init__()
        _rews_validation(self.rews, self.acts)

Pair = Tuple[T, T]
RecurrentTrajectoryPair = Pair[RecurrentTrajectory]
RecurrentTrajectoryWithRewPair = Pair[RecurrentTrajectoryWithRew]

