# So that the type "Speller" doesn't have to be in quotes everywhere
from __future__ import annotations

import enum
from typing import TYPE_CHECKING

# Only needed when actually type checking, to avoid circular imports
if TYPE_CHECKING:
    from src.speller.speller import Speller

class Direction(enum.Enum):
    LEFT = 1
    RIGHT = 2

class SpellerState:
    def __repr__(self):
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

class UnsupportedTransitionError(Exception):
    pass

class SpellerStateIdle(SpellerState):
    def select(self, speller: Speller):
        speller.state = SpellerStateWriting()

class SpellerStateWriting(SpellerState):
    def __init__(self):
        self.cursor = 0
    
    def back(self, speller: Speller):
        speller.state = SpellerStateIdle()
        
    def select(self, speller: Speller):
        speller.state = SpellerStateSectorNavigation()
    
    def move(self, speller: Speller, direction: Direction):
        raise UnsupportedTransitionError("Moving in writing state not implemented yet")

class SpellerStateSectorNavigation(SpellerState):
    def __init__(self):
        self.cursor = 0
    
    def back(self, speller: Speller):
        speller.state = SpellerStateWriting()
        
    def select(self, speller: Speller):
        speller.state = SpellerStateLetterNavigation(self.cursor)
    
    def move(self, speller: Speller, direction: Direction):
        match direction:
            case Direction.LEFT:
                self.cursor = (self.cursor - 1) % len(speller.map)
            case Direction.RIGHT:
                self.cursor = (self.cursor + 1) % len(speller.map)

class SpellerStateLetterNavigation(SpellerState):
    def __init__(self, selected_sector: int):
        self.selected_sector = selected_sector
        self.cursor = 0
        
    def back(self, speller: Speller):
        speller.state = SpellerStateSectorNavigation()
    
    def select(self, speller: Speller):
        speller.on_letter_select(speller.map[self.selected_sector][self.cursor])

        speller.state = SpellerStateWriting()
        
    def move(self, speller: Speller, direction: Direction):
        letters = speller.map[self.selected_sector]
        
        match direction:
            case Direction.LEFT:
                self.cursor = (self.cursor - 1) % len(letters)
            case Direction.RIGHT:
                self.cursor = (self.cursor + 1) % len(letters)