from enum import Enum

class AgeGroup(str, Enum):
    young = "young"
    middle = "middle"
    senior = "senior"

class Smoker(str, Enum):
    yes = "yes"
    no = "no"

class AlcoholLevel(str, Enum):
    none = "none"
    moderate = "moderate"
    high = "high"

class ExerciseLevel(str, Enum):
    none = "none"
    regular = "regular"
    light = "light"

class DietQuality(str, Enum):
    good = "good"
    average = "average"
    poor = "poor"
