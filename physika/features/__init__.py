from physika.elf import REGISTRY
from physika.features.classes import ClassFeature
from physika.features.randomness import RandomnessFeature
from physika.features.tuple_unpack import TupleUnpackFeature

REGISTRY.register(ClassFeature())
REGISTRY.register(RandomnessFeature())
REGISTRY.register(TupleUnpackFeature())
