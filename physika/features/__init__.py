from physika.elf import REGISTRY
from physika.features.classes import ClassFeature
from physika.features.randomness import RandomnessFeature
from physika.features.tuple_unpack import TupleUnpackFeature
from physika.features.indexing import Indexing

REGISTRY.register(ClassFeature())
REGISTRY.register(RandomnessFeature())
REGISTRY.register(TupleUnpackFeature())
REGISTRY.register(Indexing())
