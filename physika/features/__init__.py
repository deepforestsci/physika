from physika.elf import REGISTRY
from physika.features.classes import ClassFeature
from physika.features.randomness import RandomnessFeature
from physika.features.tuple_unpack import TupleUnpackFeature
from physika.features.indexing_and_slicing import IndexingandSlicing

REGISTRY.register(ClassFeature())
REGISTRY.register(RandomnessFeature())
REGISTRY.register(TupleUnpackFeature())
REGISTRY.register(IndexingandSlicing())
