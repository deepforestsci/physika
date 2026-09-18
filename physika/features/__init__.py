from physika.elf import REGISTRY
from physika.features.classes import ClassFeature
from physika.features.randomness import RandomnessFeature
from physika.features.tuple_unpack import TupleUnpackFeature
from physika.features.indexing_and_slicing import IndexingandSlicing
from physika.features.string import StringFeature
from physika.features.modifier_letters import ModifierLetters
from physika.features.dictionary import DictionaryFeature

REGISTRY.register(ClassFeature())
REGISTRY.register(RandomnessFeature())
REGISTRY.register(TupleUnpackFeature())
REGISTRY.register(IndexingandSlicing())
REGISTRY.register(StringFeature())
REGISTRY.register(ModifierLetters())
REGISTRY.register(DictionaryFeature())
