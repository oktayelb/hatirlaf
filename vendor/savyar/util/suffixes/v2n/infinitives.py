from util.suffix import Suffix, Type, SuffixGroup
import util.word_methods as wrd
## COMPLETE
class Infinitive(Suffix):
    def __init__(self, name, suffix):
        
        needs_y_buffer = False
        
        if suffix[0] in wrd.VOWELS:
            needs_y_buffer = True

        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=Type.VERB,
            makes=Type.NOUN,
            form_function=None,
            has_major_harmony=True,
            has_minor_harmony=True,
            needs_y_buffer=needs_y_buffer,
            group=SuffixGroup.V2N_DERIVATIONAL,
            is_unique=False
        )



infinitive_me     = Infinitive("infinitive_me"  , "me"  )
infinitive_mek    = Infinitive("infinitive_mek" , "mek" )
nounifier_iş      = Infinitive("nounifier_iş"   , "iş"  )

INFINITIVES = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]