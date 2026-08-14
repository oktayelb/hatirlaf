from util.suffix import Suffix, Type, SuffixGroup

## COMPLETE
class VerbNegativeSuffix(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.VERB,
                makes=Type.VERB,
                has_major_harmony=True, 
                has_minor_harmony=False, 
                needs_y_buffer=None, 
                group=SuffixGroup.VERB_NEGATING, 
                is_unique=True):
        
        if needs_y_buffer is None:
            if suffix[0] in ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü']:
                needs_y_buffer = True
            else:
                needs_y_buffer = False

        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=comes_to,
            makes=makes,
            form_function=None, # Force the use of the overridden _default_form
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )




negative_me     = VerbNegativeSuffix("negative_me"  , "me" )

negative_able   = VerbNegativeSuffix("negative_able", "eme")


VERB_NEGATIVES= [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]
