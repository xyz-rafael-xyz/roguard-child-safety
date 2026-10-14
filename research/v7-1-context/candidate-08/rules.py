"""Bounded language signals for human review; never a finding or authority.

These rules complement the experimental model. They deliberately do not infer
harm from hunger, solitude or ordinary requests for task help alone.
"""
import re
import unicodedata


def explicit_signals(text, language, frame):
    """Return reason codes only; an explicitly quoted frame remains scoped out."""
    if frame == 'QUOTED':
        return []
    text=unicodedata.normalize('NFC',text).lower().replace('’',"'").replace('ʼ',"'")
    if language=='ro':
        text=''.join(c for c in unicodedata.normalize('NFD',text) if not unicodedata.combining(c))
        support=r'\b(?:am|as avea|simt ca am) nevoie de (?:sprijin|sustinere)(?: emotionala?)?\s*[.!?;…]*\s*$'
        no_food=r"\b(?:nu am mancat|n-am mancat|nu mai am (?:mancare|hrana)|nu am (?:mancare|hrana)|nu am ce (?:sa )?manca|n-am ce (?:sa )?manca)\b"
        duration=r'\b(?:de ieri|de alaltaieri|de doua zile|de trei zile|de cateva zile|de [2-9] zile|de mult timp)\b'
        no_carer=r"\b(?:nu (?:este|e|sunt) (?:niciun |niciun fel de )?adult|nu (?:este|e) nimeni|nimeni nu (?:vine|are grija)|sunt (?:lasat |lasata )?singur[a]?)\b"
    elif language=='uk':
        support=r'\bмені (?:дуже |справді )?потрібна (?:чиясь |чия-небудь |емоційна )?підтримка\s*[.!?;…]*\s*$'
        no_food=r"\b(?:я не їв|я не їла|мені нічого їсти|у мене немає їжі|у мене нема їжі|у мене закінчилася їжа)\b"
        duration=r'\b(?:від учора|з учора|від вчора|з вчора|вже два дні|уже два дні|кілька днів|декілька днів|три дні|довго)\b'
        no_carer=r"\b(?:немає (?:жодного )?дорослого|нема (?:жодного )?дорослого|немає нікого|ніхто не (?:приходить|дбає)|я (?:залишився |залишилася )?сам(?:а)?)\b"
    else:
        return []
    result=[]
    if re.search(support,text):
        result.append('EXPLICIT_SUPPORT_REQUEST')
    # All three independently required. No rule fires on hunger or solitude alone.
    if all(re.search(pattern,text) for pattern in (no_food,duration,no_carer)):
        result.append('PROLONGED_UNMET_FOOD_AND_CARE')
    return result
