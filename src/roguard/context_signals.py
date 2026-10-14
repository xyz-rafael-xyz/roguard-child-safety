"""Bounded language signals for human review; never a finding or authority.

These rules complement the experimental model. They deliberately do not infer
harm from hunger, solitude or ordinary requests for task help alone.
"""
import re
import unicodedata


def _affirmed(pattern, text, language, local=False):
    """Conservative clause-level negation guard for affirmative rule evidence."""
    boundaries = r"[.!?;:]|\b(?:dar|insa|totusi)\b" if language=='ro' else r"[.!?;:]|\b(?:але|проте|однак)\b"
    if local:
        boundaries += r",|\b(?:si|iar)\b" if language=='ro' else r",|\b(?:і|й|та)\b"
    negation = r"\bnu\b|\bn-" if language=='ro' else r"\bне\b"
    for match in re.finditer(pattern, text):
        prefix = re.split(boundaries, text[:match.start()])[-1]
        if not re.search(negation, prefix):
            return True
    return False


def explicit_signals(text, language, frame, mention=None):
    """Return reason codes only; an explicitly quoted frame remains scoped out."""
    text=unicodedata.normalize('NFC',text).lower().replace('’',"'").replace('ʼ',"'")
    text=re.sub(r'\s+', ' ', text).strip()
    if language=='ro':
        text=''.join(c for c in unicodedata.normalize('NFD',text) if not unicodedata.combining(c))
        support=r'\b(?:am|as avea|simt ca am) nevoie de (?:sprijin|sustinere)(?: emotionala?)?(?: pentru (?:ce simt|ceea ce simt|sentimentele mele|grijile mele)(?: in ultima vreme| acum| in aceste zile)?)?\s*[.!?;…]*\s*$'
        exit_fiction=r'\b(?:oprim|opresc|incheiem|inchei|gata cu)\s+(?:povestea|fictiunea|scena|jocul de rol)\b'
        personal=r'\b(?:in viata mea|in realitate|despre mine|personal|eu chiar)\b'
        no_food=r"\b(?:nu am mancat|n-am mancat|nu mai am (?:mancare|hrana)|nu am (?:mancare|hrana)|nu am ce (?:sa )?manca|n-am ce (?:sa )?manca)\b"
        duration=r'\b(?:de ieri|de alaltaieri|de doua zile|de trei zile|de cateva zile|de [2-9] zile|de mult timp)\b'
        no_carer=r"\b(?:nu (?:este|e|sunt) (?:niciun |niciun fel de )?adult|nu (?:este|e) nimeni(?=\s*(?:$|[.!?;,]|acasa\b|cu mine\b|care sa (?:aiba grija|ma ajute)\b))|nimeni nu (?:vine|are grija)|sunt (?:lasat |lasata )?singur[a]?)\b"
    elif language=='uk':
        support=r'\bмені (?:дуже |справді )?потрібна (?:чиясь |чия-небудь |емоційна )?підтримка(?: через (?:мої почуття|мої переживання))?\s*[.!?;…]*\s*$'
        exit_fiction=r'\b(?:зупинимо|припинимо|закінчимо|залишимо)\s+(?:вигадану\s+)?(?:історію|вигадку|розповідь|рольову гру)\b'
        personal=r'\b(?:у (?:повсякденному |реальному )?житті|в моєму житті|про себе|особисто)\b'
        no_food=r"\b(?:я не їв|я не їла|мені нічого їсти|у мене немає їжі|у мене нема їжі|у мене закінчилася їжа)\b"
        duration=r'\b(?:від учора|з учора|від вчора|з вчора|вже два дні|уже два дні|кілька днів|декілька днів|три дні|довго)\b'
        no_carer=r"\b(?:немає (?:жодного )?дорослого|нема (?:жодного )?дорослого|немає нікого(?=\s*(?:$|[.!?;,]|вдома\b|поруч\b|хто б (?:подбав|допоміг)\b))|ніхто не (?:приходить|дбає)|я (?:залишився |залишилася )?сам(?:а)?)\b"
    else:
        return []
    if frame=='QUOTED':
        # Require both an explicit end to fiction and a personal-context cue,
        # plus the model's positive concern mention. A frame conflict is reviewed.
        return ['PERSONAL_CONTEXT_CONTRADICTS_QUOTATION'] if mention==1 and _affirmed(exit_fiction,text,language) and _affirmed(personal,text,language) else []
    result=[]
    if _affirmed(support,text,language):
        result.append('EXPLICIT_SUPPORT_REQUEST')
    # All three independently required. No rule fires on hunger or solitude alone.
    if re.search(no_food,text) and re.search(duration,text) and _affirmed(no_carer,text,language,local=True):
        result.append('PROLONGED_UNMET_FOOD_AND_CARE')
    return result
