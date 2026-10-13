"""Descriptive response lint; never certifies safety, reading age or meaning."""
from __future__ import annotations
import re
import unicodedata

WORDS = re.compile(r"[^\W\d_]+(?:[’'-][^\W\d_]+)*", re.UNICODE)
RULES = {
 'ro': {
  'SECRET_PROMISE': r'\b(?:promit ca (?:nu spun|ramane intre noi)|va ramane secret|nu voi spune nimanui)\b',
  'UNSUPPORTED_GUARANTEE': r'\b(?:totul va fi bine|iti garantez|sigur nu se va intampla nimic)\b',
  'DETAIL_PRESSURE': r'\b(?:trebuie sa imi spui tot|spune-mi toate detaliile|nu te opri pana|raspunde la toate intrebarile)\b',
  'IDENTIFYING_DATA_REQUEST': r'\b(?:spune-mi|trimite-mi|scrie-mi)\b.{0,45}\b(?:adresa|numele complet|numarul de telefon|scoala)\b',
  'DEPENDENCY_PRESSURE': r'\b(?:doar eu te inteleg|numai cu mine|nu ai nevoie de altcineva)\b',
 },
 'uk': {
  'SECRET_PROMISE': r'\b(?:обіцяю нікому не казати|це залишиться таємницею|нікому не розповім)\b',
  'UNSUPPORTED_GUARANTEE': r'\b(?:усе буде добре|все буде добре|гарантую тобі|точно нічого не станеться)\b',
  'DETAIL_PRESSURE': r'\b(?:мусиш розповісти все|розкажи всі подробиці|відповідай на всі запитання)\b',
  'IDENTIFYING_DATA_REQUEST': r'\b(?:напиши|надішли|скажи)\b.{0,45}\b(?:адресу|повне ім.я|номер телефону|назву школи)\b',
  'DEPENDENCY_PRESSURE': r'\b(?:лише я тебе розумію|тільки зі мною|тобі більше ніхто не потрібен)\b',
 }}
JARGON = {'ro': {'procedura','confidentialitate','consimtamant','jurisdictie','institutie'},
          'uk': {'процедура','конфіденційність','згода','юрисдикція','установа'}}

def normalized(text, language):
    text = text.lower()
    if language == 'ro':
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))
    return text

def complexity(text, language):
    words=WORDS.findall(text)
    sentences=[s for s in re.split(r'[.!?…]+',text) if WORDS.search(s)]
    return {'words':len(words),'sentences':len(sentences),
            'mean_words_per_sentence':round(len(words)/max(1,len(sentences)),1),
            'long_words_12plus':sum(len(w)>=12 for w in words),
            'jargon_occurrences':sum(normalized(w,language) in JARGON[language] for w in words),
            'reading_age_estimated':False,'comprehension_established':False}

def review_response(text, language, revision=None):
    if language not in RULES or not isinstance(text,str) or not 1 <= len(text.strip()) <= 6000:
        raise ValueError('Provide a nonempty Romanian/Ukrainian draft up to 6000 characters')
    source=normalized(text,language)
    findings=[]
    for code,pattern in RULES[language].items():
        # Offsets refer to the original draft only when normalization preserves length.
        for match in re.finditer(pattern,source):
            findings.append({'code':code,'start':match.start(),'end':match.end(),
                             'basis':'lexical_pattern_requires_context_review'})
    result={'status':'REVIEW_RESPONSE' if findings else 'NO_LISTED_PATTERN',
            'findings':findings,'complexity':complexity(text,language),
            'safe_response_established':False,'coverage':'limited_phrase_rules',
            'recommended_human_checks':['assumptions','pressure','privacy','meaning','support_contract']}
    if revision is not None:
        if not isinstance(revision,str) or not 1<=len(revision.strip())<=6000:
            raise ValueError('Invalid revised draft')
        before=set(re.findall(r'\d+',text));after=set(re.findall(r'\d+',revision))
        negation=r'\b(?:nu|nici|fara)\b' if language=='ro' else r'\b(?:не|ні|без)\b'
        result['revision']={'complexity':complexity(revision,language),
            'number_tokens_changed':before!=after,
            'negation_count_changed':len(re.findall(negation,source))!=len(re.findall(negation,normalized(revision,language))),
            'meaning_preservation_established':False,
            'requires_side_by_side_review':True}
    return result
