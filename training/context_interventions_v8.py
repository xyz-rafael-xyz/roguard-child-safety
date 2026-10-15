"""Training-only counterfactual prior topics with explicit, label-preserving resets."""
import random
import unicodedata
from training.context_v7 import CONCERNS
from training.context_v71 import TEXT
from training.context_transfer_v71 import DIRECT, BENIGN
from training.review_contrasts_v8 import TEXT as CONTRASTS
from training.short_review_contrasts_v8 import DATA

RESETS={
'ro':['Schimb subiectul. ', 'Acum despre mine: ', 'Altceva, din viața mea: ',
      'Închid discuția despre temă. ', 'Separat de ce spuneam înainte: ',
      'Aș vrea să trecem la situația mea. ', 'Uită pentru moment exercițiul; vorbesc despre mine. ',
      'Ceea ce urmează mi se întâmplă mie, în realitate. '],
'uk':['Зміню тему. ', 'Тепер про мене: ', 'Інше, з мого життя: ',
      'Завершую розмову про завдання. ', 'Окремо від того, що казав раніше: ',
      'Хочу перейти до моєї ситуації. ', 'Відкладемо поки вправу; я говорю про себе. ',
      'Те, що скажу далі, відбувається зі мною насправді. ']}


def generate():
    rows=[];rng=random.Random(880)
    for lang in ('ro','uk'):
        # Only the documented training slices; never held-out wording.
        positives=CONCERNS[lang][:12]+DIRECT[lang]['train']+CONTRASTS[lang]['positive']+DATA[lang]['positive']
        negatives=TEXT[lang][:24]+BENIGN[lang]['train']+CONTRASTS[lang]['negative']+DATA[lang]['negative']
        priors=TEXT[lang][:24]
        for signal,phrases in ((1,positives),(0,negatives)):
            for family,phrase in enumerate(phrases):
                for reset_index,reset in enumerate(RESETS[lang]):
                    prior=rng.choice(priors)
                    # A literal self-scope reset before task/fiction wording does
                    # not turn an otherwise ordinary task into a concern.
                    strings=[prior,reset+phrase]
                    for surface in (('original','plain') if lang=='ro' else ('original','lower')):
                        texts=[unicodedata.normalize('NFC',s) for s in strings]
                        if surface=='lower':texts=[s.lower() for s in texts]
                        if surface=='plain':texts=[''.join(c for c in unicodedata.normalize('NFD',s) if not unicodedata.combining(c)) for s in texts]
                        attrs={'subject':'SELF','source':'DIRECT','time':'CURRENT','assertion':'AFFIRMED','relation':'NEW','concern':str(signal)}
                        rows.append({'case_id':f'intervention-train-{lang}-{len(rows)}','language':lang,'turns':[{'role':'minor','text':s} for s in texts],'focus':1,'attributes':attrs,'source':'context_interventions_v8','family':f'{lang}-{signal}-{family}','reset_family':reset_index,'surface':surface,'fictional':True})
    return rows
