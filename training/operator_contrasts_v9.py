"""Training-only denial and context-switch contrasts after regression diagnosis.

Common operator vocabulary is deliberately taught on new training content.
The inspected calibration set is therefore a regression set, never fresh
validation. Reserved probe wording is not read or used by this generator.
"""
from training.context_v7 import CONCERNS
from training.disclosure_hesitancy_v8 import POSITIVE
from training.context_v71 import TEXT

WORDS={
 'ro':{
  'deny':[
   'Neg că aceste cuvinte mă descriu: «{text}»',
   'Vreau să neg că mă regăsesc în această descriere: «{text}»',
   'Pot nega sincer că aceasta ar fi situația mea: «{text}»',
   'Cuvintele «{text}» nu descriu ce trăiesc eu; le neg.',
  ],
  'affirm':[
   'Nu neg că aceste cuvinte mă descriu: «{text}»',
   'Nu pot nega că aceasta este situația mea: «{text}»',
   'Ar fi fals să neg asta; este adevărat despre mine: «{text}»',
   'Cuvintele «{text}» mă descriu; nu le neg.',
  ],
  'self':['Revin la ce trăiesc eu, în afara poveștii: «{text}»',
          'Închei partea inventată. Următorul lucru mi se întâmplă mie: «{text}»',
          'Aceste cuvinte sunt despre experiența mea personală, nu despre personaj: «{text}»',
          'Povestesc despre mine, în viața reală: «{text}»'],
  'other':['Acum redau ce mi-a povestit un prieten real despre el: «{text}»',
           'Un coleg mi-a mărturisit asta despre viața lui, nu despre o poveste: «{text}»'],
  'fiction':['Scriu doar o replică pentru un personaj fictiv, nu vorbesc despre mine: «{text}»',
             'Pentru o scenă imaginară, inventez cuvintele: «{text}»',
             'În povestirea inventată, personajul spune: «{text}»',
             'Vocea din această replică este a unui personaj inventat: «{text}»'],
  'prior':['Într-o scenă inventată, actorul joacă rolul unui explorator.',
           'Personajele din sceneta mea caută o comoară imaginară.',
           'Pentru piesa de teatru am inventat un personaj care vorbește cu luna.',
           'Analizez dialogul fictiv dintre două personaje dintr-o carte.',
           'Desenez o scenă în care un erou imaginar își găsește pălăria.',
           'În povestea mea inventată, un personaj construiește un castel.'],
 },
 'uk':{
  'deny':[
   'Заперечую, що ці слова описують мене: «{text}»',
   'Хочу заперечити, що цей опис стосується мене: «{text}»',
   'Можу щиро заперечити, що це моя ситуація: «{text}»',
   'Слова «{text}» не описують мій стан; я це заперечую.',
  ],
  'affirm':[
   'Не заперечую, що ці слова описують мене: «{text}»',
   'Не можу заперечити, що це моя ситуація: «{text}»',
   'Було б неправдою заперечувати це; ось що відбувається зі мною: «{text}»',
   'Слова «{text}» описують мій стан; я цього не заперечую.',
  ],
  'self':['Повертаюся до того, що переживаю сам, поза історією: «{text}»',
          'Завершую вигадану частину. Наступне відбувається зі мною: «{text}»',
          'Ці слова про мій особистий досвід, а не про персонажа: «{text}»',
          'Розповідаю про себе в реальному житті: «{text}»'],
  'other':['Тепер передаю, що справжній друг розповів мені про себе: «{text}»',
           'Однокласник зізнався мені в цьому про своє життя, а не про історію: «{text}»'],
  'fiction':['Пишу лише репліку для вигаданого персонажа, а не розповідаю про себе: «{text}»',
             'Для уявної сцени вигадую такі слова: «{text}»',
             'У вигаданому оповіданні персонаж каже: «{text}»',
             'Голос у цій репліці належить вигаданому персонажу: «{text}»'],
  'prior':['У вигаданій сцені актор грає роль мандрівника.',
           'Персонажі моєї сценки шукають уявний скарб.',
           'Для театральної вистави я вигадав персонажа, який говорить із місяцем.',
           'Аналізую вигаданий діалог двох персонажів книжки.',
           'Малюю сцену, де уявний герой знаходить свого капелюха.',
           'У моїй вигаданій історії персонаж будує замок.'],
 },
}


def generate():
    rows=[]
    for lang,words in WORDS.items():
        contents=[(1,t) for t in CONCERNS[lang][:12]+POSITIVE[lang]]+[(0,t) for t in TEXT[lang][:24]]
        for family,(signal,text) in enumerate(contents):
            for operator in ('deny','affirm','self','other','fiction'):
                for variant,frame in enumerate(words[operator]):
                    current=frame.format(text=text)
                    # Same irrelevant prior for matched affirmative/denial pairs.
                    prior=words['prior'][(family+variant)%len(words['prior'])]
                    attrs={'subject':'OTHER' if operator=='other' else 'FICTIONAL' if operator=='fiction' else 'SELF',
                           'source':'REPORTED' if operator=='other' else 'FICTION' if operator=='fiction' else 'DIRECT',
                           'assertion':'NEGATED' if operator=='deny' else 'AFFIRMED',
                           'relation':'NEW','time':'CURRENT','concern':str(signal)}
                    rows.append({'case_id':f'operator-v9-train-{lang}-{len(rows)}','language':lang,
                                 'turns':[{'role':'minor','text':prior},{'role':'assistant','text':'Te ascult.' if lang=='ro' else 'Я слухаю.'},{'role':'minor','text':current}],
                                 'focus':2,'attributes':attrs,'source':'operator_contrasts_v9','family':f'{lang}-{family}',
                                 'pair_group':f'operator-v9-{lang}-{family}-{variant}','fictional':True})
    return rows
