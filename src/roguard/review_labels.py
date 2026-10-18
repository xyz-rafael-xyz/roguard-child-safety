"""Localized reviewer display derived from a verified clarification certificate.

These project-authored labels describe message interpretation, not verified
facts. Display data never changes the certificate, policy or answer codes.
"""
from .assurance import verify
from .evidence import ONTOLOGY

PROMPTS = {
    'ro': {
        'subject':'Despre cine este vorba în mesaj?',
        'time':'La ce perioadă sau situație se referă mesajul?',
        'assertion':'Cum este formulată afirmația în mesaj?',
        'source':'Cum este prezentată relatarea în mesaj?',
        'relation':'Cum se leagă mesajul de afirmațiile anterioare?',
        'concern':'Ce fel de conținut este menționat în mesaj?',
    },
    'uk': {
        'subject':'Про кого йдеться в повідомленні?',
        'time':'Якого часу або ситуації стосується повідомлення?',
        'assertion':'Як сформульовано твердження в повідомленні?',
        'source':'Як подано розповідь у повідомленні?',
        'relation':'Як повідомлення пов’язане з попередніми твердженнями?',
        'concern':'Який зміст згадується в повідомленні?',
    },
}
OPTIONS = {
    'ro': {
        'subject':{'SELF':'Persoana care vorbește','OTHER':'Altă persoană reală','FICTIONAL':'Personaj fictiv'},
        'time':{'CURRENT':'În prezent','PAST':'În trecut','HYPOTHETICAL':'Situație ipotetică'},
        'assertion':{'AFFIRMED':'Afirmată în mesaj','NEGATED':'Negată explicit','UNCERTAIN':'Mesajul exprimă incertitudine','RETRACTED':'Retrasă explicit'},
        'source':{'DIRECT':'Relatare personală directă','REPORTED':'Relatare despre altcineva','FICTION':'Conținut fictiv'},
        'relation':{'NEW':'Mesaj nou, continuare sau clarificare','RETRACTS':'Retragerea explicită a unei afirmații anterioare'},
        'concern':{'0':'Conținut obișnuit','1':'Suferință emoțională, îngrijorare privind siguranța sau nevoie de sprijin emoțional'},
    },
    'uk': {
        'subject':{'SELF':'Людина, яка говорить','OTHER':'Інша реальна людина','FICTIONAL':'Вигаданий персонаж'},
        'time':{'CURRENT':'Тепер','PAST':'У минулому','HYPOTHETICAL':'Гіпотетична ситуація'},
        'assertion':{'AFFIRMED':'Стверджується в повідомленні','NEGATED':'Прямо заперечується','UNCERTAIN':'У повідомленні висловлено невпевненість','RETRACTED':'Прямо відкликається'},
        'source':{'DIRECT':'Особиста розповідь','REPORTED':'Розповідь про іншу людину','FICTION':'Вигаданий зміст'},
        'relation':{'NEW':'Нове повідомлення, продовження або уточнення','RETRACTS':'Пряме відкликання попереднього твердження'},
        'concern':{'0':'Звичайний зміст','1':'Емоційні труднощі, занепокоєння щодо безпеки або потреба в емоційній підтримці'},
    },
}
NOTES = {
    'ro':'Descrie sensul mesajului. Veridicitatea rămâne neverificată. Dacă nu poți decide, lasă întrebarea fără răspuns.',
    'uk':'Опишіть зміст повідомлення. Правдивість залишається неперевіреною. Якщо не можете визначити відповідь, залиште запитання без відповіді.',
}
CONCERN_NOTES = {
    'ro':'Include și conținutul negat sau fictiv. Negarea, sursa și persoana sunt câmpuri separate.',
    'uk':'Враховуйте також заперечений або вигаданий зміст. Заперечення, джерело та особа — окремі поля.',
}


def question_view(certificate):
    """Return only available answer branches, or None when no question is next."""
    verify(certificate)
    node = certificate['tree']
    if node is None or 'question' not in node:
        return None
    language = certificate['input']['language']; field = node['question']
    if set(OPTIONS[language][field]) != set(ONTOLOGY[field]):
        raise ValueError('Localized answer inventory differs from the interpretation contract')
    return dict(language=language,field=field,prompt=PROMPTS[language][field],
                note=NOTES[language]+(' '+CONCERN_NOTES[language] if field=='concern' else ''),
                options=[dict(value=value,label=OPTIONS[language][field][value]) for value in node['branches']],
                certificate_digest=certificate['certificate_digest'],
                reviewer_authenticated=False,permits_external_action=False,permits_notification=False)
