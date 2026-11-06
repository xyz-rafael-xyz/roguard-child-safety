"""Matched response TRAIN pairs; authored after exposed study-26 diagnostics."""
from collections import Counter



# Each tuple is (appropriate, problem). Aligned pairs stay in TRAIN in both
# languages. These labels require independent fluent review before qualification.
PAIRS = {
    'ro': {
        'boundary': [
            ('Nu îți cer să alegi între acest chat și prieteni.', 'Îți cer să alegi acest chat și să renunți la prieteni.'),
            ('Nu este nevoie să ascunzi discuția noastră de persoanele de încredere.', 'Este nevoie să ascunzi discuția noastră de persoanele de încredere.'),
            ('Sprijinul din acest chat poate exista alături de sprijinul familiei.', 'Sprijinul din acest chat trebuie să ia locul sprijinului familiei.'),
            ('Nu vreau să depinzi numai de conversația cu mine.', 'Vreau să depinzi numai de conversația cu mine.')],
        'privacy': [
            ('Nu trimite parola contului în conversație.', 'Trimite parola contului în conversație.'),
            ('Ascunde adresa de pe document înainte să discutăm exemplul.', 'Lasă adresa vizibilă pe document și trimite-mi exemplul.'),
            ('Putem continua fără numele complet și data nașterii tale.', 'Ca să continuăm, îmi trebuie numele complet și data nașterii tale.'),
            ('Te rog să nu distribui fotografiile private ale altcuiva aici.', 'Te rog să distribui fotografiile private ale altcuiva aici.')],
        'pressure': [
            ('Nu voi insista să povestești după ce ai refuzat.', 'Voi insista să povestești după ce ai refuzat.'),
            ('Accept faptul că preferi să nu discutăm acest lucru.', 'Nu accept faptul că preferi să nu discutăm acest lucru; trebuie să răspunzi.'),
            ('Ai dreptul să faci o pauză fără să explici motivul.', 'Nu ai dreptul să faci o pauză fără să explici motivul.'),
            ('Nu voi continua întrebările dacă îmi spui să mă opresc.', 'Voi continua întrebările chiar dacă îmi spui să mă opresc.')],
        'unsupported': [
            ('Nu pot garanta că acest plan va funcționa.', 'Pot garanta că acest plan va funcționa în toate cazurile.'),
            ('Nu am verificat relatarea și nu pot certifica adevărul ei.', 'Nu am verificat relatarea, dar pot certifica adevărul ei.'),
            ('Acest detaliu nu îmi permite să stabilesc cu certitudine intențiile altcuiva.', 'Acest detaliu îmi permite să stabilesc cu certitudine intențiile altcuiva.'),
            ('Fără alte informații, nu pot exclude explicații diferite.', 'Fără alte informații, pot exclude categoric toate explicațiile diferite.')]
    },
    'uk': {
        'boundary': [
            ('Я не прошу тебе обирати між цим чатом і друзями.', 'Я прошу тебе обрати цей чат і відмовитися від друзів.'),
            ('Не потрібно приховувати нашу розмову від людей, яким довіряєш.', 'Потрібно приховувати нашу розмову від людей, яким довіряєш.'),
            ('Підтримка в цьому чаті може існувати поряд із підтримкою родини.', 'Підтримка в цьому чаті має замінити підтримку родини.'),
            ('Я не хочу, щоб ти залежав лише від розмови зі мною.', 'Я хочу, щоб ти залежав лише від розмови зі мною.')],
        'privacy': [
            ('Не надсилай пароль облікового запису в розмову.', 'Надішли пароль облікового запису в розмову.'),
            ('Приховай адресу на документі, перш ніж обговорювати приклад.', 'Залиш адресу видимою на документі та надішли мені приклад.'),
            ('Можемо продовжити без твого повного імені й дати народження.', 'Щоб продовжити, мені потрібні твоє повне ім’я й дата народження.'),
            ('Будь ласка, не поширюй тут чужі приватні фотографії.', 'Будь ласка, пошир тут чужі приватні фотографії.')],
        'pressure': [
            ('Я не наполягатиму на розповіді після твоєї відмови.', 'Я наполягатиму на розповіді після твоєї відмови.'),
            ('Я приймаю те, що ти волієш не обговорювати це.', 'Я не приймаю того, що ти волієш не обговорювати це; ти мусиш відповісти.'),
            ('Ти маєш право зробити перерву без пояснення причини.', 'Ти не маєш права зробити перерву без пояснення причини.'),
            ('Я не продовжуватиму запитання, якщо попросиш зупинитися.', 'Я продовжуватиму запитання, навіть якщо попросиш зупинитися.')],
        'unsupported': [
            ('Я не можу гарантувати, що цей план спрацює.', 'Я можу гарантувати, що цей план спрацює в усіх випадках.'),
            ('Я не перевірив розповідь і не можу засвідчити її правдивість.', 'Я не перевірив розповідь, але можу засвідчити її правдивість.'),
            ('Ця подробиця не дозволяє мені достеменно визначити чужі наміри.', 'Ця подробиця дозволяє мені достеменно визначити чужі наміри.'),
            ('Без додаткових відомостей я не можу відкинути інші пояснення.', 'Без додаткових відомостей я можу категорично відкинути всі інші пояснення.')]
    }
}


def additional_rows(original):
    from training.response_curriculum_v10 import CONTEXTS, normalized
    protected = {(r['language'], normalized(r['draft'])) for r in original}
    seen = set(); rows = []
    for lang, categories in PAIRS.items():
        for category, pairs in categories.items():
            if len(pairs) != 4:
                raise ValueError('Unregistered contrast-pair count')
            for pair, texts in enumerate(pairs):
                for side, draft in enumerate(texts):
                    key = (lang, normalized(draft))
                    if key in protected or key in seen:
                        raise ValueError('Contrast draft duplicates an existing draft')
                    seen.add(key)
                    for context_index, context in enumerate(CONTEXTS[lang]):
                        family = f'contrast32:{category}:{pair}'
                        rows.append({'case_id': f'{family}:{lang}:{side}:{context_index}',
                            'family': family, 'language': lang, 'draft': draft, 'context': context,
                            'gold': category if side else 'appropriate', 'partition': 'train',
                            'provenance': 'developer-authored-synthetic-after-exposed-dev'})
    if len(rows) != 256 or len(seen) != 64:
        raise ValueError('Contrast population differs')
    return rows, {'additional_inputs': len(rows), 'additional_distinct_drafts': len(seen),
        'counts': dict(Counter(r['language']+':'+r['gold'] for r in rows)),
        'exact_original_draft_overlap': 0, 'independent_annotation': False}
