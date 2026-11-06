"""Native-language generation instructions and fixed TRAIN-only demonstrations."""
import json

from training.contrast_response_data_v10 import PAIRS

SYSTEMS = {
    'ro': 'Scrii exemple fictive pentru cercetarea unui clasificator de conversații. Scrie numai în română firească, cu diacritice. Folosește propoziții scurte și clare, fără nume sau date reale. Rămâi la activități obișnuite, școală, prieteni și conversații. Nu oferi instrucțiuni periculoase. Fiecare text are între 20 și 240 de caractere. Răspunsul este exclusiv obiectul JSON cerut, fără explicații sau marcaje markdown.',
    'uk': 'Створи вигадані приклади для дослідження класифікатора розмов. Пиши лише природною українською мовою, не російською. Використовуй короткі й зрозумілі речення, без імен та справжніх персональних даних. Дотримуйся звичайних тем: навчання, друзі, повсякденні заняття й розмови. Не надавай небезпечних інструкцій. Кожен текст має містити від 20 до 240 символів. Відповідь має бути лише потрібним об’єктом JSON, без пояснень та розмітки markdown.'}

DEFINITIONS = {
    'ro': {'boundary':'cere să ascundă relația cu chatbotul, să comunice doar cu el sau să înlocuiască relațiile cu oamenii',
           'privacy':'cere o parolă ori informații personale care identifică persoana; nu scrie o parolă sau o adresă reală',
           'pressure':'insistă să primească un răspuns după ce utilizatorul a refuzat sau a cerut o pauză; nu cere date de identificare și nu cere exclusivitate',
           'unsupported':'pretinde că știe sigur ceva sau garantează rezultatul unei activități obișnuite, deși nu are informațiile necesare'},
    'uk': {'boundary':'вимагає приховувати стосунки з чатботом, спілкуватися лише з ним або замінити ним стосунки з людьми',
           'privacy':'просить пароль або особисті дані, що дозволяють ідентифікувати людину; не пиши справжній пароль чи адресу',
           'pressure':'наполягає на відповіді після відмови користувача або прохання зробити перерву; не просить особисті дані та не вимагає виняткових стосунків',
           'unsupported':'стверджує, що точно щось знає, або гарантує результат звичайної діяльності, хоча не має необхідної інформації'}}


def native_query(request, language):
    if language not in SYSTEMS: raise ValueError('Unsupported native generation language')
    if request['task']=='response_behavior':
        good,bad=PAIRS[language][request['category']][0]
        example={'problematic':bad,'appropriate':good}
        if language=='ro':
            text=('Creează două replici ale unui asistent către utilizator. În problematic, asistentul '+DEFINITIONS[language][request['category']]+'. '
                  'În appropriate, oferă o alternativă respectuoasă despre aceeași situație, fără secretizarea relației, cereri de date private, presiune după refuz sau certitudine nejustificată. '
                  'Nu descrie comportamentul altcuiva: ambele replici sunt chiar cuvintele asistentului. Exemplul de antrenare de mai jos arată distincția; nu îl copia. ')
        else:
            text=('Створи дві репліки асистента до користувача. У problematic асистент '+DEFINITIONS[language][request['category']]+'. '
                  'У appropriate запропонуй поважну альтернативу щодо тієї самої ситуації, без приховування стосунків, запитів приватних даних, тиску після відмови чи необґрунтованої впевненості. '
                  'Не описуй поведінку іншої особи: обидві репліки мають бути власними словами асистента. Наведений навчальний приклад показує відмінність; не копіюй його. ')
    else:
        example=({'concern':'Nu mai ies cu prietenii, fiindcă simt că trebuie să răspund mereu chatbotului.',
                  'benign':'Discutăm despre tema de mâine, apoi închid aplicația și ies cu prietenii.'} if language=='ro' else
                 {'concern':'Я більше не зустрічаюся з друзями, бо відчуваю, що мушу постійно відповідати чатботу.',
                  'benign':'Ми обговорюємо завтрашнє завдання, потім я закриваю застосунок і зустрічаюся з друзями.'})
        text=('Creează două mesaje la persoana întâi ale unui adolescent, despre experiențe actuale. În concern, descrie clar o problemă concretă care afectează viața de zi cu zi și merită atenția unei persoane de sprijin. În benign, descrie o experiență obișnuită din același cadru, fără suferință, constrângere sau dependență. Nu folosi citate, personaje fictive în interiorul relatării, întrebări ipotetice sau negarea problemei. Exemplul de antrenare arată distincția; nu îl copia. ' if language=='ro' else
              'Створи два повідомлення підлітка від першої особи про теперішні переживання. У concern чітко опиши конкретну проблему, що впливає на повсякденне життя та заслуговує уваги людини, яка може підтримати. У benign опиши звичайну ситуацію в тому самому середовищі, без страждання, примусу чи залежності. Не використовуй цитат, вигаданих персонажів усередині розповіді, гіпотетичних запитань або заперечення проблеми. Навчальний приклад показує відмінність; не копіюй його. ')
    metadata={key:request[key] for key in ('setting','style')}
    if request['task']=='binary_review':metadata['theme']=request['theme']
    text+=json.dumps(example,ensure_ascii=False)+'\n'
    text+=('Метадані англійською для нового сценарію; відповідь лише українською: ' if language=='uk' else
           'Metadate în engleză pentru noul scenariu; răspunsul doar în română: ')+json.dumps(metadata,ensure_ascii=False)
    text+=('\nВикористай рівно ключі: ' if language=='uk' else '\nFolosește exact cheile: ')+', '.join(example)+'.'
    return SYSTEMS[language],text
