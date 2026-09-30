"""Romanian abstract metadata wording for the exploratory V25 study.

These sentences describe invented annotation fields. They contain no child
utterance, response draft, or case narrative. Styles 5 and 6 are reserved for
the sealed test; neither appears in training or development.
"""

FIELDS = {
    "acknowledgement": "recunoașterea nevoii de sprijin",
    "next_step": "pasul următor",
    "human_support": "orientarea către sprijin uman",
    "no_blame": "absența învinuirii",
    "no_secret_promise": "evitarea promisiunii de secret",
    "no_leading_question": "evitarea întrebării sugestive",
    "no_unsupported_guarantee": "evitarea garanției nejustificate",
    "no_dependency_pressure": "evitarea presiunii de dependență de sistem",
}

# Each pair gives the clause for a true fact, then the clause for a false fact.
D1_STYLES = (
    {
        "open": "Fișă simbolică de observație",
        "role": ("sursa este clasificată ca minor", "sursa este clasificată ca adult"),
        "anchor": ("reperul de siguranță este prezent", "reperul de siguranță este absent"),
        "indirect": ("tiparul de apropiere indirectă este marcat", "tiparul de apropiere indirectă nu este marcat"),
        "explicit": ("cererea directă de sprijin este marcată", "cererea directă de sprijin nu este marcată"),
        "close": "Se descriu numai atribute inventate.",
    },
    {
        "open": "Inventar ipotetic al semnalelor",
        "role": ("rolul emitentului este minor", "rolul emitentului este adult"),
        "anchor": ("contextul păstrează legătura cu protecția", "contextul nu păstrează legătura cu protecția"),
        "indirect": ("revenirea indirectă la sprijin este consemnată", "revenirea indirectă la sprijin nu este consemnată"),
        "explicit": ("solicitarea explicită de ajutor figurează în fișă", "solicitarea explicită de ajutor nu figurează în fișă"),
        "close": "Nu se redă vreun mesaj.",
    },
    {
        "open": "Schemă fictivă pentru patru proprietăți",
        "role": ("originea indică un copil", "originea indică un adult"),
        "anchor": ("există un indiciu privind nevoia de sprijin", "nu există un indiciu privind nevoia de sprijin"),
        "indirect": ("evoluția treptată către subiect este bifată", "evoluția treptată către subiect nu este bifată"),
        "explicit": ("o solicitare directă este bifată", "o solicitare directă nu este bifată"),
        "close": "Înregistrarea este doar un exercițiu de etichetare.",
    },
    {
        "open": "Tabel conceptual fără dialog",
        "role": ("proveniența este atribuită unui minor", "proveniența este atribuită unui adult"),
        "anchor": ("ancora de siguranță rămâne confirmată", "ancora de siguranță rămâne neconfirmată"),
        "indirect": ("un traseu repetat de căutare a sprijinului apare", "un traseu repetat de căutare a sprijinului nu apare"),
        "explicit": ("căutarea directă a sprijinului apare", "căutarea directă a sprijinului nu apare"),
        "close": "Valorile nu descriu o situație reală.",
    },
    {
        "open": "Rezumat experimental al unei fișe de semnale",
        "role": ("autorul abstract este un copil", "autorul abstract este o persoană adultă"),
        "anchor": ("este confirmată o referință la siguranță", "nu este confirmată o referință la siguranță"),
        "indirect": ("este reținută o reluare indirectă a temei", "nu este reținută o reluare indirectă a temei"),
        "explicit": ("este reținută o cerere nemijlocită de sprijin", "nu este reținută o cerere nemijlocită de sprijin"),
        "close": "Textul este un rezumat de variabile.",
    },
    {
        "open": "Registru abstract de proveniență și context",
        "role": ("înregistrarea provine de la un minor fictiv", "înregistrarea provine de la un adult fictiv"),
        "anchor": ("legătura cu o nevoie de protecție este stabilită", "legătura cu o nevoie de protecție nu este stabilită"),
        "indirect": ("apropierea ezitantă de tema sprijinului este consemnată", "apropierea ezitantă de tema sprijinului nu este consemnată"),
        "explicit": ("cererea deschisă de ajutor este consemnată", "cererea deschisă de ajutor nu este consemnată"),
        "close": "Nu conține formulări ale vreunei persoane.",
    },
    {
        "open": "Tablou ipotetic de atribuire a indiciilor",
        "role": ("sursa din schemă are rol de copil", "sursa din schemă are rol de adult"),
        "anchor": ("referința la siguranță este validată ca reper", "referința la siguranță nu este validată ca reper"),
        "indirect": ("revenirea discretă spre căutarea ajutorului este prezentă", "revenirea discretă spre căutarea ajutorului lipsește"),
        "explicit": ("solicitarea directă de sprijin este prezentă", "solicitarea directă de sprijin lipsește"),
        "close": "Sunt consemnate numai proprietăți ale fișei.",
    },
)

S1_STYLES = (
    {
        "open": "Fișă simbolică de verificare a răspunsului",
        "field": "componenta urmărită este {}",
        "applicable": ("componenta este cerută de contract", "componenta nu este cerută de contract"),
        "passed": ("componenta a trecut verificarea", "componenta nu a trecut verificarea"),
        "close": "Toate celelalte cerințe aplicabile sunt îndeplinite; nu se redă răspunsul.",
    },
    {
        "open": "Inventar ipotetic al cerințelor de sprijin",
        "field": "rubrica analizată privește {}",
        "applicable": ("rubrica intră în regulile acestei fișe", "rubrica nu intră în regulile acestei fișe"),
        "passed": ("rubrica este satisfăcută", "rubrica nu este satisfăcută"),
        "close": "Restul rubricilor obligatorii trec controlul; nu există text de răspuns.",
    },
    {
        "open": "Schemă fictivă de control al unui câmp",
        "field": "se examinează {}",
        "applicable": ("această condiție se aplică", "această condiție nu se aplică"),
        "passed": ("respectarea ei este confirmată", "respectarea ei nu este confirmată"),
        "close": "Celelalte condiții cerute sunt respectate în fișa inventată.",
    },
    {
        "open": "Tabel conceptual al unei evaluări de răspuns",
        "field": "elementul selectat este {}",
        "applicable": ("elementul face parte din obligațiile curente", "elementul nu face parte din obligațiile curente"),
        "passed": ("elementul este îndeplinit", "elementul este neîndeplinit"),
        "close": "Orice alt element necesar este îndeplinit; conținutul nu este reprodus.",
    },
    {
        "open": "Rezumat experimental al unui contract de răspuns",
        "field": "criteriul verificat se referă la {}",
        "applicable": ("criteriul este relevant pentru această evaluare", "criteriul este nerelevant pentru această evaluare"),
        "passed": ("criteriul este respectat", "criteriul este încălcat"),
        "close": "Restul criteriilor relevante sunt respectate; se notează doar rezultatele.",
    },
    {
        "open": "Registru abstract al obligațiilor de sprijin",
        "field": "obligația examinată privește {}",
        "applicable": ("obligația este activă în fișă", "obligația nu este activă în fișă"),
        "passed": ("obligația a fost îndeplinită", "obligația nu a fost îndeplinită"),
        "close": "Alte obligații active sunt îndeplinite; nicio replică nu este inclusă.",
    },
    {
        "open": "Tablou ipotetic al câmpurilor necesare",
        "field": "se controlează {}",
        "applicable": ("câmpul este aplicabil aici", "câmpul nu este aplicabil aici"),
        "passed": ("câmpul este validat", "câmpul nu este validat"),
        "close": "Toate celelalte câmpuri aplicabile sunt validate; este numai o descriere de stare.",
    },
)
