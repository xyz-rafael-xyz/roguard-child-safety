"""Additional Romanian metadata styles for the prospective V26 fact study."""

from training.v25_ro import D1_STYLES as PRIOR_D1, FIELDS, S1_STYLES as PRIOR_S1

D1_STYLES = PRIOR_D1 + (
    {
        "open": "Notă abstractă despre originea semnalului",
        "role": ("proveniența este atribuită copilului", "proveniența este atribuită persoanei adulte"),
        "anchor": ("cadrul de protecție este confirmat", "cadrul de protecție este infirmat"),
        "indirect": ("apare reluarea prudentă a nevoii de ajutor", "reluarea prudentă a nevoii de ajutor lipsește"),
        "explicit": ("apelul direct la sprijin figurează", "apelul direct la sprijin lipsește"),
        "close": "Sunt doar valori fictive ale unor câmpuri.",
    },
    {
        "open": "Matrice demonstrativă de dovezi",
        "role": ("emitentul fictiv aparține grupei minorilor", "emitentul fictiv aparține grupei adulților"),
        "anchor": ("legătura cu siguranța rămâne valabilă", "legătura cu siguranța este invalidă"),
        "indirect": ("este bifată o apropiere treptată de ajutor", "nu este bifată o apropiere treptată de ajutor"),
        "explicit": ("este bifată solicitarea fără ocol de ajutor", "nu este bifată solicitarea fără ocol de ajutor"),
        "close": "Fișa nu conține un mesaj al copilului.",
    },
    {
        "open": "Model fictiv de clasificare a atributelor",
        "role": ("rolul de copil este confirmat pentru sursă", "rolul de adult este confirmat pentru sursă"),
        "anchor": ("indiciul de protecție este acceptat", "indiciul de protecție este respins"),
        "indirect": ("căutarea indirectă de sprijin este validată", "căutarea indirectă de sprijin este respinsă"),
        "explicit": ("cererea directă de sprijin este validată", "cererea directă de sprijin este respinsă"),
        "close": "Nicio relatare personală nu este inclusă.",
    },
)

S1_STYLES = PRIOR_S1 + (
    {
        "open": "Notă abstractă despre verificarea sprijinului",
        "field": "rubrica vizată descrie {}",
        "applicable": ("rubrica este obligatorie în acest caz fictiv", "rubrica nu este obligatorie în acest caz fictiv"),
        "passed": ("controlul rubricii a reușit", "controlul rubricii a eșuat"),
        "close": "Celelalte rubrici obligatorii au reușit; răspunsul nu este reprodus.",
    },
    {
        "open": "Matrice demonstrativă a criteriilor",
        "field": "obiectul măsurării este {}",
        "applicable": ("cerința este în vigoare", "cerința nu este în vigoare"),
        "passed": ("cerința are rezultat conform", "cerința are rezultat neconform"),
        "close": "Restul cerințelor în vigoare sunt conforme; se arată numai metadate.",
    },
    {
        "open": "Model fictiv de audit al răspunsului",
        "field": "punctul urmărit se referă la {}",
        "applicable": ("punctul trebuie evaluat", "punctul nu trebuie evaluat"),
        "passed": ("rezultatul punctului este pozitiv", "rezultatul punctului este negativ"),
        "close": "Orice alt punct cerut are rezultat pozitiv; nu se redă conținutul.",
    },
)
