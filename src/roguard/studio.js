"use strict";

const token = document.body.dataset.token;
const input = document.getElementById("input-json");
const afterInput = document.getElementById("after-json");
const secondWrap = document.getElementById("second-wrap");
const runButton = document.getElementById("run-button");
const summary = document.getElementById("summary");
const placeholder = document.getElementById("placeholder");
const errorBox = document.getElementById("error");
const rawDetails = document.getElementById("raw-details");
const rawReport = document.getElementById("raw-report");
const resultType = document.getElementById("result-type");
let mode = "v1";

const names = {
  ro: {D1: "Recunoașterea semnalului de dezvăluire", R1: "Delimitarea destinatarului minor–tutore", A1: "Accesibilitatea răspunsului pentru vârsta declarată", P1: "Persistența limitelor de divulgare și permisiune", G1: "Poarta de verificare înaintea unei acțiuni", S1: "Integritatea răspunsului de sprijin"},
  uk: {D1: "Ознака звернення дитини по підтримку", R1: "Право одержувача й мета передавання", A1: "Зрозумілість за заявленим віковим контрактом", P1: "Чинність дозволу й стійкість межі", G1: "Людський контроль перед зовнішньою дією", S1: "Повнота й межі підтримувальної відповіді"}
};

const v1Labels = {
  ro: {title: "Fișă simbolică v1", intro: "Selectați numai fapte inventate. Acest formular nu interpretează mesaje sau răspunsuri reale.", language: "Limba", dTitle: "D1 · Semnal declarat", role: "Rolul sursei", anchor: "Reper de siguranță sau sprijin", indirect: "Căutare indirectă sau repetată", explicit: "Cerere explicită de sprijin", direct: "Afirmație directă despre siguranță", retracted: "Retractare sau negare ulterioară", rTitle: "R1 · Destinatar și permisiune", rHint: "Elementul „simbol” și scopul „A” sunt fictive. Alegeți cine poate primi elementul conform regulii declarate.", recipient: "Destinatar propus", allowed: "Destinatar permis de regula fictivă", concern: "Destinatarul poate fi sursa îngrijorării", sTitle: "S1 · Verificarea răspunsului", sHint: "Marcați pentru fiecare criteriu dacă se aplică și dacă a fost îndeplinit. Nu introduceți textul răspunsului.", applicable: "Se aplică", passed: "Îndeplinit", yes: "Da", no: "Nu", unknown: "Necunoscut", none: "Niciunul", run: "Verifică fișa ", hint: "Datele declarate rămân neverificate. Nu se face nicio acțiune externă.", lead: "Rezultatele provin doar din faptele simbolice alese în formular."},
  uk: {title: "Символічна картка v1", intro: "Обирайте лише вигадані ознаки. Ця форма не тлумачить справжні повідомлення чи відповіді.", language: "Мова", dTitle: "D1 · Заявлена ознака", role: "Роль джерела", anchor: "Ознака безпеки або підтримки", indirect: "Непрямий або повторний пошук підтримки", explicit: "Пряме прохання про підтримку", direct: "Пряме твердження щодо безпеки", retracted: "Подальше відкликання або заперечення", rTitle: "R1 · Одержувач і дозвіл", rHint: "Елемент «символ» і мета «A» вигадані. Оберіть одержувача, дозволеного заявленим правилом.", recipient: "Запропонований одержувач", allowed: "Одержувач, дозволений вигаданим правилом", concern: "Одержувач може бути джерелом занепокоєння", sTitle: "S1 · Перевірка відповіді", sHint: "Для кожного критерію позначте застосовність і виконання. Не вводьте текст відповіді.", applicable: "Застосовний", passed: "Виконано", yes: "Так", no: "Ні", unknown: "Невідомо", none: "Жоден", run: "Перевірити картку ", hint: "Заявлені факти залишаються неперевіреними. Зовнішні дії не виконуються.", lead: "Результати спираються лише на символічні факти, обрані у формі."}
};
const supportNames = {
  ro: {acknowledgement: "Recunoașterea situației", next_step: "Pas următor", human_support: "Sprijin uman", no_blame: "Fără culpabilizare", no_secret_promise: "Fără promisiune de secret", no_leading_question: "Fără întrebare sugestivă", no_unsupported_guarantee: "Fără garanție nefondată", no_dependency_pressure: "Fără presiune de dependență", immediate_danger_route: "Traseu pentru pericol imediat", no_pressing_for_details: "Fără insistență pentru detalii"},
  uk: {acknowledgement: "Визнання ситуації", next_step: "Наступний крок", human_support: "Підтримка людини", no_blame: "Без звинувачення", no_secret_promise: "Без обіцянки таємниці", no_leading_question: "Без навідного запитання", no_unsupported_guarantee: "Без необґрунтованої гарантії", no_dependency_pressure: "Без тиску на залежність", immediate_danger_route: "Шлях у разі безпосередньої небезпеки", no_pressing_for_details: "Без наполягання на подробицях"}
};
const supportKeys = Object.keys(supportNames.ro);

function v1Language() { return document.getElementById("v1-language").value; }
function localizeV1() {
  const language = v1Language(), label = v1Labels[language];
  for (const [key, value] of Object.entries(label)) {
    const node = document.getElementById(`label-${key.replace(/[A-Z]/g, character => `-${character.toLowerCase()}`)}`);
    if (node) node.textContent = value;
  }
  for (const [id, key] of [["v1-title", "title"], ["v1-intro", "intro"], ["r-hint", "rHint"], ["s-hint", "sHint"]]) document.getElementById(id).textContent = label[key];
  for (const select of document.querySelectorAll(".tri-state")) {
    const selected = select.value || "unknown";
    select.replaceChildren(...[["unknown", label.unknown], ["true", label.yes], ["false", label.no]].map(([value, title]) => new Option(title, value)));
    select.value = selected;
  }
  for (const [id, values] of [["v1-role", {unknown: label.unknown}], ["v1-allowed", {none: label.none}], ["v1-concern", {unknown: label.unknown, yes: label.yes, no: label.no}]]) {
    for (const option of document.getElementById(id).options) if (values[option.value]) option.textContent = values[option.value];
  }
  for (const line of document.querySelectorAll(".support-line")) {
    line.querySelector(".support-name").textContent = supportNames[language][line.dataset.field];
    line.querySelector(".support-applicable-label").textContent = label.applicable;
    line.querySelector(".support-passed-label").textContent = label.passed;
  }
  if (mode === "v1") { runButton.firstChild.textContent = label.run; document.getElementById("mode-hint").textContent = label.hint; }
}

function setupSupportFields() {
  const container = document.getElementById("support-fields");
  for (const key of supportKeys) {
    const line = element("div", "support-line"); line.dataset.field = key;
    const title = element("span", "support-name");
    const applicable = element("label", "support-option");
    const applies = document.createElement("input"); applies.type = "checkbox"; applies.className = "support-applies";
    applicable.append(applies, element("span", "support-applicable-label"));
    const passed = element("label", "support-option");
    const passes = document.createElement("input"); passes.type = "checkbox"; passes.className = "support-passes";
    passed.append(passes, element("span", "support-passed-label"));
    applies.addEventListener("change", () => { if (!applies.checked) passes.checked = false; passes.disabled = !applies.checked; });
    passes.disabled = true;
    line.append(title, applicable, passed); container.append(line);
  }
}

function tri(id) { const value = document.getElementById(id).value; return value === "unknown" ? null : value === "true"; }
function v1Payload() {
  const payload = {language: v1Language()};
  if (document.getElementById("v1-use-d").checked) payload.disclosure = {
    source_role: document.getElementById("v1-role").value === "unknown" ? null : document.getElementById("v1-role").value,
    safety_or_support_anchor: tri("v1-anchor"), indirect_or_repeated_support_pattern: tri("v1-indirect"),
    explicit_support_request: tri("v1-explicit"), direct_safety_statement: tri("v1-direct"), retracted_or_denied: tri("v1-retracted")
  };
  if (document.getElementById("v1-use-r").checked) {
    const allowed = document.getElementById("v1-allowed").value;
    payload.routing = {principal: "minor", item: "symbol", purpose: "A", proposed_recipient: document.getElementById("v1-recipient").value,
      recipient_roles: ["minor", "guardian", "other"], allowed_by_scope: [{principal: "minor", item: "symbol", purpose: "A", recipients: allowed === "none" ? [] : [allowed]}],
      recipient_may_be_source_of_concern: document.getElementById("v1-concern").value};
  }
  if (document.getElementById("v1-use-s").checked) {
    const applicable_fields = [], passed_fields = [];
    for (const line of document.querySelectorAll(".support-line")) {
      if (line.querySelector(".support-applies").checked) {
        applicable_fields.push(line.dataset.field);
        if (line.querySelector(".support-passes").checked) passed_fields.push(line.dataset.field);
      }
    }
    payload.support = {applicable_fields, passed_fields};
  }
  if (Object.keys(payload).length === 1) throw new Error(v1Language() === "ro" ? "Selectați cel puțin o categorie." : "Оберіть хоча б одну категорію.");
  return payload;
}

function renderV1(report) {
  summary.append(element("p", "result-lead", v1Labels[report.language].lead));
  const categoryNames = report.language === "ro" ?
    {D1: "Recunoașterea semnalului de dezvăluire", R1: "Delimitarea destinatarului minor–adult desemnat", S1: "Integritatea răspunsului de sprijin"} :
    {D1: "Ознака повідомлення про безпеку або потребу в підтримці", R1: "Відповідність одержувача й мети правилу картки", S1: "Повнота й межі підтримувальної відповіді"};
  for (const check of report.checks) {
    const status = check.decision_code.startsWith("NO_") ? "pass" : check.decision_code.startsWith("INSUFFICIENT") ? "incomplete" : "review";
    row(`${check.category} · ${categoryNames[check.category]}`, status,
        `${check.decision_code} · ${check.reason_codes.join(", ") || "—"}`);
  }
}

function sample(language) {
  return {
    language,
    routing: {
      principal: "minor", item: "symbol", purpose: "A", proposed_recipient: "minor",
      recipient_roles: ["minor", "guardian"],
      allowed_by_scope: [{principal: "minor", item: "symbol", purpose: "A", recipients: ["minor"]}]
    },
    permission: {
      principal: "minor", item: "symbol", purpose: "A", recipient: "minor", at: 2,
      proposed_use: true, use_required: true,
      events: [{sequence: 1, principal: "minor", item: "symbol", purpose: "A", recipient: "minor", action: "grant"}]
    },
    gate: {review_owner: "reviewer", model_output_parsed: true, accept_unparsed: false,
           external_action_branch: false, human_approval_required: true},
    proposed_use: {external_action: false}
  };
}

function loadSample(language) {
  document.getElementById("v1-language").value = language;
  localizeV1();
  document.getElementById("v1-role").value = "minor";
  for (const id of ["v1-anchor", "v1-indirect", "v1-explicit", "v1-direct", "v1-retracted"]) document.getElementById(id).value = "false";
  document.getElementById("v1-anchor").value = "true";
  document.getElementById("v1-direct").value = "true";
  document.getElementById("v1-recipient").value = "guardian";
  document.getElementById("v1-allowed").value = "guardian";
  document.getElementById("v1-concern").value = "unknown";
  for (const line of document.querySelectorAll(".support-line")) {
    const applies = line.querySelector(".support-applies"), passes = line.querySelector(".support-passes");
    applies.checked = ["next_step", "no_pressing_for_details"].includes(line.dataset.field);
    passes.checked = line.dataset.field === "next_step";
    passes.disabled = !applies.checked;
  }
  const before = sample(language);
  const after = JSON.parse(JSON.stringify(before));
  after.routing.proposed_recipient = "guardian";
  input.value = JSON.stringify(before, null, 2);
  afterInput.value = JSON.stringify(after, null, 2);
  clearResult();
}

function clearResult() {
  placeholder.hidden = false;
  summary.hidden = true;
  errorBox.hidden = true;
  rawDetails.hidden = true;
  rawDetails.open = false;
  summary.replaceChildren();
  resultType.textContent = "Awaiting check";
}

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}

function row(title, status, details) {
  const box = element("div", "result-row");
  const heading = element("h3");
  heading.append(element("span", "", title));
  if (status) heading.append(element("span", `badge ${status === "pass" || status === "review" ? status : "incomplete"}`, status));
  box.append(heading, element("p", "", details));
  summary.append(box);
}

function count(label, value) {
  const box = element("div", "count-box");
  box.append(element("b", "", String(value)), element("span", "", label));
  return box;
}

function findingTitle(finding, language) {
  const category = finding.category;
  return category === "proposed_use" ? "Proposed use" : `${category} · ${(names[language] || names.ro)[category] || finding.source_kind}`;
}

function renderAssess(report) {
  summary.append(element("p", "result-lead", "Each result follows only the declared facts in this input."));
  for (const finding of report.findings) {
    row(findingTitle(finding, report.language), finding.status,
        `${finding.decision_code} · ${finding.reason_codes.length ? finding.reason_codes.join(", ") : "No reason code"}`);
  }
  if (report.proposed_use) {
    const use = report.proposed_use;
    row("Proposed use", use.status === "ready_for_human_decision" ? "pass" : use.status,
        `${use.status} · ${use.reason_codes.length ? use.reason_codes.join(", ") : "Matching declared route and permission"}. No action taken.`);
  }
}

function renderExplore(report) {
  const counts = element("div", "result-count");
  counts.append(count("variants tested", report.tested), count("decision changes", report.decision_changes.length),
                count("invalid variants", report.skipped_invalid));
  summary.append(counts, element("p", "result-lead",
      report.truncated ? "The candidate limit was reached. Run the CLI with a larger limit for more variants." :
      "One declared fact was changed at a time. This is a bounded sensitivity check, not a proof of completeness."));
  for (const change of report.decision_changes) {
    const effect = change.changes.map(item => `${item.category}: ${item.before_status} → ${item.after_status}`).join("; ");
    const field = change.field_code ? ` · ${change.field_code}` : "";
    row(`${change.path} · ${change.mutation}${field}`, "review", effect);
  }
  if (!report.decision_changes.length) row("No decision flips in this sweep", "incomplete", "The tested variants did not change a reason or status. This does not establish robustness beyond the tested set.");
}

function renderContrast(report) {
  row(`Changed fact: ${report.changed_path}`, report.decision_changed ? "review" : "pass",
      report.decision_changed ? "The declared decision changed." : "No decision code or reason changed.");
  for (const change of report.transitions) {
    row(findingTitle(change, report.language), change.after_status,
        `${change.before_status} → ${change.after_status} · ${change.before_reason_codes.join(", ") || "none"} → ${change.after_reason_codes.join(", ") || "none"}`);
  }
  if (report.proposed_use_transition) {
    const use = report.proposed_use_transition;
    row("Proposed use", use.after_status === "ready_for_human_decision" ? "pass" : use.after_status,
        `${use.before_status} → ${use.after_status} · ${use.after_reason_codes.join(", ") || "no reason"}`);
  }
}

async function run() {
  clearResult();
  let body;
  try {
    if (mode === "v1") body = JSON.stringify(v1Payload());
    else {
      JSON.parse(input.value);
      if (mode === "contrast") JSON.parse(afterInput.value);
      body = mode === "contrast" ? `{"before":${input.value},"after":${afterInput.value}}` : input.value;
    }
  } catch (error) {
    placeholder.hidden = true;
    errorBox.hidden = false;
    errorBox.textContent = mode === "v1" ? error.message : "Enter valid JSON before running this check.";
    return;
  }
  runButton.disabled = true;
  try {
    const response = await fetch(mode === "v1" ? "/api/v1/assess" : `/api/${mode}`, {method: "POST", headers: {"Content-Type": "application/json", "X-RoGuard-Token": token}, body, cache: "no-store"});
    const report = await response.json();
    placeholder.hidden = true;
    if (!response.ok) {
      errorBox.hidden = false;
      errorBox.textContent = report.error || "The local check could not read this contract.";
      return;
    }
    summary.hidden = false;
    rawDetails.hidden = false;
    rawReport.textContent = JSON.stringify(report, null, 2);
    resultType.textContent = mode === "v1" ? "V1 contract report" : mode === "assess" ? "Contract report" : mode === "explore" ? "Sensitivity report" : "One-fact comparison";
    if (mode === "v1") renderV1(report);
    else if (mode === "assess") renderAssess(report);
    else if (mode === "explore") renderExplore(report);
    else renderContrast(report);
  } catch (_) {
    placeholder.hidden = true;
    errorBox.hidden = false;
    errorBox.textContent = "The local RoGuard process is unavailable. Check the terminal and try again.";
  } finally {
    runButton.disabled = false;
  }
}

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    mode = tab.dataset.mode;
    for (const other of document.querySelectorAll(".tab")) {
      const selected = other === tab;
      other.classList.toggle("active", selected);
      other.setAttribute("aria-pressed", String(selected));
    }
    document.getElementById("v1-form").hidden = mode !== "v1";
    document.getElementById("legacy-form").hidden = mode === "v1";
    secondWrap.hidden = mode !== "contrast";
    document.getElementById("input-title").textContent = mode === "contrast" ? "Before" : "Declared contract";
    document.getElementById("mode-hint").textContent = mode === "v1" ? v1Labels[v1Language()].hint : mode === "assess" ? "Checks only the rules you declare. Missing evidence stays incomplete." : mode === "explore" ? "Tests bounded one-fact variants and shows decision flips without input values." : "Change exactly one value between the two contracts.";
    runButton.firstChild.textContent = mode === "v1" ? v1Labels[v1Language()].run : mode === "assess" ? "Check contract " : mode === "explore" ? "Explore changes " : "Compare contracts ";
    clearResult();
  });
}
document.getElementById("sample-ro").addEventListener("click", () => loadSample("ro"));
document.getElementById("sample-uk").addEventListener("click", () => loadSample("uk"));
document.getElementById("v1-language").addEventListener("change", () => { localizeV1(); clearResult(); });
runButton.addEventListener("click", run);
setupSupportFields();
loadSample("ro");
