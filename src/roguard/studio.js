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
let mode = "assess";

const names = {
  ro: {D1: "Recunoașterea semnalului de dezvăluire", R1: "Delimitarea destinatarului minor–tutore", A1: "Accesibilitatea răspunsului pentru vârsta declarată", P1: "Persistența limitelor de divulgare și permisiune", G1: "Poarta de verificare înaintea unei acțiuni", S1: "Integritatea răspunsului de sprijin"},
  uk: {D1: "Ознака звернення дитини по підтримку", R1: "Право одержувача й мета передавання", A1: "Зрозумілість за заявленим віковим контрактом", P1: "Чинність дозволу й стійкість межі", G1: "Людський контроль перед зовнішньою дією", S1: "Повнота й межі підтримувальної відповіді"}
};

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
    JSON.parse(input.value);
    if (mode === "contrast") JSON.parse(afterInput.value);
    body = mode === "contrast" ? `{"before":${input.value},"after":${afterInput.value}}` : input.value;
  } catch (_) {
    placeholder.hidden = true;
    errorBox.hidden = false;
    errorBox.textContent = "Enter valid JSON before running this check.";
    return;
  }
  runButton.disabled = true;
  try {
    const response = await fetch(`/api/${mode}`, {method: "POST", headers: {"Content-Type": "application/json", "X-RoGuard-Token": token}, body, cache: "no-store"});
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
    resultType.textContent = mode === "assess" ? "Contract report" : mode === "explore" ? "Sensitivity report" : "One-fact comparison";
    if (mode === "assess") renderAssess(report);
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
    secondWrap.hidden = mode !== "contrast";
    document.getElementById("input-title").textContent = mode === "contrast" ? "Before" : "Declared contract";
    document.getElementById("mode-hint").textContent = mode === "assess" ? "Checks only the rules you declare. Missing evidence stays incomplete." : mode === "explore" ? "Tests bounded one-fact variants and shows decision flips without input values." : "Change exactly one value between the two contracts.";
    runButton.firstChild.textContent = mode === "assess" ? "Check contract " : mode === "explore" ? "Explore changes " : "Compare contracts ";
    clearResult();
  });
}
document.getElementById("sample-ro").addEventListener("click", () => loadSample("ro"));
document.getElementById("sample-uk").addEventListener("click", () => loadSample("uk"));
runButton.addEventListener("click", run);
loadSample("ro");
