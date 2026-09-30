# RoGuard — Romanian and Ukrainian child-safety research toolkit

**Status: working Romanian and Ukrainian structured-check toolkit with experimental model research.** The read-only CLI and typed contract checks run in either language. Romanian RoMistral, Qwen, mmBERT, and BERT adapters have been measured on abstract synthetic cards. A [fresh mixed-input Romanian study](BENCHMARK.md#frozen-romanian-mixed-input-workflow-batch-0026) passed its registered symbolic target at 72/72 one-fact pairs: D1/S1 used a frozen model on abstract card text, while four policy categories received explicit typed facts. The same text-only model got 55/72 pairs and missed every A1 positive. This is evidence for the declared-fact workflow, not six-category text understanding. A separate Ukrainian prompt-only probe matched 21/48 symbolic cards and no complete pairs; a later [binary-prompt diagnostic](docs/UK_BINARY_PROMPT_DIAGNOSTIC.md) parsed every task but reached at most 5/24 pairs on those consumed cards. Neither language has a validated detector of real disclosures. Nothing reports, notifies, or changes an account automatically.

The [release-gate record](docs/RELEASE_GATES.md) tracks which project requirements are met and which remain open. This public source release includes the working structured checker, synthetic research records, and model metadata. Trained adapter weights and their Git history remain in a separate private research archive; the [public-release note](docs/PUBLIC_RELEASE.md) explains what can be verified from this checkout.

## Download and run

The repository is public. You need Git, Python 3.10 or newer, and a terminal. On macOS or Linux:

```sh
git clone https://github.com/xyz-rafael-xyz/roguard-child-safety.git
cd roguard-child-safety
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m roguard examples/contracts_ro.json
.venv/bin/python -m roguard examples/contracts_uk.json
```

On Windows, use `py -3.12 -m venv .venv`, then replace `.venv/bin/python` with `.venv\Scripts\python.exe` in the last three commands.

Both example commands print a JSON report for fictional, caller-declared facts. To check your own case, copy an [example input](examples/contracts_ro.json), edit its declared fields, and pass the new JSON file to `roguard` using the same command. The [CLI guide](docs/CLI.md) explains the input fields and additional commands. This install has no model download and does not screen child messages; the language models are separate experimental research artifacts. No browser interface, package-index release, or one-click installer is available yet.

The latest [bilingual synthetic metadata study](BENCHMARK.md#bilingual-same-origin-metadata-studies-v25v29) trained one Romanian/Ukrainian D1/S1 encoder and scored it once on a sealed test: **165/192 complete one-fact pairs**, below its fixed per-cell target. Romanian D1 was 43/48, Romanian S1 48/48, Ukrainian D1 36/48, and Ukrainian S1 38/48. Four further registered designs, including factorized and entailment hybrids, failed their development gates, so their tests remain sealed. The new cards describe only invented annotation fields; these results do not validate a child-message detector or Ukrainian language quality.

For a local human-reviewed D1 workflow that stores no message, use the [Romanian](examples/reviewer_evidence_ro.json) or [Ukrainian](examples/reviewer_evidence_uk.json) evidence template. Missing fields remain `incomplete`; the [CLI guide](docs/CLI.md#compare-one-declared-fact) shows one-fact contrasts. This checks what a reviewer declares, not the accuracy of the reviewer's language judgment.

To check whether one fictional use has consistent routing and current permission, run `roguard examples/proposed_use_ro.json`. The optional `proposed_use` report requires both contracts to describe the same principal, item, purpose, and recipient; external branches also require a matching human-review gate. Even a `ready_for_human_decision` result is only a local check of caller-supplied facts and takes no action. See [the combined-use guide](docs/CLI.md#check-one-proposed-use-across-contracts).

Two separately prepared D1/S1 evidence cards can be compared with `roguard-agreement first.json second.json`. A disagreement yields field paths for adjudication and no joint finding; matching incomplete cards stay incomplete. The tool cannot verify reviewer independence or correctness. See the [agreement CLI guide](docs/CLI.md#check-agreement-between-two-evidence-declarations).

For D1, `roguard-review-card --language ro > first.json` (or `--language uk`) asks only numbered questions and writes a content-free card. Add `--category S1` for numeric checks of applicable support fields. A second reviewer can prepare another file for `roguard-agreement`. Unknown answers remain incomplete, and the command does not accept or store the underlying message or response.

For a blinded collection of permitted abstract cards, `roguard-review-card --language ro --item-id abstract_01 --compact` emits one JSONL row. `roguard-agreement-batch` matches two reviewer streams by opaque item ID and reports decision agreement without card values. [The CLI guide](docs/CLI.md#audit-agreement-across-a-reviewer-batch) includes synthetic demonstration files. No human annotation study has been completed yet.

`roguard-blind-packets --batch batch-0035 --category D1 --output-dir review_runs/pilot-0035` prepares two differently ordered local abstract-card packets without reference labels and keeps the ID map in an owner-only file. It is preparation for independent review, not a completed review or a live-message intake path.

`roguard-review-audit` can then bind two answer streams to the exact attested packets before calculating agreement. It verifies file and item alignment; it cannot verify reviewer independence or label correctness.

`roguard-annotate-packet review_runs/reviewer-a.jsonl review_runs/reviewer-a-answers.jsonl` guides a reviewer through a blinded abstract packet with numeric D1/S1 questions, saves each completed content-free answer, and resumes after interruption only when the packet bytes still match a private SHA-256 binding beside the answers. Send both files to the study owner; independent-batch audits require both reviewers' bindings. It displays the permitted abstract cards locally; it never writes their text to the answer stream. The `review_runs/` directory is ignored by Git.

After independent annotation and separate adjudication, `roguard-human-eval-registered` compares a previously frozen model prediction file with the adjudicated declarations while auditing packet, answer, private packet-binding bytes, and the pre-author model identifier through the final report. It cannot attest which model actually ran or when. The original `roguard-human-eval` command remains pinned for the pre-author V16 input-integrity record. Both report abstentions and human contrast disagreements separately from model errors. No actual human review result exists yet; the [CLI guide](docs/CLI.md#audit-agreement-across-a-reviewer-batch) gives the file contract.

For a new, human-approved independent Romanian abstract batch, `roguard-predict-abstract` scores only an opaque shuffled packet with a pinned mmBERT research adapter at its fixed cutoff. The predictor cannot see the source labels or verify their approval; `roguard-human-eval-registered` later verifies the packet and model identifier against the attested batch and rejects source-ordered predictions for independent tests. The [CLI guide](docs/CLI.md#audit-agreement-across-a-reviewer-batch) gives the command and freeze order; this is research scoring, not live-message screening.

The [prospective human validation protocol](docs/HUMAN_VALIDATION_PROTOCOL.md) specifies the outstanding Romanian and Ukrainian taxonomy review, blinded abstract-card annotation, and the limits of any later model claim. It is a study plan, not a completed language validation.

The [independent reviewer handoff](docs/REVIEWER_HANDOFF.md) lists the exact taxonomy digests, reviewer roles, official institutional introduction routes, and packet handling steps. These are candidate routes; no reviewer has been recruited or contacted through the project.

Once a reviewer accepts the scope, `python -m roguard.review_bundle --language ro --output review_runs/ro-taxonomy-review.zip` (or `--language uk`) makes a private, exact-byte taxonomy and review-question bundle that can be shared without repository access. It includes no labeled cards or model scores and does not count as a completed review.

The [independent abstract-batch intake](docs/INDEPENDENT_BATCH_INTAKE.md) accepts a new Romanian or Ukrainian test set from two outside authors only after separate content review and independent taxonomy approval. It binds exact bytes and author declarations, then feeds the existing blinded packet workflow. Intended labels cannot be scored by `roguard-pairs` before adjudication. No independent batch has been received yet.

The [offline author kit](docs/AUTHOR_KIT.md) prepares two balanced private workbooks, assembles a candidate outside Git, and admits it only after a separate content-review declaration. It cannot supply the missing reviewers or certify their judgments.

For a future independent model test, the [prediction-seal protocol](docs/STUDY_PREDICTION_SEAL.md) keeps the approved, author-labeled batch local until the blind model predictions have been committed and pushed. A Git-order check verifies the commit sequence without claiming to prove human blinding or remote push timing.

A narrower [frozen D1/S1 advisory study](BENCHMARK.md#frozen-romanian-d1s1-advisory-transfer-batch-0025) passed its symbolic-card target: 47/48 complete pairs and no false reviews on 48 negatives with unchanged weights and cutoff. Those cards contain pattern and field descriptions, not child messages or realistic responses. The six-category classifier and authentic-language detection remain unvalidated.

A subsequent [preregistered D1/S1 prose transfer](BENCHMARK.md#frozen-romanian-d1s1-prose-transfer-batch-0027) with the same frozen model **failed** at 23/48 complete pairs, with nine false reviews and S1 recall of 8/24. This exposes wording sensitivity even among invented Romanian cards. The research record keeps both results.

A [new prose-specialized adapter](BENCHMARK.md#frozen-romanian-d1s1-prose-repair-batch-0030) improved S1 recall to 24/24 and exact pairs to 33/48 on another sealed symbolic test, versus 22/48 for unchanged v11. It still failed the registered false-review ceiling with 15/48 D1 negatives flagged. It remains a research artifact, not a live detector.

A [prospective D1 cutoff study](BENCHMARK.md#prospective-romanian-d1-cutoff-transfer-batch-0032) selected a stricter threshold on a new development batch, then failed on a separate test surface at 24/48 exact pairs and 24/48 false reviews. Moving the cutoff did not repair the D1 evidence gap.

A [factorized D1 study](BENCHMARK.md#prospective-romanian-factorized-d1-study-batch-0034) trained a new research adapter to score four abstract evidence fields before applying the exact D1 rule. It failed its development gate and the fresh diagnostic test: 32/72 complete pairs versus 46/72 for unchanged V16 on the same cards. The selected adapter and failure are preserved for research, not use on child messages.

A [joint-field D1 study](BENCHMARK.md#prospective-romanian-joint-field-d1-study-batch-0036) also failed its development gate and a further sealed abstract test: 14/48 complete pairs versus 40/48 for frozen V16. Its selected research adapter is preserved, not offered for live screening.

A [class-balanced follow-up](BENCHMARK.md#prospective-class-balanced-romanian-d1-study-batch-0037) improved joint-field development performance but failed on fresh abstract wording: 18/48 complete pairs versus 36/48 for frozen V18. All four models' scores and the failure are preserved. The Romanian D1 detector remains experimental.

A [multilingual NLI adaptation](BENCHMARK.md#prospective-romanian-nli-adapted-d1-study-batch-0038) also failed its frozen Romanian abstract-card test: 9/48 complete pairs with 25/48 false reviews, versus 42/48 pairs and 4/48 false reviews for unchanged V16. All 1,248 saved scores replay exactly from the private selected adapters. This result does not support screening child messages.

A [fixed-V16 transfer audit](BENCHMARK.md#retrospective-fixed-v16-d1-transfer-audit) checks eight previously consumed D1 surfaces together: positive cards outranked their paired negatives in 185/192 pairs, while the unchanged cutoff yielded only 138/192 complete pairs and ranged from 4/24 to 24/24 by surface. It is a diagnostic of invented-card instability, not a new test or a basis for tuning on those cards.

The [V22 development-only adaptation](BENCHMARK.md#v22-balanced-d1-development-candidate-no-new-test) balanced seven consumed Romanian surface families and selected by its weakest of four consumed development surfaces. It still failed: 73/96 complete pairs and 75/96 positive reviews, versus unchanged V16's 75/96 pairs and 85/96 positives. The selected private adapter and all development scores are preserved. No new held-out V22 result exists.

The [V23 group-weighted continuation](BENCHMARK.md#v23-group-weighted-d1-development-candidate-no-new-test) improved the consumed development result to 76/96 pairs, 81/96 positive reviews, and 6/96 false reviews, but its weakest surface reached only 11/24. It failed the unchanged gate and has no new held-out result. All three epoch scores and the selected private adapter are preserved.

The [V24 Romanian BERT candidate](BENCHMARK.md#v24-romanian-bert-d1-development-candidate-no-new-test) tested a language-specific encoder under a frozen recipe. It failed the same development gate at 25/96 pairs, including 0/24 on two consumed surfaces. All six epoch scores and the selected private adapter are preserved. This provides no new held-out or child-language accuracy evidence.

On a separate, preregistered [benign Romanian request probe](BENCHMARK.md#frozen-benign-romanian-request-probe-neutral-0001), the unchanged adapter made 0/48 false D1 reviews. This negative-only result measures neither disclosure recall nor live specificity.

A second, preregistered [near-domain benign probe](BENCHMARK.md#frozen-benign-romanian-near-domain-probe-0002) found 0/36 false D1 reviews for each unchanged V11, V16, and V18 adapter on generic questions containing safety vocabulary. It does not repair their failures on unfamiliar abstract evidence cards or establish live screening accuracy.

The [combined study](BENCHMARK.md#frozen-romanian-mixed-input-workflow-batch-0026) also passed its **mixed-input symbolic** target. Four categories were given generated contract facts that a caller must provide and verify in actual use; the software did not infer those facts from the card text.

RoMistral is the Romanian base model. Its [model card](https://huggingface.co/OpenLLM-Ro/RoMistral-7b-Instruct) lists a noncommercial license. The pinned RoMistral weights were verified, converted locally to 4-bit MLX, and fitted to approved abstract cards. The [Ukrainian Mistral candidate](https://huggingface.co/SherlockAssistant/Mistral-7B-Instruct-Ukrainian) has not been selected or tested. The software's MIT license does not relicense model weights.

## Pornire rapidă (română)

Într-un mediu Python curat, din directorul proiectului:

```sh
python3.12 -m venv .venv312
. .venv312/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
roguard-contrast examples/contrast_route_before.json examples/contrast_route_after.json
```

Consultați [taxonomia română v0.2](taxonomy/taxonomy_ro.md), [baza ei de cercetare](docs/TAXONOMY_RESEARCH.md) și [fișa de a doua lectură](docs/RO_REVIEW_PACKET.md). Rafael a aprobat taxonomia și loturile 0004–0007; aprobările rămân înregistrări istorice. Pentru loturile generate ulterior, utilizatorul a înlocuit revizuirea umană a fiecărui lot cu [validare automată reproductibilă](docs/METHODOLOGY.md). Aceasta verifică generatorul înregistrat, etichetele și fișierele exacte, fără a pretinde că exemplele sintetice reprezintă limbajul real al copiilor. [Taxonomia ucraineană](taxonomy/taxonomy_uk.md) a fost redactată separat și rămâne un proiect care necesită revizuire de limbă și de specialitate.

Studiul pilot a folosit [lotul 0005: antrenare](data/synthetic/batch-0005.preview.md), [lotul 0006: dezvoltare](data/synthetic/batch-0006.preview.md) și [lotul 0007: test](data/synthetic/batch-0007.preview.md). [Lotul 0008](data/synthetic/batch-0008.preview.md) a trecut validarea automată și a fost folosit o singură dată pentru comparația nouă cu modelul de bază. Modificarea ulterioară a fișierelor invalidează atestarea. Niciun test consumat nu poate fundamenta o nouă selecție de model.

După diagnosticul primului test, o a doua versiune a promptului a fost antrenată numai pe loturile aprobate 0004–0006. Pe testul nou, a obținut 23/40 potriviri exacte, față de 25/40 pentru modelul de bază cu același prompt. [Detaliile și limitele](docs/TRAINING_PLAN.md) nu justifică folosirea practică a clasificatorului.

`screen()` întoarce scoruri și coduri experimentale, `calibrate()` stabilește praguri pe setul de dezvoltare pentru sisteme cu scoruri reale, iar `escalate()` întoarce o sugestie pentru om. Ieșirile MLX obișnuite sunt coduri 0/1. Varianta experimentală cu logiturile `da`/`nu` produce un scor condiționat, nu o probabilitate calibrată de risc, și a eșuat pe [testul nou](BENCHMARK.md#frozen-romanian-token-margin-comparison-batch-0013). `assess_contracts()` verifică regulile declarate, iar `assemble_report()` păstrează separat semnalele modelului și constatările deterministe. Nicio funcție nu trimite informații către altă parte. Pragurile de lizibilitate pentru română și ucraineană nu sunt încă stabilite.

`assess_case()` oferă un apel comun pentru fișele structurate și un scor de model opțional: verifică întâi regulile declarate, apoi scorul textului, și întoarce aceleași rezultate separate. În ucraineană funcționează calea structurată; modelul experimental acceptă numai româna. Un rezultat pozitiv din model nu poate crea o permisiune de distribuire.

Taxonomia separă un posibil semnal de sprijin de erorile unui răspuns sau ale unui flux. `R1`, `P1` și `G1` descriu reguli care trebuie verificate pe fișe structurate; pentru `A1`, un plafon de cuvinte declarat se poate verifica mecanic. Un scor de clasificator nu stabilește autoritatea, permisiunea ori dreptul de divulgare. Loturile abstracte testează schema de adnotare și nu demonstrează recunoașterea relatărilor reale.

Pentru fișe structurate complete, `check_readability_contract()`, `check_routing()`, `check_permission()`, `check_boundary()` și `check_gate()` întorc coduri de motiv obținute din regulile declarate. `check_disclosure_evidence()` și `check_support_contract()` verifică doar câmpurile evaluate și declarate de apelant; nu citesc mesajul copilului sau răspunsul propus. Pentru A1, CLI poate număra local cuvintele unui răspuns în română sau ucraineană și respinge un număr declarat greșit; plafonul rămâne furnizat de apelant. Această verificare nu validează un prag de înțelegere. Aceste funcții nu trimit date. De exemplu:

```python
from roguard import RoutingCard, check_routing

card = RoutingCard(
    principal="minor", item="element-simbolic", purpose="A", proposed_recipient="tutore",
    recipient_roles=frozenset({"minor", "tutore"}),
    allowed_by_scope={("minor", "element-simbolic", "A"): frozenset({"minor"})},
)
print(check_routing(card).reason_codes)  # ('PRINCIPAL_SCOPE_CONFLICT',)

from roguard import assess_contracts, assemble_report
report = assemble_report(contracts=assess_contracts(routing=card))
print(report.contracts.findings[0].status)  # review
```

## Quickstart (English)

From the repository root, in a clean Python environment:

```sh
python3.12 -m venv .venv312
. .venv312/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -q
python -m roguard examples/contracts_ro.json
python -m roguard examples/contracts_uk.json
```

These commands install only the lightweight package, run the provenance/API tests, and check one caller-declared contract set per language. The CLI returns JSON and takes no external action. See the [reproduction guide](docs/REPRODUCE.md), [decision architecture](docs/DECISION_ARCHITECTURE.md), [model card](docs/MODEL_CARD.md), [methodology](docs/METHODOLOGY.md), [findings](docs/FINDINGS.md), [novelty statement](docs/NOVELTY.md), and [research roadmap](docs/RESEARCH_ROADMAP.md) for the evidence, limits, and next gates.

The read-only CLI evaluates declared fictional contracts in either language without downloading a model. Its [JSON Schema](schema/contract-input.schema.json) supports editor completion for the input shape:

```sh
python -m roguard examples/routing_ro.json
python -m roguard examples/routing_uk.json
python -m roguard examples/contracts_ro.json
python -m roguard examples/contracts_uk.json
python -m roguard examples/readability_text_ro.json
python -m roguard examples/readability_text_uk.json
```

Each JSON result lists independent findings, reason codes, evidence bases, the categories actually checked, and whether review is suggested. It explicitly marks caller facts and language tags as unverified. The `contracts_*` files exercise all six category codes with caller-declared facts, including D1 pattern fields and S1 response-field judgments. These two checks do not interpret text. The Ukrainian result is marked `draft_uk_v0.1`; its taxonomy has not had an independent language or child-safety review. The [JSON examples](examples/) are symbolic policy cards, not child messages.

For multiple local contracts, run `python -m roguard --jsonl examples/contracts_mixed.jsonl`. It validates every line before printing any results. To evaluate another classifier on an attested abstract pair batch, run `roguard-pairs --batch batch-0013 --predictions eval/runs/ro-v5-margin-test-0013.json`; this reports exact pairs, per-category precision and recall, and false reviews. See the [CLI reference](docs/CLI.md) and [declared-contract audit](docs/CONTRACT_AUDIT.md).

See the [CLI contract reference](docs/CLI.md) for every input field and output status.

The trained-model workflow is deliberately gated. A local Apple Silicon path uses [MLX-LM QLoRA](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md); the 7B base model and adapter are stored under ignored `checkpoints/`:

```sh
# Local model setup:
python -m pip install -e '.[mlx]'
hf download OpenLLM-Ro/RoMistral-7b-Instruct --revision fc097a5fdc2a5a689df084c76ffacf2cfb9f6550 --local-dir checkpoints/romistral-source
python training/convert_base.py --source checkpoints/romistral-source --revision fc097a5fdc2a5a689df084c76ffacf2cfb9f6550 --output checkpoints/romistral-4bit
# The named historical training batches already have exact-byte approvals:
python training/train_mlx.py --batch batch-0004 --batch batch-0005 --batch batch-0006 --model checkpoints/romistral-4bit --base-revision fc097a5fdc2a5a689df084c76ffacf2cfb9f6550 --output checkpoints/ro-v1
python eval/evaluate_mlx.py --batch batch-0007 --adapter checkpoints/ro-v1/adapter --output eval/runs/ro-v1.json
```

The private research archive preserves selected [v9a](models/ro-mmbert-v9a-abstract/README.md), [v10](models/ro-mmbert-v10-abstract/README.md), and [v11](models/ro-mmbert-v11-abstract/README.md) mmBERT research adapters. Their selected weights and portable configs are withheld from this public checkout; published metadata and saved scores document the symbolic results, but a public reader cannot replay weight-dependent checks. All three failed their broader six-category synthetic targets and are research comparators only. The base model's [model card](https://huggingface.co/jhu-clsp/mmBERT-base) lists an MIT license. No trained adapter is offered as a validated detector of real child language.

Selected RoMistral [v4](models/ro-v4-abstract/README.md) and [v6](models/ro-v6-abstract/README.md), plus Qwen [v7](models/ro-qwen-v7-abstract/README.md) and [v8](models/ro-qwen-v8-abstract/README.md), are also preserved in the private research archive. Their frozen failed results and metadata remain here. The [reproduction guide](docs/REPRODUCE.md#private-selected-mlx-research-artifacts) describes the checks available to authorized archive readers.

The historical RoMistral [v1](models/ro-v1-abstract/README.md), [v2](models/ro-v2-abstract/README.md), and [v3](models/ro-v3-abstract/README.md) final-step pilots are preserved privately as well. Their saved failures remain visible here; replay of their weights requires access to the research archive and pinned local base.

The three named pilot batches were human-approved separately. The 200-step MLX run completed on this 16 GB Mac; [its held-out pilot results](BENCHMARK.md) show major category failures. The trainer requires train and dev splits and rejects test rows. The evaluator requires test rows and checks that the adapter was trained under the same taxonomy without the test batch. Missing attestations stop either command before model loading. MLX code outputs yield hard 0/1 decisions, not calibrated probabilities. See [training and evaluation limits](docs/TRAINING_PLAN.md). The package cannot claim live reliability from synthetic evaluation.

The second adapter uses a versioned, card-specific Romanian prompt. Its [fresh test result](BENCHMARK.md#romanian-abstract-card-challenge-batch-0008) was below the prompt-only base model. [Batch 0008](data/synthetic/batch-0008.review.json) was attested by the registered automated validator; later batches are never implicitly accepted. `assess_contracts()` is the recommended entry point when the caller has declared policy, permission, gate, or word-cap facts. Model labels cannot replace those checks.

The later [paired v3](BENCHMARK.md#frozen-romanian-paired-card-comparison-batch-0011) and [binary v4](BENCHMARK.md#frozen-romanian-binary-paired-card-comparison-batch-0012) studies each used a new sealed test and the prior four-model local panel plus RoMistral. The v4 adapter was selected on development pairs before its test was opened, yet it still missed three entire categories. The [v4 protocol](docs/V4_PROTOCOL.md), [development selection](eval/runs/ro-v4-dev-selection.json), and [frozen comparison](eval/runs/ro-v4-test-0012-comparison.json) make this failure inspectable. Do not use the learned labels as a real-child screening or routing decision.

The [v5 protocol](docs/V5_PROTOCOL.md) froze a score-bearing readout of that same adapter and a new 96-card test. Its [frozen comparison](eval/runs/ro-v5-test-0013-comparison.json) found that the development-fitted thresholds generalized poorly: 2/48 complete pairs and zero recall for five categories. This further narrows the learned model's role to research diagnostics. The useful operational interface here is the read-only checker for caller-declared contracts; the [state audit](docs/CONTRACT_AUDIT.md) tests permission transitions independently.

The [v6 compositional study](docs/V6_PROTOCOL.md) reached 9/72 complete pairs; the [Qwen3 transfer study](docs/V7_PROTOCOL.md) failed its literal output contract; and the [balanced study](docs/V8_PROTOCOL.md) reached 13/72 pairs. The [v9a pair-trained encoder study](docs/V9A_AMENDMENT.md) improved to 28/72 pairs but overflagged A1 and S1. The [v10 cross-category study](docs/V10_PROTOCOL.md) reached 45/48 pairs in development but only 28/72 on its test. The [v11 broad-surface study](docs/V11_PROTOCOL.md) reached 47/72 on a new test, below its unchanged v10 comparator's 53/72. Its selected cutoff missed every A1 positive. None of these model results supports use on authentic child language.

## Ukrainian summary

Окремий [український проєкт таксономії](taxonomy/taxonomy_uk.md) уже містить шість категорій; його ще мають перевірити незалежні фахівці з мови та захисту дітей за [пакетом перегляду](docs/UK_REVIEW_PACKET.md). Структуровані правила можна перевірити локально через `python -m roguard examples/routing_uk.json`. Попередня [перевірка базової моделі без навчання](BENCHMARK.md#ukrainian-draft-taxonomy-prompt-only-probe) дала 21/48 точних карток і 0/24 повних пар на умовних описах. Пізніше було навчено експериментальні двомовні моделі на вигаданих метаданих. У [зафіксованому тесті V25](BENCHMARK.md#bilingual-same-origin-metadata-studies-v25v29) українські категорії D1 і S1 досягли 36/48 та 38/48 повних пар і не пройшли заздалегідь визначений поріг. Наступні варіанти не пройшли перевірку на наборі розробки; їхні тестові набори залишилися закритими. Ці числа не оцінюють реальні повідомлення дітей. `screen()` і `escalate()` лише повертають сигнали для людини; вони не надсилають повідомлень і не виконують дій.

The research plan's proposed “first” novelty statement has not been established. See [findings](docs/FINDINGS.md) for current comparison sources and provenance. No package name or weights have been published.

The approved Romanian taxonomy keeps its exact reviewed bytes, including links to a sibling private research workspace. The [findings register](docs/FINDINGS.md) provides a readable summary and stable links to that private repository for authorized readers.
