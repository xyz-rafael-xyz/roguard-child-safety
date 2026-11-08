# Frozen TRAIN-selected scaling evaluation 133

The scales selected in study 128 were frozen at 2.0 for Romanian and 1.5 for
Ukrainian. After independent audit 129 passed, study 133 evaluated those exact
settings on all 8,837 original DEV inputs and all 3,080 larger-bank inputs.
It performs no new scale search, fitting, encoder inference or head replay.

Zero TRAIN errors did not generalize. Romanian DEV binary errors rise from 14
to 18 and Ukrainian errors rise from 2 to 7. Across both languages, scaling
introduces 13 errors on previously correct study 75 predictions and fixes four.
On the larger bank, binary errors change from 185 to 187 in Romanian and from
192 to 188 in Ukrainian. The full 40-case panel remains at four Romanian and
six Ukrainian binary errors. All unary field predictions and their errors stay
unchanged. This candidate fails the no-regression and zero-error requirements.

The report retains every population and group comparison against unchanged
study 75 and original v9. It accounts for 11,917 unique inputs, 35,751 candidate
and reference predictions, and 120 panel responses. The panel inputs are
already included in the larger bank and add no independent examples.

NLL and Brier scores are descriptive, with no calibration claim. This evaluation
does not inherit joint prediction-set coverage or establish native runtime
parity. The exposed engineering data are not independently annotated final
validation. No settings were retuned after these results, reserved TEST remains
closed, and no v10 release qualifies.

Exact evaluator source and path-redacted registrations and reports are included.
Private arrays and row outputs remain represented by their original digests.

A separate saved-output audit independently verifies all 35,751 predictions,
576 group summaries, every comparison and all 120 panel responses. It uses the
independent study 129 transform and outcome implementation, with separate
grouping and panel code. Arrays, decisions and integer counts agree exactly;
only loss and Brier reductions allow a 1e-10 numerical tolerance. The audit
also proves that the 11,917 candidate token hashes are distinct. It uses the
same pinned policy primitive and does not independently regenerate the encoder
or heads. Its exact source, report and registration summary are included.
