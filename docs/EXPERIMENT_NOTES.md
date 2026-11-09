Preserved technical description from the original project. Commands using default paths overwrite outputs; prefer the isolated reproduction workflow in the [README](../README.md). Historical path-based provenance keys have been made portable; numeric research results are unchanged.

# Detailed Experimental Notes

The original simulation experiment models:

source → inlet valve → Tank 1 → transfer pump → Tank 2 → outlet pump → output

Both tanks have a capacity of 1,000 L, start at 600 L, and advance in one-minute intervals. Stage 1 generates normal operation. Stage 2 adds genuine physical equipment faults and truthful measurements of their consequences. Stage 3 trains a supervised Random Forest to identify abnormal timesteps. Stage 4 evaluates that frozen model with sensor-only measurement manipulation. Stage 5 adds one physical sensor-consistency validation layer to the unchanged classifier.

## Run

For Stage 5, use the existing Stage 3 model, dataset and saved Stage 4 attack archive:

~~~bash
python evaluate_defence.py
python -m unittest -v test_attacks test_defence
~~~

No earlier stages need to be regenerated. Stage 5 preserves all Stage 3 and Stage 4 artifacts, verifies their hashes, and writes to **output/stage5/**. To run Stage 4 separately:

~~~bash
python evaluate_attacks.py
python -m unittest -v test_attacks
~~~

The commands below describe building the earlier stages from scratch. They are not required for Stages 4 or 5, which never invoke model training.

Use the project's Python environment and install the dependencies:

~~~bash
python -m pip install -r requirements.txt
python simulate.py
python train_baseline.py
~~~

The simulator defaults to **Stage 2**: 10,080 minutes (seven days), seed 3015, with output in **output/stage2/**. The separate training script reads that CSV and writes the Stage 3 baseline to **output/stage3/**. If the Stage 2 CSV already exists, only the training command is needed. To reproduce the earlier normal-only dataset:

~~~bash
python simulate.py --stage 1
~~~

Optional arguments:

~~~bash
python simulate.py --stage 2 --steps 10080 --seed 3015 --output-dir output/stage2
~~~

The Stage 2 event plan needs at least 6,000 minutes of space; the requested default remains 10,080. A run that cannot complete all planned events and their recovery reports an error rather than exporting an incomplete experiment.

## Files and outputs

- **simulate.py**: normal conditions, hysteresis controller, physical faults, event scheduling, labels, CSV export and printed summaries.
- **plotting.py**: four overview plots and three event close-ups.
- **test_simulate.py**: checks conservation, physical fault effects, labels, event spacing and reproducibility.
- **output/normal_operation.csv**: the existing Stage 1 dataset.
- **output/stage2/water_treatment_stage2.csv**: the complete labelled Stage 2 dataset.
- **output/stage2/events.csv**: planned and actual starts, fault durations, fault ends, and recovery ends.
- **output/stage2/levels.png**, **flows.png**, **pressure.png**, **actuators.png**: week-long plots with coloured fault periods and lighter recovery shading.
- **output/stage2/zoom_transfer_pump_failure.png**, **zoom_inlet_valve_stuck_open.png**, **zoom_outlet_pump_failure.png**: the first example of each type, showing all process variables and commands versus actual states, with about an hour before and after the event plus its complete recovery.

## Process variables and equipment states

Each CSV row represents one complete minute. The timestamp and levels refer to the **end** of that minute; flows and device states apply **during** that minute.

| Column | Meaning |
| --- | --- |
| timestamp | End-of-minute datetime; first row is 2026-01-01 00:01 |
| tank1_level, tank2_level | Actual modelled fill levels, in percent of 1,000 L |
| inlet_flow, transfer_flow, outlet_flow | Actual delivered flows, in L/min; these values enter the water balance |
| pressure | Transfer-line pressure in bar, with small measurement variation |
| inlet_valve, pump1, pump2 | Compatibility columns representing **actual physical states** |
| inlet_valve_command, pump1_command, pump2_command | Controller requests: 0 = closed/off, 1 = open/on |
| inlet_valve_actual, pump1_actual, pump2_actual | Physical valve position or pump operation: 0 = closed/stopped, 1 = open/running |
| label | 0 = normal; 1 = physical fault or its labelled recovery |
| anomaly_type | normal, transfer_pump_failure, inlet_valve_stuck_open, or outlet_pump_failure |
| event_id | Unique physical event number, retained during recovery; 0 for normal rows |
| event_phase | normal, active (fault physically present), or recovery (fault removed, process settling) |

Command and actual values coincide when equipment is healthy. During a failure, the controller still makes ordinary decisions, but the failed actuator may not obey them. An open valve or running pump can have limited flow when there is no available water or free capacity; its state is not a guarantee of nominal flow.

The four label/event columns are experiment ground truth, separate from the truthful process readings.

## Controller and conservation

The original hysteresis rules and controller memory are preserved:

| Device | Starts/opens | Stops/closes |
| --- | --- | --- |
| Inlet valve | Tank 1 ≤45% | Tank 1 ≥75% |
| Transfer pump | Tank 1 ≥50% AND Tank 2 ≤60% | Tank 1 ≤40% OR Tank 2 ≥78% |
| Outlet pump | Tank 2 ≥78% | Tank 2 ≤45% |

Between thresholds, each command retains its previous state. Faults alter physical equipment operation, not these rules.

~~~text
V1_next = V1 + (inlet_flow − transfer_flow) × 1 minute
V2_next = V2 + (transfer_flow − outlet_flow) × 1 minute
level (%) = 100 × volume / 1000
~~~

Real flows are limited **before** updating volumes. Outlet flow cannot exceed Tank 2's available water; transfer cannot exceed Tank 1's water or the space freed in Tank 2; inlet cannot exceed the space freed in Tank 1. Volumes are never reset, clipped back to normal levels, or forced onto a desired trajectory.

At full capacity, the model idealises hydraulic backpressure as blocked/reduced inflow. For example, a stuck-open inlet can fill Tank 1 to 100%; after that, inlet flow only replaces simultaneous transfer outflow. There is no overflow or unrecorded water loss, so the same balance equations remain valid.

## Preserved normal variability

Independent background offsets start at zero. Every 120 minutes, new targets are sampled uniformly from −1 to +1. Each minute:

~~~text
offset = previous_offset + (target − previous_offset) / 30
~~~

This produces gradual bounded drift with a response time of about 30 minutes. Conditions continue evolving while devices are off.

| Flow | Baseline | Background drift | Minute jitter | Additional variation | Active rate bounds |
| --- | ---: | ---: | ---: | --- | ---: |
| Inlet | 28 | ±0.5 | ±0.3 | none | 27.2–28.8 |
| Transfer | 20 | ±0.3 | ±0.2 | none | 19.5–20.5 |
| Outlet | 25 | ±0.3 | ±0.2 | daily sine, amplitude 0.75 | 23.75–26.25 |

All rates are L/min, before physical flow limits or failures. The daily term is 0.75 × sin(2π × elapsed_minutes / 1440). Demand means the requested outlet rate while the pump operates; unmet demand is not accumulated.

Pressure is approximately 2.25 bar when the physical transfer pump operates and 0.35 bar otherwise, plus uniform variation of ±0.04 bar. Flow variation affects actual water movement; levels are calculated from actual volumes without separate level-sensor noise. The Stage 1 process random stream is unchanged; event planning uses an independent stream derived from the same seed.

## Physical abnormal conditions

| Condition | Implementation | Physical consequences |
| --- | --- | --- |
| Transfer pump failure | Force pump1_actual = 0 for the entire event, independent of its command | Transfer flow becomes exactly zero and transfer-line pressure falls. Tank 1 can fill until its inlet command closes; Tank 2 drains until its outlet command stops. |
| Inlet valve stuck open | Force inlet_valve_actual = 1 even with a closed command | Inlet flow continues and Tank 1 rises above its usual pattern, potentially reaching 100%. Capacity can then limit inflow. |
| Outlet pump failure | Force pump2_actual = 0 despite an on command | Outlet flow becomes exactly zero. Continuing transfer initially raises Tank 2, until the unchanged controller stops transfer near its upper threshold. |

These are complete physical failures, with no bypass leakage. Exact zero flow is the idealised version of approximately zero. The original controller can limit a fault's consequences: an outlet failure does not necessarily fill Tank 2 to capacity because transfer stops around 78%. Pressure is a simple indicator of transfer-pump operation, not a detailed hydraulic calculation.

## Events, labels and recovery

There are **five events of each type**, 15 total. Fifteen evenly spread time windows cover the week after a six-hour initial stabilisation period, leaving a four-hour final reserve. Each group of three windows contains one of each type in seeded random order. Within each window, a target start and a duration of **60–160 minutes** are drawn.

The actual start waits for a suitable controller phase. Pump failures begin when both pumps and the inlet are commanded on, ensuring a visible loss of transfer or discharge. A stuck valve begins when the inlet is commanded closed and transfer is commanded on. New events wait for the preceding event and its recovery to finish. Thus physical fault events and their labelled recovery intervals do not overlap.

This phase selection is intentional experimental design, rather than a model of real equipment failure probabilities. In the default seed, durations span **64–154 minutes** and events occur across all seven days.

When an event ends, the equipment immediately resumes obeying its command. Actual volumes and controller memory carry forward normally. Recovery keeps **label = 1**, the originating **anomaly_type**, and the same **event_id**, with **event_phase = recovery**. It ends only after both tanks have stayed within **40–80%** for **15 consecutive minutes**, with at least one device commanded on. The confirmation minutes remain labelled abnormal; the following minute returns to label 0. This conservative rule prevents high residual Tank 1 levels after a stuck valve from being silently marked normal.

The default dataset contains:

| Type | Physical-fault rows | Recovery rows | Total labelled rows | Events |
| --- | ---: | ---: | ---: | ---: |
| transfer_pump_failure | 629 | 75 | 704 | 5 |
| inlet_valve_stuck_open | 631 | 135 | 766 | 5 |
| outlet_pump_failure | 425 | 75 | 500 | 5 |
| normal | 0 | 0 | 8,110 | 0 |

Overall, **8,110 rows (80.46%) are normal** and **1,970 (19.54%) are abnormal including recovery**. No rows are deleted or resampled to achieve these percentages.

The event manifest uses intervals [start, end_exclusive). Because process timestamps mark interval ends, an event starting at 07:19 first appears in the row timestamped 07:20. Recovery begins at the physical event end and ends at recovery_end_exclusive.

## Summaries and verification

Each Stage 2 run prints total rows, normal/abnormal counts and percentages, rows per type split into active/recovery, physical event counts, and full-dataset minimum, maximum, mean and sample standard deviation (ddof=1) for all six continuous variables. Statistics include zero-flow periods. Stage 1 continues printing all nine numeric-variable summaries.

Run the checks with:

~~~bash
python -m unittest -v test_simulate
~~~

The model omits treatment chemistry, detailed pipe dynamics, overflow paths, leakage, pump startup delays and energy use. The rates and faults are illustrative rather than plant-calibrated. The controller's existing all-off idle region at Tank 1 levels of 45–50% is preserved; the default experiment is checked for continued operation and complete recovery. Different seeds, larger variability or altered event settings should be checked again.

## Stage 3: supervised anomaly-detection baseline

The baseline classifies each timestep as normal (0) or abnormal (1), including the Stage 2 recovery labels. It is a binary classifier, not a classifier that identifies which equipment fault occurred. The simulator and its labelled dataset remain unchanged.

Run:

~~~bash
python train_baseline.py
# Optional paths:
python train_baseline.py --data output/stage2/water_treatment_stage2.csv --output-dir output/stage3
~~~

The primary model uses exactly these nine instantaneous features, in this order:

~~~text
tank1_level
tank2_level
inlet_flow
transfer_flow
outlet_flow
pressure
inlet_valve_actual
pump1_actual
pump2_actual
~~~

An explicit allowlist prevents label, anomaly_type, event_id, event_phase and timestamp from entering the model. Legacy state aliases are omitted to avoid duplicate inputs. Command-state columns are used only in one predeclared diagnostic comparison, which adds inlet_valve_command, pump1_command and pump2_command to the same nine features. The primary feature list was chosen before inspecting scores and is not changed based on that comparison. There are no rolling features or other history-based inputs in this baseline.

### Chronological split with complete events

The script chooses the chronological boundary closest to 70% of the rows, excluding every boundary inside a nonzero event_id block. This keeps the **physical fault and its complete recovery together**. Both labels are required in each set. When possible, the boundary also preserves all observed fault types in both sets; otherwise it prefers test-set coverage. Ties choose the earlier boundary. This uses metadata for splitting, never as model features or to optimise a model score.

For the supplied CSV, the exact 70% boundary is valid and needs no adjustment:

| Set | Rows | Normal | Abnormal | Recorded time range | Physical event IDs |
| --- | ---: | ---: | ---: | --- | --- |
| Train | 7,056 | 5,576 (79.02%) | 1,480 (20.98%) | Jan 1 00:01–Jan 5 21:36 | 1–11 |
| Test | 3,024 | 2,534 (83.80%) | 490 (16.20%) | Jan 5 21:37–Jan 8 00:00 | 12–15 |

Dates are in 2026. Source row indices 0–7055 are training; 7056–10079 are testing. The test set has one transfer-pump event, one stuck-valve event and two outlet-pump events. No nonzero event ID appears in both sets. Event ID 0 is simply the normal-row marker and legitimately occurs in both sets. Rows are never shuffled, and test rows occur strictly after training rows.

### Random Forest configuration

The [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) uses:

~~~text
n_estimators = 200
random_state = 3015
class_weight = "balanced"
min_samples_leaf = 2
max_depth = None
max_features = "sqrt"
bootstrap = True
n_jobs = -1
~~~

Other parameters retain their defaults and are recorded in run_metadata.json. Balanced weighting is calculated from training labels only. The minimum leaf size is a modest fixed regularisation choice. Only the training features and labels are passed to fit; no scaling, global preprocessing, hyperparameter search or test-based threshold selection is performed. Predictions use the classifier's native predict method, and the class-1 probability is also exported. The diagnostic command model uses identical settings and the identical split; it does not replace the primary model.

### Held-out baseline results

All detection metrics treat **abnormal (1) as positive**. FPR = FP/(FP+TN); FNR = FN/(FN+TP); precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN). Undefined ratios are represented as NaN, rather than a misleading successful score. Saved metric CSVs contain fractions from 0 to 1; percentages are shown here and in the text summary.

| Metric | Primary test result |
| --- | ---: |
| Accuracy | 97.7183% |
| Precision | 99.5294% |
| Recall | 86.3265% |
| F1-score | 92.4590% |
| False-positive rate | 0.0789% |
| False-negative rate | 13.6735% |

Confusion matrix: rows are true labels, columns are predicted labels, both ordered [normal, abnormal].

~~~text
                 Predicted normal   Predicted abnormal
True normal            2532                  2
True abnormal            67                423
~~~

Thus 67 of 490 abnormal rows are missed, while 2 of 2,534 normal rows cause false alarms. Accuracy alone hides this difference.

| Anomaly type | Test events | Abnormal rows | Recall including recovery | Active recall | Recovery recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| transfer_pump_failure | 1 | 147 | 99.32% | 100.00% | 93.33% |
| inlet_valve_stuck_open | 1 | 176 | 93.18% | 98.60% | 69.70% |
| outlet_pump_failure | 2 | 167 | 67.66% | 81.02% | 6.67% |

Overall active-fault recall is **93.20% (384/412)**, while recovery recall is **50.00% (39/78)**. Per-anomaly CSVs also contain precision, F1, FPR, FNR and confusion counts. Each type is compared against the same 2,534 normal test rows; other fault types are excluded, rather than incorrectly relabelled normal. These are binary detection results grouped by the ground-truth fault type.

### Leakage and high-performance investigation

The audit verifies that the fitted feature names contain no label/event/time inputs, that training and test event IDs are disjoint including recovery, and that fitting uses training rows only. It flags overall accuracy or recall at or above 99%, and separately records near-perfect anomaly-group recall. The primary overall score is not perfect, although transfer-pump recall is close to 100%.

The diagnostic model with commands achieves 98.68% accuracy, 91.84% recall, **100% active-fault recall**, and 48.72% recovery recall. All 412 active-fault test rows have at least one command/actual mismatch; none of the normal or recovery test rows do. Commands therefore make active actuator failures particularly easy to identify, rather than representing hidden label leakage.

The simulation itself also has clean signatures: complete pump stoppages produce exact zero flows; pressure has narrow operating/stopped bands; stuck-valve events can sustain 100% Tank 1 levels. These are deliberately simple synthetic faults. Conversely, the 15 healthy confirmation minutes retained in recovery labels overlap ordinary process readings, and an instantaneous classifier has no event history to resolve that ambiguity. The weaker outlet/recovery results are reported directly. Neither labels nor inputs are altered to lower or inflate the score.

The training accuracy is 99.89% and training recall is 100%, versus the lower held-out scores above. Only four physical events are held out, with one example of two fault types, so the per-type estimates have limited support. Event separation removes within-event leakage; temporal dependence and the shared simulator rules remain. This is a baseline for one synthetic week, not evidence of performance on a real plant or robustness to future sensor manipulation.

### Feature importance and saved artefacts

The plot uses the forest's training-derived impurity importance. The highest scores are Tank 1 level (47.57%), outlet flow (13.24%), Tank 2 level (10.22%) and inlet flow (10.07%). The [official feature-importance example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) explains this measure and its limitations. Correlated features can share importance, and continuous inputs offer more possible splits than binary states. Importance describes this fitted model's use of inputs; it does not establish causation or prove an effective adversarial target.

All Stage 3 artefacts are in **output/stage3/**:

| File | Contents |
| --- | --- |
| test_predictions.csv | Original source row, timestamp, nine inputs, true/predicted labels, class-1 probability and event metadata for tracing errors; metadata is never used as input |
| overall_metrics.csv, summary.txt | Overall scores and the complete printed report |
| confusion_matrix.png | Held-out confusion matrix with explicit class labels |
| feature_importance.png, feature_importance.csv | Primary model feature ranking |
| per_anomaly_metrics.csv | Binary performance grouped by fault type |
| phase_metrics.csv | Active and recovery support, true positives, misses and recall |
| split_summary.csv | Row counts, label distribution and time ranges |
| command_comparison.csv | Primary model versus the single command-state diagnostic |
| command_mismatch_by_phase.csv | Counts underlying the mismatch investigation |
| sanity_checks.txt | Leakage checks, training scores and interpretation of high scores |
| random_forest.joblib | Saved **primary** classifier with its feature names |
| run_metadata.json | Feature list, full parameters, event IDs, split boundary, input SHA-256 and software versions |

The recorded environment uses Python 3.12.3 and scikit-learn 1.9.1. A fixed model seed and fixed chronological split reproduce predictions in this environment; the tests allow tiny floating-point differences from parallel probability accumulation.

Run all simulator and baseline checks with:

~~~bash
python -m unittest -v test_simulate test_baseline
~~~

## Stage 4: adversarial sensor-manipulation evaluation

Stage 4 loads the saved primary nine-feature Random Forest. It never calls fit, changes model parameters, or changes the classifier's native predict decision rule. It restores the exact Stage 3 held-out rows from test_predictions.csv and run_metadata.json, rather than making a new split. Baseline predictions must match Stage 3 before any experiment begins. The command-augmented diagnostic classifier is not the target model.

Run only the evaluation:

~~~bash
python evaluate_attacks.py
# Optional paths:
python evaluate_attacks.py --data output/stage2/water_treatment_stage2.csv --stage3-dir output/stage3 --output-dir output/stage4
~~~

The model SHA-256 before and after evaluation is:

~~~text
a5da86a6f122174716c96fa95f3ac9b4900c7347865d49a5175d3cc8092194bb
~~~

Hashes of the Stage 2 source CSV and every Stage 3 output are also checked unchanged and saved in the Stage 4 run metadata. Output is kept outside the Stage 3 directory. The physical simulator is not rerun, and the original source data is never overwritten.

### Threat model and reference construction

The attacker can alter only six reported continuous values: tank1_level, tank2_level, inlet_flow, transfer_flow, outlet_flow and pressure. Actual actuator states, commands, labels, event information and timestamps remain truthful. All 2,534 already-normal test rows are untouched. Abnormal rows include both active faults and their labelled recovery.

This is an **oracle-targeted evaluation**: ground truth selects the abnormal rows to attack, and experiment C additionally uses the fault type to choose its prescribed sensor group. This models an attacker with that knowledge; it is not an online method for discovering abnormal rows. Ground truth is never changed or passed into the Random Forest.

References are unweighted medians from **only the 5,576 normal rows in the saved Stage 3 training portion** (source rows before 7056). No held-out normal values contribute. The reference function receives only the training frame. Values shown below are rounded for display; computations use full precision.

| Sensor | Normal-training median | Fixed allowed range |
| --- | ---: | --- |
| tank1_level | 58.739676% | 0–100% |
| tank2_level | 55.363062% | 0–100% |
| inlet_flow | 27.571473 L/min | 0–28.8 L/min |
| transfer_flow | 19.940678 L/min | 0–20.5 L/min |
| outlet_flow | 24.512295 L/min | 0–26.25 L/min |
| pressure | 2.242834 bar | 0–3 bar |

For selected sensors on targeted abnormal rows:

~~~text
x_adv = x_original + alpha × (normal_training_median − x_original)
alpha ∈ {0.00, 0.25, 0.50, 0.75, 1.00}
~~~

Alpha 0 preserves the original exactly. Alpha 1 assigns the exact median, avoiding floating-point cancellation at the endpoint. Generated values are clipped to the fixed per-sensor engineering bounds above, derived from the simulator design rather than the test distribution. These bounds constrain individual readings; a fabricated set of measurements can still contradict water balance or truthful actuator states. The underlying physical volumes and flows remain those of Stage 2.

### Experiment grid

There are **13 configurations × 5 strengths = 65 runs**:

- **A:** six separate single-sensor attacks, each targeting all 490 abnormal test rows.
- **B:** groups of the top 1, 2, 3 and all 6 continuous sensors. The frozen Stage 3 importance ranking is tank1_level, outlet_flow, tank2_level, inlet_flow, pressure, transfer_flow. This is a practical selection heuristic, not a causal claim or an attack-optimised ranking. These groups also target all abnormal rows.
- **C:** transfer failure uses transfer_flow, pressure, tank1_level and tank2_level; stuck valve uses tank1_level and inlet_flow; outlet failure uses tank2_level and outlet_flow. Only rows of the corresponding fault type are targeted; all other rows remain unchanged.

No group, strength, threshold or model parameter is selected by retraining. The complete prescribed grid is reported, including attacks that have little effect or make some missed rows detectable.

### Evaluation and attack success

Recall, false-negative rate, precision, F1 and confusion counts are saved for 12 scopes: all abnormal rows; all active rows; all recovery rows; each of the three fault types; and active/recovery separately within each type. Each scope is compared with the same untouched normal test rows for precision and F1. Other abnormal scopes are excluded rather than relabelled normal. This gives **780 aggregate result rows**.

Attack success rate (ASR) is:

~~~text
baseline true positives changed to normal by this attack
------------------------------------------------------
baseline true positives targeted by this attack
~~~

The baseline has **423 true positives and 67 already-missed abnormal rows**. Those 67 misses are never credited as successful evasion. For fault-specific experiments, the denominator includes only baseline true positives of that targeted fault (146 transfer, 164 valve, 113 outlet), not all 423. Alpha 0 uses the same target cohort as a no-manipulation control and has ASR 0. Untargeted scopes have an undefined ASR, recorded as blank/NaN.

The tables also count previously missed rows that become detected. Consequently, a loss of 11 baseline true positives can coexist with a net loss of only 10 detections when one existing miss is recovered. ASR counts successful flips directly and is distinct from overall FNR. Normal predictions remain TN=2,532 and FP=2 in every run because normal inputs are unchanged.

### Stage 4 results

At alpha 1, single-sensor attacks have these results across all abnormal test rows:

| Sensor | Anomaly recall | ASR | Baseline detections evaded |
| --- | ---: | ---: | ---: |
| Tank 1 level | 52.65% | 39.24% | 166/423 |
| Tank 2 level | 84.08% | 2.60% | 11/423 |
| Inlet flow | 78.37% | 9.22% | 39/423 |
| Transfer flow | 85.31% | 1.42% | 6/423 |
| Outlet flow | 85.31% | 2.60% | 11/423 |
| Pressure | 83.47% | 3.78% | 16/423 |

**Tank 1 level is the most influential individual sensor in this grid.** Its largest recall degradation already occurs at alpha 0.75, reducing baseline recall from 86.33% to 52.65%, and plateaus at alpha 1.

Ranked multi-sensor results at alpha 1:

| Sensors | Recall | ASR | Baseline detections evaded |
| --- | ---: | ---: | ---: |
| Top 1 | 52.65% | 39.24% | 166/423 |
| Top 2 | 47.76% | 45.39% | 192/423 |
| Top 3 | 46.94% | 45.86% | 194/423 |
| All 6 | 0.00% | 100.00% | 423/423 |

The six-sensor endpoint conceals every abnormal row, including all previously detected rows. Adding sensors is more effective at alpha 1, but not at every intermediate strength: at alpha 0.5, top-2 ASR is 11.35%, versus 13.24% for top-1. Selection by feature importance is therefore only a heuristic.

Across the five tested strengths, **ASR is nondecreasing for all 13 configurations**, with several plateaus. Recall itself is not always monotonic. For example, an outlet-flow attack at alpha 0.25 makes three baseline misses detectable while evading no true positives, increasing recall slightly to 86.94%. Random Forest decisions and combinations of reported readings do not guarantee a linear response to attack strength.

Fault-specific results at alpha 1, including each fault's recovery rows:

| Fault-specific group | Baseline recall | Attacked recall | ASR | Baseline detections evaded |
| --- | ---: | ---: | ---: | ---: |
| Transfer pump failure | 99.32% | 92.52% | 7.53% | 11/146 |
| Inlet valve stuck open | 93.18% | 0.00% | 100.00% | 164/164 |
| Outlet pump failure | 67.66% | 56.29% | 16.81% | 19/113 |

The **stuck inlet valve is easiest to conceal using the prescribed fault-specific groups**, judged by ASR rather than by a low baseline detection rate. Its group manipulates only Tank 1 level and inlet flow.

### Truthful actuator states and limitations

At all-six/alpha 1, every abnormal row has the same six reported continuous values, while the three actual actuator states remain unchanged. The archive contains all eight observed binary state combinations. The frozen classifier predicts normal for every combination at this median vector, so truthful actuator states **do not prevent this maximal interpolation attack**. This does not show that actuator states are never useful for other inputs or attacks. No command-state model or extra detection rule is added.

An unconditional median is a deliberately simple target. It combines multiple on/off modes, may not form a jointly observed normal state, and does not enforce temporal smoothness or joint hydraulic consistency. It can produce sensor/actuator contradictions. The bounded interpolation is not a minimal-distortion or optimised adversarial attack, and equal alpha values do not imply equal physical perturbations across sensors. Oracle row/type selection also grants more knowledge than some real attackers would have. Results concern one synthetic week and four held-out events, not a real plant.

### Stage 4 artefacts and examples

All output is in **output/stage4/**:

| File | Contents |
| --- | --- |
| adversarial_results.csv | All 780 configuration/strength/scope metric rows, including new false negatives, ASR denominators and newly detected baseline misses |
| attack_success_summary.csv | 65 primary summaries: all-abnormal scope for A/B, targeted-fault scope for C |
| per_sensor_attacks.csv | 30 individual-sensor summaries |
| multi_sensor_attacks.csv | 20 ranked-group summaries |
| anomaly_specific_attacks.csv | 15 fault-specific summaries |
| normal_training_references.csv | Training-only medians, supporting quantiles, sample counts and fixed bounds |
| test_set_truthful.csv | Complete original 3,024-row test set and source row indices |
| adversarial_rows.csv.gz | All 196,560 per-row results (65 × 3,024), in original test ordering within each run |
| row_level_examples.csv | One concrete targeted example per fault, with metadata, actual states and both predictions |
| row_level_example_values.csv | Six truthful/manipulated sensor comparisons per example, with change flags |
| actuator_state_diagnostic.csv | All-six/alpha 1 predictions grouped by unchanged actual states |
| individual_sensor_recall.png | Individual sensor recall versus strength |
| multi_sensor_recall.png | Recall versus strength for progressively larger ranked groups |
| attack_success_rate.png | ASR curves for all three experiment categories |
| per_anomaly_targeted_recall.png | Truthful versus alpha-1 fault-specific recall, split into all/active/recovery panels |
| summary.txt, run_metadata.json | Full report, interpretation, provenance, bounds, configuration definitions and frozen-file hashes |

Original columns in the row archive remain truthful. Only separate sensor columns ending in _adv contain manipulated values. Actual states are stored once, unchanged; those states plus the six _adv readings reconstruct the model input. Each record includes selected_sensors, changed_sensors, attack_eligible, baseline_prediction and adversarial_prediction. Pandas reads the compressed CSV directly with read_csv; gzip uses a fixed timestamp for deterministic output.

Examples are retrospective illustrations, not attack tuning. For the stuck-valve example, the first successful full-tank active row is selected: the physical Tank 1 level is 100%, the reported level becomes 58.739676%, and the prediction changes from abnormal to normal. The other two examples use the first successful active-fault evasion at alpha 1. Full sensor values and unchanged actual equipment states are in the example CSVs.

Stage 4 checks are run separately so earlier baseline tests do not train additional forests:

~~~bash
python -m unittest -v test_attacks
~~~

These nine checks explicitly prohibit calls to RandomForestClassifier.fit and cover exact alpha-0 controls, interpolation endpoints, training-only references, all 65 configurations' protected fields and bounds, fault-specific eligibility, correct ASR denominators, reproducibility and unchanged frozen files. No defensive mechanism is implemented.

## Stage 5: physical sensor-consistency validation

Stage 5 adds **one consistency layer**, containing six physical checks, to the original frozen 200-tree Random Forest. It does not train another classifier, retrain the forest, change its native decision threshold, rerun the simulator, or regenerate Stage 4 attacks. The saved Stage 4 archive supplies all 65 attack runs exactly as exported.

The implementation is separated into:

- **physical_consistency.py:** calibration, six rules, diagnostic residuals and the OR combination. This engine accepts exactly the nine process/actual-state features and rejects labels, event metadata, timestamps and command columns.
- **evaluate_defence.py:** restores the frozen assets, selects normal training rows, evaluates truthful data first, applies the same rules to every saved attack sequence, and exports comparisons and provenance.
- **defence_plotting.py:** the five requested comparison figures.
- **test_defence.py:** 13 focused checks, with RandomForestClassifier.fit prohibited.

Run with the project's environment:

~~~bash
python evaluate_defence.py
# Optional paths:
python evaluate_defence.py --data output/stage2/water_treatment_stage2.csv --stage3-dir output/stage3 --stage4-dir output/stage4 --output-dir output/stage5
~~~

### Calibration and every threshold

Calibration uses only the **5,576 normal Stage 3 training rows**, before source row 7056. Training labels select these rows outside the rule engine; no label or event information enters a rule. The quantile is fixed at **0.99** before held-out evaluation, and is not adjusted after observing test results.

For each flow/pressure check and actual actuator state, learn the normal-training median and the 99th percentile of absolute deviations from that median:

~~~text
center[state] = median(normal training measurements with this actual state)
residual = reported measurement − center[actual state]
tolerance[state] = quantile_0.99(abs(normal training residual))
flag = abs(residual) > effective tolerance
~~~

The four state checks are inlet flow versus inlet_valve_actual, transfer flow versus pump1_actual, pressure versus pump1_actual, and outlet flow versus pump2_actual. Thus an off pump paired with on-range flow or pressure triggers a violation, even if the fabricated measurement looks individually normal.

The two water-balance checks use the current minute's reported flows and consecutive reported levels:

~~~text
expected_delta_tank1 = 100 / 1000 × 1 × (inlet_flow[t] − transfer_flow[t])
expected_delta_tank2 = 100 / 1000 × 1 × (transfer_flow[t] − outlet_flow[t])
residual_tank_i = level_i[t] − level_i[t−1] − expected_delta_tank_i
~~~

Levels are percentages, flows are L/min, capacity is 1,000 L, and the timestep is one minute. The factor is therefore **0.1**, with residuals in percentage points per one-minute observation. Calibration uses **5,564 pairs whose two adjacent original training rows are both normal**. Filtering normal rows does not join across excluded fault intervals.

The simulator conserves water to float64 precision, with learned absolute-balance quantiles around 1.8e−14 percentage points. Each effective tolerance is the larger of its learned quantile and **eight float64 representable increments at the feature scale**. For balance, the scale is the known 100% maximum level, giving **1.1368683772161603e−13**. This is a round-off allowance, not a hand-tuned sensor noise threshold. State-flow checks with exactly zero normal flow get a negligible numerical tolerance of 1.7763568394002505e−15 L/min.

The complete threshold table below is rounded for reading; **defence_thresholds.csv** retains every value at full exported precision, including the raw quantiles, numerical floors, centers and sample counts.

| Rule and state | Normal samples | Center | Effective absolute tolerance | Allowed measurement range |
| --- | ---: | ---: | ---: | --- |
| Inlet flow, valve off | 2,189 | 0 | 1.776357e−15 | approximately 0 L/min |
| Inlet flow, valve on | 3,387 | 27.905010 | 0.729786 | 27.175223–28.634796 L/min |
| Transfer flow, pump off | 871 | 0 | 1.776357e−15 | approximately 0 L/min |
| Transfer flow, pump on | 4,705 | 19.982867 | 0.406453 | 19.576414–20.389320 L/min |
| Transfer pressure, pump off | 871 | 0.348861 | 0.040351 | 0.308511–0.389212 bar |
| Transfer pressure, pump on | 4,705 | 2.250355 | 0.039549 | 2.210806–2.289904 bar |
| Outlet flow, pump off | 1,782 | 0 | 1.776357e−15 | approximately 0 L/min |
| Outlet flow, pump on | 3,794 | 25.006491 | 1.011132 | 23.995360–26.017623 L/min |
| Tank 1 balance | 5,564 pairs | 0 | 1.136868e−13 | absolute residual at most this tolerance |
| Tank 2 balance | 5,564 pairs | 0 | 1.136868e−13 | absolute residual at most this tolerance |

### Causal application and hybrid prediction

The layer checks **all normal and abnormal test rows**, independently of their labels. It records a flag for each rule, signed residuals, rule count, triggered rule names and consistency_alert. The alert is 1 when at least one rule fails.

~~~text
defended_prediction = Random_Forest_prediction OR consistency_alert
~~~

The RF prediction remains separately available. A consistency alert can only add detections; it cannot clear an RF alert.

Every attack is evaluated as its own complete 3,024-row chronological sequence. The first test sample has no preceding test reading, so its state checks run while its two balance checks are unavailable. Later balance checks always use the **previous reported level in that same run**, including manipulated values. They never substitute a hidden truthful level or look ahead. A gap in source-row indices resets balance availability. No history is carried from one attack configuration into the next.

An attack ending can create a balance alert on a currently normal row because the preceding reading was manipulated. Such rows are counted as false positives under the existing physical ground truth; their labels are not changed.

### Truthful test results and false-positive cost

The held-out set remains 3,024 rows: 2,534 normal and 490 abnormal. The truthful comparison is performed before attacks:

| Metric | Original RF | RF + consistency |
| --- | ---: | ---: |
| Accuracy | 97.72% | 93.75% |
| Precision | 99.53% | 77.61% |
| Recall | 86.33% | 86.33% |
| F1 | 92.46% | 81.74% |
| False-positive rate | 0.079% | 4.815% |
| False-negative rate | 13.67% | 13.67% |
| TN / FP / FN / TP | 2,532 / 2 / 67 / 423 | 2,412 / 122 / 67 / 423 |

Confusion matrices, with true classes on rows, are **[[2532, 2], [67, 423]]** and **[[2412, 122], [67, 423]]**.

The layer newly flags **120 normal rows**. Its truthful normal rule counts are inlet flow 10, transfer flow 25, pressure 40 and outlet flow 48; three normal rows trigger multiple rules. Neither water-balance rule flags a truthful normal row. The state checks' 99% conditional intervals are OR-combined, so they do **not** promise a 1% combined false-positive rate. The extra alerts significantly reduce truthful precision and accuracy; robustness comes with this measured cost, and the thresholds are not tuned on test labels to hide it.

### Attack results

Stage 5 preserves Stage 4's groups and strengths: six individual sensors, ranked top-1/top-2/top-3/all-six groups, and the three prescribed fault-specific groups, each at alpha 0, 0.25, 0.5, 0.75 and 1. All 13 alpha-zero controls match truthful defence evaluation.

For every configuration, the full results include recall, FNR, precision, F1, accuracy, FPR, confusion counts and ASR before and after defence, in the same 12 scopes as Stage 4: all abnormal, active, recovery, each fault type, and each type's active/recovery rows. This produces **780 comparison rows**. Every scoped evaluation includes the same normal test rows; other positive scopes are excluded rather than relabelled.

Both ASRs use the **original Stage 3 RF true positives targeted by that attack** as denominator. The undefended numerator is those rows predicted normal by the attacked RF; the defended numerator is those still predicted normal by the RF plus consistency layer. The denominator is 423 for A/B and 146, 164 or 113 for the corresponding C fault. Baseline misses never count as successful evasion, even when the layer newly detects them. Untargeted scopes have undefined ASR rather than a misleading zero.

At alpha 1:

| Attack group | RF recall | Defended recall | RF ASR | Defended ASR |
| --- | ---: | ---: | ---: | ---: |
| Tank 1 level only / ranked top 1 | 52.65% | 95.10% | 39.24% | 2.36% |
| Tank 2 level only | 84.08% | 100% | 2.60% | 0% |
| Inlet flow only | 78.37% | 100% | 9.22% | 0% |
| Transfer flow only | 85.31% | 100% | 1.42% | 0% |
| Outlet flow only | 85.31% | 100% | 2.60% | 0% |
| Pressure only | 83.47% | 89.18% | 3.78% | 0.24% |
| Ranked top 2 | 47.76% | 100% | 45.39% | 0% |
| Ranked top 3 | 46.94% | 100% | 45.86% | 0% |
| All six | 0% | 100% | 100% | 0% |
| Transfer-failure group, transfer rows | 92.52% | 100% | 7.53% | 0% |
| Stuck-valve group, valve rows | 0% | 100% | 100% | 0% |
| Outlet-failure group, outlet rows | 56.29% | 100% | 16.81% | 0% |

For all-six attacks, defended recall is **100% at every nonzero tested strength**, while alpha 0 retains truthful recall of 86.33%. At alpha 1 the layer recovers all **423 evaded baseline detections**, and also detects the 67 baseline misses, giving 490 true positives. There are **126 false positives (4.972% FPR)**: the 122 truthful hybrid false positives plus four normal rows affected by attack-ending history. The defence is checking consistency rather than simply predicting every row abnormal.

Some evasion remains across the grid for Tank 1-only/top-1 and pressure-only attacks. At alpha 1, Tank 1 attacks retain **10/423 successful evasions**, all in recovery (eight valve and two outlet recovery rows). There are also 14 pre-existing RF misses left, giving 24 total false negatives. Constant reported level offsets can pass balance checks while a tank's level is not changing. Pressure-only attacks retain **1/423 successful evasions**; an on-range pressure value can agree with the actual pump state and does not affect water balance. The other pressure-only false negatives mostly predate the attack, explaining why recall is 89.18% despite a low evasion ASR.

### Rule effectiveness and a concrete example

Water balance is the most useful protection against coordinated manipulation. At all-six/alpha 1, **each balance rule independently catches all 423 evaded baseline detections**; all 490 targeted abnormal rows trigger multiple rules. Tank 1 balance saves all 164 valve-group baseline detections; Tank 2 balance saves all 19 outlet-group evaded detections. Transfer-failure attacks create pump-off/high-flow and pump-off/high-pressure contradictions as well as balance violations; four rules tie at saving its 11 evaded detections.

For Tank 1-only attacks, the inlet-flow state rule saves 111 evaded detections, the largest individual contribution. Those flow violations can already exist during truthful physical faults; a consistency alert does not prove a reading was manipulated. **rule_trigger_summary.csv** distinguishes new rule alerts introduced by manipulation from alerts already present on truthful inputs, as well as exclusive catches and rows triggering multiple rules. Per-rule counts overlap and must not be added as unique detections.

The saved stuck-valve example is source row **9309**, timestamp **2026-01-07 11:10**, alpha 1:

- Truthful Tank 1 level: **100%**; reported adversarial level: **58.739676%**.
- Truthful inlet flow: **27.718572 L/min**; reported adversarial inlet flow: **27.571473 L/min**.
- Transfer flow remains **20.022846 L/min**; actual actuator states and the other sensors stay unchanged.
- The preceding reported Tank 1 level is also 58.739676%, so observed change is zero. The reported flows imply a **+0.754863 percentage-point** change.
- Tank 1 balance residual is **−0.754863**, exceeding its **1.136868e−13** tolerance. All four state checks pass; **tank1_balance** alone detects this example.
- Truthful RF prediction is **1**, attacked RF prediction **0**, consistency_alert **1**, and defended_prediction **1**.

The example illustrates why individually plausible readings can fail a cross-sensor check. Full values, residuals, rule flags and thresholds are saved in the two example files.

### Limitations

The strong balance result is specific to this simple simulator: flows and levels conserve water almost exactly, with no independent tank-level measurement noise, delay or uncertain hydraulic parameters. The numerical balance tolerance is **not** evidence of realistic sensor precision. Real plants need calibration on representative noisy normal data and correct timing; the current numbers should not be transferred to a real system.

These six checks test relationships, not authenticity. Coordinated measurements that satisfy actuator ranges and balance, or stable level offsets during idle/balanced periods, can evade them. Independent state ranges do not describe every legitimate operating mode. Static per-rule percentiles create a measurable combined false-positive cost. True physical faults can themselves violate normal ranges, so a rule cannot distinguish an equipment fault from sensor manipulation. Trusted actual states and the Stage 4 oracle attack-selection assumptions remain part of this experiment. One synthetic week and four held-out events limit generalisation. No second defence or additional ML model is introduced.

### Stage 5 outputs and verification

All output is in **output/stage5/**:

| File | Contents |
| --- | --- |
| defence_thresholds.csv | All ten state/balance thresholds, centers, raw quantiles, numerical floors, units and training sample counts |
| truthful_defence_metrics.csv | Truthful RF versus hybrid metrics in all 12 scopes |
| defended_attack_results.csv | All 780 configuration/strength/scope comparisons |
| defended_attack_summary.csv | 65 primary summaries: all-abnormal for A/B and targeted fault for C |
| rule_trigger_summary.csv | Truthful and attacked per-rule, any-rule and multiple-rule counts in every scope, including new versus pre-existing alerts |
| rule_effectiveness.csv | Leading rule(s), saved baseline detections and multiple-rule counts for each run |
| truthful_defended_predictions.csv | Every truthful test row, unchanged RF prediction, residuals, flags and hybrid prediction |
| defended_rows.csv.gz | All 196,560 Stage 4 row records, unchanged, with added rule diagnostics and defended predictions |
| defence_examples.csv, defence_example_rules.csv | Four rescued examples and each rule's residual, threshold and flag |
| truthful_defence_performance.png | Truthful metrics and error-rate comparison, showing the false-positive cost |
| recall_vs_attack_strength.png | RF versus defended recall for all 13 configurations |
| attack_success_vs_strength.png | RF versus defended ASR for all 13 configurations |
| all_six_sensor_recall.png | All-six attack recall across the five strengths |
| fault_specific_defence.png | Fault-specific comparisons for all, active and recovery rows |
| summary.txt, run_metadata.json | Report, assumptions, calibration details and unchanged source/model/Stage 3/Stage 4 hashes |

The defended row archive retains every original Stage 4 field. Original sensor columns remain truthful; _adv columns retain Stage 4 manipulated measurements; baseline_prediction is the truthful RF result; adversarial_prediction is the unchanged attacked RF result; defended_prediction is the hybrid result. Original source-row ordering is preserved within every run. Gzip timestamps are fixed for reproducibility.

Run the Stage 4 and Stage 5 checks without invoking the earlier training tests:

~~~bash
python -m unittest -v test_attacks test_defence
~~~

The **22 checks** prohibit RF fitting and cover normal-only calibration, consecutive-pair selection, metadata rejection, actuator relationships, balance units, sample gaps, reported history, causality, hybrid OR behaviour, correct ASR denominators, alpha-zero controls, reproducibility, and unchanged frozen files. Saved attacked RF metrics must also reproduce every Stage 4 metric before Stage 5 exports its results.
