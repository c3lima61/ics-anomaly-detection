# Local public release review

Review date: 2026-11-10. Proposed repository: **ics-anomaly-detection**.

**Publication is not cleared yet.** No remote was connected, no commit was made and nothing was uploaded or published. All work stayed within this project. The directory's `.git` is an empty environment placeholder, not a usable repository; Git reported no repository. Consequently there is no local history, index or staged content to scan. A disposable bare repository under ignored `.validation/` was used only to enumerate files with Git's ignore rules; it is not project history.

## Findings

| Classification | Finding and evidence |
| --- | --- |
| PASS | Original research checkpoint: `.release-backup/original-project.tar.gz` contains the original source, README, requirements and complete output tree. It excludes the environment, IDE files and environment-owned hidden directories. Keep this private: it contains historical machine paths. No original numerical results were removed. |
| PASS | Original 39 tests passed before edits, and all 41 tests (including two portability checks) passed after edits. Logs are under `.validation/`. |
| PASS | Full Stages 1–5 regeneration in `.validation/reproduction/` completed. All 32 CSV tables/archives match retained results with atol=1e-12, rtol=0, including row-level attacks, thresholds and defended decisions. Figures were regenerated successfully; pixel identity was not required. |
| PASS | All five direct dependencies are used. Existing Python 3.12.3 environment passes `pip check`. Exact observed versions recorded in requirements and the environment snapshot. No dependency upgrades or methodology changes were made. |
| PASS | Source review found no project network connections, shell execution, dynamic eval/exec or external downloads. The reproduction helper invokes fixed local script names via argument lists, without a shell. Simulation input validation and existing scientific tests are retained. |
| PASS | Plaintext source/data/docs and decompressed CSV archives were screened for credential assignments, common token/private-key formats, emails, student/author metadata and machine paths. No credential values or student identifiers were detected in intended public text. This is a heuristic screen, not a guarantee. |
| PASS | Absolute machine paths were found in Stage 4/5 JSON provenance map keys. Those keys now use stable data/stage3/stage4 roles. Original SHA-256 values are preserved except Stage 5's digest of the deliberately sanitised Stage 4 metadata, which was recomputed. Future runs emit portable keys. Numeric research outputs remain unchanged. |
| PASS | Candidate tracking inventory is documented in `PUBLIC_FILES.tsv`; useful results (~48 MB original tree) including 11.6 MB and 23.9 MB compressed archives remain included. Environments, IDE configuration, caches, local checkpoints, validation runs and credential/config directories are ignored. |
| REVIEW REQUIRED | `joblib.load` is necessary for existing frozen-model evaluation but is pickle-based and can execute code. Only trusted local models should be loaded. The retained model is byte-for-byte identical to the newly trained model in the verified local environment. This is stronger provenance evidence than prediction agreement, though it is not a standalone serialization security audit. Review its provenance before distribution; regenerate locally if uncertain. See [official persistence guidance](https://scikit-learn.org/stable/model_persistence.html). |
| BLOCKER | Confirm that the university permits public release of this assessment solution and generated artefacts, including any embargo or academic-integrity restrictions. No assignment brief, licence or permission record was supplied. Coursework origin is acknowledged; a polished README does not establish publication rights. |
| REVIEW REQUIRED | Confirm sole authorship or obtain collaborator permission; identify any lecturer-provided starter code, copied algorithms/text, or other third-party material and required attribution. No obvious assignment handout or proprietary plant dataset was present. Synthetic data is generated locally by the simulator, but source ownership remains unverified. |
| BLOCKER | Clean installation is unverified: a separate empty venv was created, but pip returned no available distributions from the configured source (even for unpinned numpy). Retry the pinned full snapshot on a connected machine. Do not assume all recorded versions are available on every platform. |
| BLOCKER | A current complete dependency vulnerability audit could not be performed. Neither pip-audit nor gitleaks/ruff was installed locally; installing pip-audit/ruff failed at the package source. A direct OSV batch request for installed package versions failed with DNS resolution error. A web check confirmed pickle persistence risks but is not a version-specific vulnerability audit. No claim of zero known vulnerabilities is made. |
| REVIEW REQUIRED | Review all figure content and GitHub-rendered Mermaid/Markdown visually. Relative file links were checked locally; actual GitHub rendering was not exercised because publication is prohibited for this task. PNGs contain ordinary plotting metadata; no local paths were detected in that metadata. |

Final candidate scan covered 82 files before adding the inventory itself (83 intended public files in total), including decompressed gzip archives, decompressed joblib bytes and PNG metadata. No screened secret/email/local-path patterns matched. Every local Markdown file target resolves. A byte comparison of all 62 original output files against the checkpoint found only the two documented JSON metadata changes. The manifest lists every candidate except itself to avoid a self-referential hash. No actual staged-content or history scan was possible.

## Reproduction evidence and limits

Original test run: 39 tests, all passed. After path sanitisation: 39 tests, all passed. Final run: 41 tests, all passed. Test coverage includes conservation/capacity, fault effects, complete event recovery, event-separated chronological split, feature allowlisting, training-only references/calibration, protected attack fields, all 65 configurations, exact alpha-zero controls, reported-history causality and frozen assets.

`tools/reproduce.py` ran both simulation modes, baseline training (including the original diagnostic comparison), all 65 attack runs and the complete defensive evaluation. It compares all 32 existing CSV/CSV.gz files using pandas with atol=1e-12, zero relative tolerance; indices, columns and nonnumeric values must also match. It writes only into a new project-local directory. Stage logs and the comparison listing are in `.validation/reproduction/`. Existing original tests still use the retained source dataset/model; the helper verifies full regeneration independently.

No scientific behaviour was changed. The source changes affect default simulator output location (repository-relative, consistent with other scripts), trusted-model documentation and path identifiers in provenance. Explicit CLI relative paths still resolve from the caller's working directory. Original script CLI defaults can overwrite outputs; the README recommends isolated runs. Probability reductions may differ by tiny parallel floating-point accumulation; no measured result was adjusted or fabricated.

The retained truthful hybrid result has 122 false positives versus 2 for RF alone and unchanged recall of 423/490. The all-six alpha-1 attack has RF recall 0/490 and defended recall 490/490. The README reports both the apparent protection and its idealised balance assumptions/false-alarm cost.

## Significant changes by file

| File | Change |
| --- | --- |
| `README.md` | Professional 14-section technical overview, accurate diagram, retained figures, actual feature/model/attack definitions, reproduction commands, truthful false-alarm tradeoff, limits and coursework acknowledgement. |
| `.gitignore` | Excludes local environments, IDE/cache/config/credentials and private checkpoint/validation directories; intentionally retains experimental outputs. |
| `requirements.txt` | Pins the five actually imported direct packages to the recorded research versions. |
| `requirements-lock.txt` | Full observed installed package snapshot, including transitive dependencies. It is not a hash-verified lock or a cross-platform guarantee. |
| `simulate.py` | Defaults output to the project directory even when launched from another working directory; no simulation equations, seeds, controller, event logic or labels changed. |
| `evaluate_attacks.py` | Hash maps use role-relative identifiers; added explicit trust warning at joblib loading. No attack selection, interpolation or RF logic changes. |
| `evaluate_defence.py` | Stage 4 hash-map identifiers become portable. No calibration or rule changes. |
| `output/stage4/run_metadata.json` | Removed local path prefixes from provenance keys; all content digests unchanged. |
| `output/stage5/run_metadata.json` | Same key sanitisation; Stage 4 metadata hash updated to represent its sanitised bytes. |
| `tools/reproduce.py` | Non-destructive full pipeline runner with saved-output numerical comparisons and per-stage logs; refuses an existing/out-of-project destination. |
| `test_release.py` | Two lightweight regressions: hash maps survive relocation and still detect file changes; Stage 4 keys contain no machine paths. |
| `docs/EXPERIMENT_NOTES.md` | Preserves the existing detailed scientific documentation, with professional title and a default-output overwrite caveat. |
| `docs/RELEASE_REVIEW.md` | This audit, limitations, changes and manual release gates. |
| `docs/PUBLIC_FILES.tsv` | Intended public file inventory, sizes and hashes (excludes itself to avoid a self-referential digest). |

All other simulation, model, attack, defence, plotting and original test modules are retained. Their existing functions already have useful scientific docstrings and validation; broad renaming/rearchitecture would add risk without improving this small project. No redundant research outputs were deleted. Original result tables, figures, compressed archives and the frozen model remain unchanged apart from the two explicitly identified metadata files.

## Final manual release gates

1. Obtain university publication permission and confirm authorship/third-party attribution. Keep the repository private locally until resolved.
2. On a connected machine, create a Python 3.12 venv and install `requirements-lock.txt`; run `pip check`, all tests and the reproduction helper. If resolving versions requires changes, rerun and compare the science before updating recorded results.
3. Install `pip-audit` separately and audit the resolved environment (including transitive packages). Also run a dedicated secret scanner over the actual index before the first commit. Treat any discovered credential as compromised and revoke/rotate it without publishing its value.
4. Inspect `docs/PUBLIC_FILES.tsv`, figures and the actual `git diff --cached` once a real repository is initialised. This task has no staged-content result because there is no real index. Ensure `.release-backup/`, `.validation/`, `.idea/`, `.venv/` and hidden credential/config directories remain excluded.
5. After ownership is settled, consider MIT for original code if permissive reuse is desired; review separate data/figure and third-party obligations first. No licence file has been invented or added.

Proposed GitHub description: **Simulation-based study of adversarial sensor manipulation against Random Forest anomaly detection in a two-tank water process, with physical-consistency defence and reproducible evaluation.**

Suggested topics: `ics-security`, `ot-security`, `anomaly-detection`, `adversarial-machine-learning`, `cybersecurity`, `random-forest`, `industrial-control-systems`, `python`, `simulation`, `reproducible-research`.
