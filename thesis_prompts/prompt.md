You are working inside my full Edge AI Waste Characterization and Mass Estimation repository.

I have installed/downloaded the output artifacts of a YOLO segmentation training run. These artifacts may include training curves, validation curves, confusion matrices, precision-recall curves, F1 curves, result CSV files, metrics JSON files, annotated prediction samples, dataset manifests, model checkpoints, exported models, configs, and other training/evaluation outputs.

Your task is to deeply analyze these artifacts and update the existing thesis evidence pack so Prism AI can use the real results in my Master thesis.

Important:

* Do not run new training.
* Do not fabricate missing metrics.
* Do not invent results that are not present in the artifacts.
* Use only values that can be extracted from the installed artifacts.
* If a result is missing, keep a placeholder such as `[REAL_VALUE_REQUIRED]`.
* If you are uncertain about a metric meaning, mark it as `needs confirmation`.
* If an artifact path is missing or ambiguous, ask me before making unsupported assumptions.
* You may ask me important questions if needed.
* Do not modify core application code unless it is only to add a safe helper script for parsing artifacts.
* Do not delete or overwrite training artifacts.
* Do not expose secrets, tokens, or private credentials.

First, inspect the repository and locate the thesis evidence pack:

```text
docs/thesis_evidence_pack/
```

Then inspect the installed training output artifacts. The artifact root is:

```text
./kaggle-output
```

Also inspect any files named like:

```text
results.csv
metrics.json
metrics_history.json
evaluation.json
training_summary.json
dataset_manifest.json
optuna_best.json
exports_manifest.json
args.yaml
hyp.yaml
confusion_matrix.png
confusion_matrix_normalized.png
PR_curve.png
F1_curve.png
P_curve.png
R_curve.png
results.png
labels.jpg
labels_correlogram.jpg
val_batch*.jpg
train_batch*.jpg
pred*.jpg
```

Your goal is to produce a clean, thesis-ready results extension inside:

```text
docs/thesis_evidence_pack/results_analysis/
```

Create this structure:

```text
docs/thesis_evidence_pack/results_analysis/
│
├── README.md
├── 00_artifact_inventory.md
├── 01_training_run_summary.md
├── 02_dataset_manifest_analysis.md
├── 03_training_metrics_analysis.md
├── 04_validation_and_segmentation_metrics.md
├── 05_curves_and_visual_evidence.md
├── 06_class_level_performance_analysis.md
├── 07_error_analysis_and_failure_modes.md
├── 08_model_selection_and_best_checkpoint.md
├── 09_thesis_ready_results_section.md
├── 10_prism_ai_results_integration_notes.md
├── 11_missing_values_and_questions_for_karim.md
│
├── tables/
│   ├── training_run_summary.csv
│   ├── dataset_distribution.csv
│   ├── overall_metrics.csv
│   ├── class_level_metrics.csv
│   ├── curve_artifacts_index.csv
│   ├── checkpoint_artifacts.csv
│   └── missing_metrics.csv
│
└── figures_index/
    ├── figure_index.md
    └── selected_figures_for_thesis.md
```

Also update these existing evidence-pack files if they exist:

```text
docs/thesis_evidence_pack/05_training_pipeline_and_mlops.md
docs/thesis_evidence_pack/10_challenges_failures_and_solutions.md
docs/thesis_evidence_pack/11_project_contributions.md
docs/thesis_evidence_pack/12_results_placeholders_and_replacement_guide.md
docs/thesis_evidence_pack/13_prism_ai_thesis_generation_instructions.md
docs/thesis_evidence_pack/15_code_and_artifact_traceability.md
docs/thesis_evidence_pack/OPEN_QUESTIONS_FOR_KARIM.md
```

When updating existing files:

* Preserve previous content.
* Add a new section titled `Training Results Artifact Analysis`.
* Do not remove placeholders unless you have real values from artifacts.
* Replace placeholders only when the value is clearly supported by artifact files.
* Add traceability: every result must mention the file path it came from.

Detailed requirements:

1. `README.md`
   Explain:

* what this results-analysis folder contains,
* which artifact root was analyzed,
* what values were extracted,
* what values remain missing,
* how Prism AI should use this folder,
* warning: do not invent missing results.

2. `00_artifact_inventory.md`
   Create a full inventory of discovered artifacts.
   For each artifact, include:

* relative path,
* file type,
* likely purpose,
* whether it is useful for thesis,
* whether it contains extractable numeric results,
* notes.

Also create `tables/curve_artifacts_index.csv` listing all curve/image artifacts.

3. `01_training_run_summary.md`
   Extract and summarize:

* model name/checkpoint if available,
* task type: segmentation or detection,
* dataset YAML path if available,
* number of epochs,
* image size,
* batch size,
* optimizer,
* learning rate if available,
* augmentation settings if available,
* training device if available,
* run name,
* start/end timestamps if available,
* best checkpoint path,
* last checkpoint path,
* export artifacts if available.

If a value is not found, use `[NOT_FOUND_IN_ARTIFACTS]`.

4. `02_dataset_manifest_analysis.md`
   If `dataset_manifest.json` or similar exists, analyze:

* dataset sources,
* number of images,
* number of annotations,
* train/val/test split sizes,
* class list,
* class distribution,
* duplicate handling if recorded,
* rejected images/annotations if recorded,
* dataset fingerprint/hash if recorded.

If exact numbers are not available, keep placeholders and write what artifact is missing.

5. `03_training_metrics_analysis.md`
   Analyze `results.csv`, `metrics_history.json`, `metrics.json`, or similar.
   Extract:

* final epoch metrics,
* best epoch metrics,
* training losses,
* validation losses,
* precision,
* recall,
* mAP@50,
* mAP@50:95,
* mask mAP if available,
* box mAP if available,
* segmentation-specific metrics if available.

Create `tables/overall_metrics.csv` with columns:

```text
metric,value,source_file,epoch_or_step,notes
```

Also explain the difference between:

* box metrics,
* mask metrics,
* mAP@50,
* mAP@50:95,
* precision,
* recall,
  but do not over-explain inside the CSV.

6. `04_validation_and_segmentation_metrics.md`
   Focus on segmentation results.
   Extract all segmentation metrics available from artifacts.
   Clearly separate:

* bounding-box detection performance,
* mask/segmentation performance,
* validation performance,
* test performance if available.

If only validation metrics exist, say so clearly.
If test metrics are missing, add them to missing values.

7. `05_curves_and_visual_evidence.md`
   Analyze the visual curves and figures:

* training/validation loss curves,
* precision curve,
* recall curve,
* F1 curve,
* precision-recall curve,
* confusion matrix,
* labels distribution,
* annotated prediction samples,
* validation batches,
* qualitative prediction images.

For each figure:

* describe what it shows,
* explain how it can be used in the thesis,
* identify possible concerns, such as overfitting, unstable curves, class imbalance, weak classes, or confusion between similar waste categories.
  Do not claim a concern unless the figure or metrics support it.
  If visual interpretation is uncertain, write `visual interpretation needs confirmation`.

Create:

```text
figures_index/figure_index.md
figures_index/selected_figures_for_thesis.md
```

The selected figures file should recommend the best figures to include in the thesis and explain why.

8. `06_class_level_performance_analysis.md`
   If class-level metrics exist, extract them.
   Create `tables/class_level_metrics.csv` with columns:

```text
class_name,precision,recall,mAP50,mAP50_95,mask_mAP50,mask_mAP50_95,source_file,notes
```

If class-level metrics do not exist, create the file with placeholder rows and explain how to generate/evaluate them later.

Look for weak classes and strong classes.
If weak classes are found, explain possible reasons:

* class imbalance,
* visual similarity,
* small object size,
* deformation,
* occlusion,
* annotation noise,
* domain mismatch.

Do not invent reasons; label them as hypotheses unless supported.

9. `07_error_analysis_and_failure_modes.md`
   Use metrics, confusion matrix, and annotated prediction images to identify:

* false positives,
* false negatives,
* class confusion,
* poor mask quality,
* missed small objects,
* wrong category predictions,
* over-segmentation,
* under-segmentation,
* class imbalance effects.

Create a table:

```text
Observed issue | Evidence artifact | Possible cause | Thesis interpretation | Suggested mitigation | Confidence
```

Confidence must be one of:

* high,
* medium,
* low,
* needs Karim confirmation.

10. `08_model_selection_and_best_checkpoint.md`
    Identify:

* best checkpoint,
* last checkpoint,
* exported model files,
* whether ONNX/TensorRT exports exist,
* which artifact should be used for deployment,
* which artifact should be reported in thesis.

Create `tables/checkpoint_artifacts.csv` with:

```text
artifact_path,type,purpose,should_use_for_thesis,should_use_for_deployment,notes
```

Explain why `best.pt` is normally used for evaluation/export, while `last.pt` is mostly useful for resume/debugging, unless repository-specific evidence says otherwise.

11. `09_thesis_ready_results_section.md`
    Write a clean thesis-ready draft section titled:

```text
Experimental Results: YOLO Segmentation Training
```

This section should be academic and easy to paste into the thesis later.

It must include:

* short experimental setup,
* dataset summary,
* training configuration,
* quantitative results table,
* curve analysis,
* qualitative prediction analysis,
* class-level performance discussion if available,
* limitations,
* transition to future mass-estimation/inference evaluation.

Rules:

* Use real values only when extracted from artifacts.
* Use `[REAL_VALUE_REQUIRED]` when missing.
* Every numeric claim must include the source artifact path in parentheses.
* Do not claim final system performance on Jetson Nano unless Jetson benchmark artifacts exist.
* Do not claim mass-estimation performance unless mass metrics exist.

12. `10_prism_ai_results_integration_notes.md`
    Write instructions for Prism AI explaining:

* which files contain real training results,
* which values are safe to use,
* which values are still placeholders,
* how to cite artifact paths in the thesis draft,
* how to discuss weak classes or failures professionally,
* how not to overclaim.
  Tell Prism AI to distinguish training/validation segmentation results from full end-to-end mass-estimation results.

13. `11_missing_values_and_questions_for_karim.md`
    List all missing or ambiguous items, prioritized.
    Include questions such as:

* Is this the final training run or an intermediate one?
* Which model should be considered the thesis best model?
* Was the run trained on TACO only, RealWaste only, or merged dataset?
* Are these validation metrics or test metrics?
* Are there separate test-set results?
* Are there Jetson Nano FPS/latency benchmarks?
* Are there mass-estimation results?
* Which figures should be included in the thesis?
* Which weak classes should be discussed?

14. CSV files
    Create all CSV files with clean headers.
    If data is available, fill it.
    If missing, add placeholder rows with `missing` or `[REAL_VALUE_REQUIRED]`.

15. Update `12_results_placeholders_and_replacement_guide.md`
    Replace generic placeholders with real extracted values where safe.
    Keep missing values as placeholders.
    Add a section:

```text
## Extracted YOLO Segmentation Training Results
```

Include:

* the artifact root,
* extracted values,
* source files,
* values still missing.

16. Update `15_code_and_artifact_traceability.md`
    Add artifact-level traceability:

```text
Thesis claim | Artifact path | Extracted value | Status | Notes
```

Examples:

* YOLO segmentation model was trained.
* Best checkpoint was produced.
* Training curves were generated.
* Evaluation metrics were produced.
* Dataset manifest was generated.
* Export artifacts were produced.

17. Optional helper script
    If useful, create a safe script:

```text
scripts/analyze_training_artifacts.py
```

The script should:

* accept `--artifact-root`,
* parse common YOLO/Ultralytics result files,
* extract metrics into CSV/Markdown,
* never modify artifacts,
* never require GPU,
* never start training.

Only create this script if it helps reproducibility.

18. Final response
    After finishing, print:

* the artifact root analyzed,
* files created/updated,
* key extracted metrics,
* best checkpoint found,
* selected thesis figures,
* missing important values,
* top questions for Karim.

Remember:
This is for a Master thesis. Be honest, organized, traceable, and careful. The results must be usable by Prism AI without creating fake academic claims.
