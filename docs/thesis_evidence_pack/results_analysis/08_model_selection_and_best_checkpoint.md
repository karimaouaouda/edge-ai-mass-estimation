# Model Selection and Best Checkpoint

Best checkpoint identified: `kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/best.pt` (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)  
Last checkpoint identified: `kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/last.pt` (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)

## Checkpoint and Export Artifacts

| artifact_path | type | purpose | should_use_for_thesis | should_use_for_deployment | notes |
| --- | --- | --- | --- | --- | --- |
| kaggle-output/artifacts/training/yolo/waste-seg-yolo/checkpoints/epoch_000054/best.pt | checkpoint | best weights snapshot inside managed checkpoint | supporting | no, prefer models/best.pt unless repository owner confirms otherwise | canonical run artifact; size=106181167 bytes |
| kaggle-output/artifacts/training/yolo/waste-seg-yolo/checkpoints/epoch_000054/weights.pt | checkpoint | managed checkpoint weights for resume/debug | supporting | no | canonical run artifact; size=106183087 bytes |
| kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/best.pt | checkpoint | best model checkpoint for evaluation/export | yes | candidate, after export and benchmarking | canonical run artifact; size=54472625 bytes |
| kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/last.pt | checkpoint | last epoch checkpoint for resume/debugging | supporting | no | canonical run artifact; size=54472625 bytes |
| kaggle-output/artifacts/training/yolo/waste-seg-yolo/training_runs/waste-seg-yolo/weights/best.pt | checkpoint | best weights snapshot inside managed checkpoint | supporting | no, prefer models/best.pt unless repository owner confirms otherwise | canonical run artifact; size=54472625 bytes |
| kaggle-output/artifacts/training/yolo/waste-seg-yolo/training_runs/waste-seg-yolo/weights/last.pt | checkpoint | training/resume/debug | no | no | canonical run artifact; size=54472625 bytes |
| kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/checkpoints/epoch_000054/best.pt | checkpoint | best weights snapshot inside managed checkpoint | supporting | no, prefer models/best.pt unless repository owner confirms otherwise | MLflow mirror/copy; size=106181167 bytes |
| kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/checkpoints/epoch_000054/weights.pt | checkpoint | managed checkpoint weights for resume/debug | supporting | no | MLflow mirror/copy; size=106183087 bytes |
| kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/weights/best.pt | checkpoint | best weights snapshot inside managed checkpoint | supporting | no, prefer models/best.pt unless repository owner confirms otherwise | MLflow mirror/copy; size=54472625 bytes |
| kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/weights/last.pt | checkpoint | training/resume/debug | no | no | MLflow mirror/copy; size=54472625 bytes |

## Recommendation

- Use `models/best.pt` as the thesis-reported model checkpoint because it is the repository's best model artifact and the training docs state that evaluation/export should use the best checkpoint.
- Use `last.pt` mainly for resume/debugging unless Karim confirms a repository-specific reason to prefer it.
- No ONNX or TensorRT export artifacts were found. Deployment claims on Jetson Nano should remain placeholders until export and benchmark artifacts exist.
- The large `checkpoints/epoch_000054/weights.pt` and `checkpoints/epoch_000054/best.pt` files are managed checkpoint snapshots; the smaller `models/best.pt` and `models/last.pt` are the final model artifacts copied from the Ultralytics run.
