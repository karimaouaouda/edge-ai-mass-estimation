# Dataset Manifest Analysis

Source artifact: `kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/lineage/dataset_manifest.json`

## Overview

- Task: `segment`
- Dataset YAML: `dataset.yaml`
- Dataset fingerprint: `7a66d7ddbd3e522cb0b7fa8d09839de4e4ce58638075b204bfd64b49c34c32e6`
- Data config digest: `a9dccdf3ec99569a51f02dfb6fc81420e45eaae91cd808e9350e831581aa5253`
- Total images: 6460
- Total annotations/instances: 9702
- Duplicate images recorded: 0
- Classes: organic_waste, plastic_bottle, plastic_bag, rigid_plastic, metal_can, glass, paper_cardboard, mixed_waste

## Dataset Sources

| source | images | instances | invalid_annotations | empty_images | box_polygon_annotations | notes |
| --- | --- | --- | --- | --- | --- | --- |
| aquatrash | 368 | 450 | 0 | 0 | 0 | resolved root: /kaggle/input/datasets/harshpanwar/aquatrash/Images |
| realwaste | 4592 | 4623 | 0 | 5 | 160 | resolved root: /kaggle/input/datasets/joebeachcapital/realwaste/realwaste-main/RealWaste |
| taco | 1500 | 4629 | 155 | 19 | 0 | resolved root: /kaggle/working/data/raw/taco |

## Split Sizes

| split | images | instances | images_by_source | instances_by_class |
| --- | --- | --- | --- | --- |
| test | 969 | 1574 | {"aquatrash": 55, "realwaste": 689, "taco": 225} | {"glass": 173, "metal_can": 212, "mixed_waste": 386, "organic_waste": 110, "paper_cardboard": 241, "plastic_bag": 110, "plastic_bottle": 114, "rigid_plastic": 228} |
| train | 4522 | 6614 | {"aquatrash": 258, "realwaste": 3214, "taco": 1050} | {"glass": 434, "metal_can": 774, "mixed_waste": 1657, "organic_waste": 508, "paper_cardboard": 1043, "plastic_bag": 664, "plastic_bottle": 457, "rigid_plastic": 1077} |
| val | 969 | 1514 | {"aquatrash": 55, "realwaste": 689, "taco": 225} | {"glass": 101, "metal_can": 199, "mixed_waste": 367, "organic_waste": 109, "paper_cardboard": 218, "plastic_bag": 138, "plastic_bottle": 113, "rigid_plastic": 269} |

## Overall Class Distribution

| class_name | total_instances |
| --- | --- |
| organic_waste | 727 |
| plastic_bottle | 684 |
| plastic_bag | 912 |
| rigid_plastic | 1574 |
| metal_can | 1185 |
| glass | 708 |
| paper_cardboard | 1502 |
| mixed_waste | 2410 |

## Data Quality Notes

- RealWaste records 160 box-polygon annotations and 160 removed images, which supports mentioning a mask-quality/data-cleaning step.
- TACO records 155 invalid annotations, 167 dimension mismatches, 19 empty images, and 234 multipart reductions.
- AquaTrash records no invalid annotations, missing images, or bbox fallbacks in the manifest.
- Exact duplicate handling is recorded as 0 duplicate images.
