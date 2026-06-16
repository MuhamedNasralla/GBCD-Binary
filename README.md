# GBCD: A Berlin-Specific Dataset for Building Change Detection from High-Resolution Aerial Orthophotos

GBCD-Binary is the first annotated building change detection dataset constructed specifically for Berlin, Germany. It provides 1,769 bi-temporal orthophoto tile pairs covering all twelve administrative districts of the city, captured at 0.2 metres per pixel resolution from 2022 and 2025 Digital Color Orthophotos (DOP). The dataset was developed to address the underrepresentation of German and broader European urban environments in existing building change detection benchmarks, which remain heavily concentrated in East Asian and limited globally distributed contexts.

![Dataset sample](dataset.jpg)
*<!-- INSERT: visualize_sample.py output — Before / After (with red change contours) / Binary mask -->*

## Dataset Overview

| Property | Value |
|---|---|
| Total image pairs | 1,769 |
| Image resolution | 512 × 512 pixels |
| Ground resolution | 0.2 m / pixel |
| Ground coverage per tile | ~102 × 102 m |
| Temporal range | 2022 – 2025 |
| Geographic coverage | All 12 Berlin districts (~891 km²) |
| Annotation categories | Binary — changed / unchanged |
| Overall change pixel ratio | 5.64% |

| Split | Total Tiles | With Change | Without Change |
|---|---|---|---|
| Train | 1,417 | 637 (45.0%) | 780 (55.0%) |
| Validation | 176 | 83 (47.2%) | 93 (52.8%) |
| Test | 176 | 83 (47.2%) | 93 (52.8%) |

## Exploratory Data Analysis

![EDA charts](images/eda_charts.png)
*<!-- INSERT: gbcd_eda_charts.py output — pixel-level class distribution + tile distribution per split -->*

The dataset reflects a realistic class imbalance: building construction and demolition events are spatially sparse relative to Berlin's total built-up area over the three-year observation window, with changed pixels constituting only 5.64% of the total labelled area.

## Folder Structure

```
GBCD-Binary/
├── train/
│   ├── before/   # 2022 orthophoto tiles
│   ├── after/    # 2025 orthophoto tiles
│   └── label/    # Binary change masks
├── val/
│   ├── before/
│   ├── after/
│   └── label/
└── test/
    ├── before/
    ├── after/
    └── label/
```

Each tile triplet (`before`, `after`, `label`) shares an identical filename across the three folders.

## Annotation Methodology

Building changes were identified using a hybrid annotation workflow combining AI-assisted segmentation via the [AI Segmentation Plugin for QGIS](https://plugins.qgis.org/plugins/AI_Segmentation/) with systematic manual review and correction. Given the variable performance of automated segmentation across different urban morphologies and lighting conditions, all AI-generated candidate annotations were verified and, where necessary, redrawn manually to ensure spatial precision and label consistency.

Annotated categories include newly constructed buildings, demolished structures, and significant structural modifications visible between the two temporal acquisitions. Polygon masks were rasterised using [Rasterio](https://rasterio.readthedocs.io/) and [GeoPandas](https://geopandas.org/), with coordinate reference systems matched explicitly between imagery and labels to prevent spatial misalignment during tiling.

## Model Performance on GBCD-Binary

GBCD-Binary was used to fine-tune and evaluate three building change detection model configurations, all based on the [HybridSiam-CD](https://github.com/) architecture combining a frozen DINOv3 ViT-L encoder with a Siamese ResNet34 spatial branch.

| Model | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| HybridSiam-CD (zero-shot, no GBCD training) | 50.66% | 67.25% | 78.69% | 58.71% |
| HybridSiam-CD+ (fine-tuned on GBCD only) | 61.30% | 76.01% | 70.12% | 82.97% |
| HybridSiam-CD+ (trained on GBCD + 3 other datasets) | 66.06% | 79.56% | 75.54% | 84.04% |

![Model predictions comparison](images/model_predictions_comparison.png)
*<!-- INSERT: visualize_zeroshot.py / visualize_berlin.py / visualize_combined.py outputs — side-by-side Before/After/Ground Truth/Prediction grids for each model -->*

## Citation

If you use the GBCD-Binary dataset in your research, please cite:

```bibtex
@mastersthesis{nasralla2026gbcd,
  title  = {Deep Learning-Based Multi-Temporal Building Change Detection Using
            High-Resolution Aerial Orthophotos: A Case Study of Berlin, Germany},
  author = {Nasralla, Mohammed Ghafar},
  school = {Berlin School of Business and Innovation},
  year   = {2026}
}
```

## Data Source

Orthophotos were sourced from the [Berlin Open Data](https://daten.berlin.de/) platform via Web Map Service (WMS), provided under an open data licence by the Berlin state government.

## License

Specify your chosen license here (e.g. CC BY 4.0, MIT, etc.).

## Acknowledgements

This dataset was developed as part of an MSc Data Analytics dissertation at the Berlin School of Business and Innovation (BSBI), 2026.
