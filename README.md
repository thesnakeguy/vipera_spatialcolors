# 🐍 Vipera aspis Color Variation Pipeline

> **Proof of concept** — leveraging opportunistic georeferenced image data from open biodiversity databases for quantitative color analysis in wildlife research.

---

## Overview

This repository contains a complete image processing and analysis pipeline to study **intraspecific color variation** in *Vipera aspis* (the asp viper). The workflow extracts and clusters color information from opportunistic photographs sourced from [GBIF](https://www.gbif.org/), with the goal of modeling color variation as a function of subspecies identity and environmental predictors:

```
color ~ subspecies + environment
```

*Vipera aspis* is a well-suited study system for this kind of analysis:
- Abundant georeferenced photographic records available via GBIF
- Subspecies well-characterized and geographically defined
- Clear color polymorphism present (dorsal background color, zigzag pattern color)
- Gene flow barriers present within the distribution range
- Rich environmental covariate data available across Europe

---

## Pipeline

### 1. 🏋️ Train Image Segmentation Model
Annotated training images are sourced from [Roboflow](https://roboflow.com/). A custom segmentation model is trained to isolate the snake from its background in field photographs.

### 2. 🌍 Apply Model to GBIF Images
The trained segmentation model is applied to georeferenced images downloaded from GBIF, producing masked outputs that isolate the animal body.

### 3. 🌑 Shadow Removal
A pretrained shadow removal model is applied to the segmented images to minimize the effect of lighting conditions on perceived color.

### 4. 🎨 Color Clustering
K-means clustering (or an alternative method) is applied to the shadow-corrected, segmented images to extract dominant colors. Two primary color variables are targeted:
- **Background color** — the dorsal base coloration
- **Zigzag color** — the contrasting dorsal pattern

---

## Goal

The study is designed as a **proof of concept** for using opportunistic citizen science imagery (e.g., iNaturalist observations aggregated on GBIF) as a valid data source for quantitative morphological research. If successful, the approach could be generalized to other taxa where large volumes of georeferenced photographs exist but systematic morphological sampling does not.

---

## Results

### Segmentation Output
![Segmentation example](__Vipera_segmentation.png)

### Color Clustering Output
![Color clustering example](__Vipera_colorclustering.png)

---

## Dependencies

> *Fill in based on your environment (e.g., Python version, key packages such as `ultralytics`, `scikit-learn`, `Pillow`, `rasterio`, etc.)*

---

## Usage

> *Add instructions for running the pipeline steps here.*

---

## Data Sources

- **Training annotations**: [Roboflow](https://roboflow.com/)
- **Occurrence images**: [GBIF](https://www.gbif.org/)
- **Environmental data**: e.g., CHELSA, WorldClim, Copernicus Land Cover

---

## Citation

> *Add citation or reference if/when published.*

---

## License

> *Specify license here.*
