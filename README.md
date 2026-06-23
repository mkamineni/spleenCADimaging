# Spleen CAD Imaging

## Description
Abdominal-MRI derived splenic radiomics pipeline for studying associations between spleen imaging features and coronary artery disease (CAD).

## Overview
Emerging evidence suggests that the spleen plays an important role in inflammatory pathways and cardiovascular disease. 
This project develops and applies a radiomics-based framework to quantify spleen morphology and texture from abdominal MRI imaging and evaluate associations with coronary artery disease.
MRI images were acquired from different stations and stitched together using a publicly available stitching algorithm (https://github.com/biomedia-mira/stitching) (1).
Organ segmentation was implemented using a pre-trained deep learning model: (code: Abdominal Organ Segmentations of UK Biobank (UKBB) and German National Cohort (GNC) Studies, trained models: https://gitlab.com/turkaykart/ukbb-gnc-abdominal-segmentation) (2-4)

## Workflow
- Abdominal MRI
    ↓
- Image Stitching
    ↓
- Organ Segmentation
    ↓
- Spleen Radiomic Feature Extraction
    ↓
- Feature Selection and Modeling
    ↓
- Association with Prevalent and Incident CAD

## Repository Structure
* stitch_segment: code to stitch together MRI images acquired from different stations and then segment out various organs, utilzing a pre-trained model on UK Biobank data
* predictCAD: preprocess data and implement models to identify splenic radiomics features that are associated with prevalent and incident CAD
* radiomics: extracted radiomics features for both spleen and liver

## Related Repositiories
* spleenCADimaging: MRI processing, segmentation, and radiomics extraction
* spleenCADgenetics: GWAS and downstream genetic analyses
* spleenResultsAnalysis: gene prioritization and visualization of data into figures
* spleen_replication: External replication analyses in independent Massachusetts General Brigham Biobank (MGBB) cohort

## Publication
- If you use this repository, please cite: 
Kamineni, M., Raghu, V., Truong, B., Alaa, A., Schuermans, A., Friedman, S., Reeder, Ch., Bhattacharya, R., Libby, P., Ellinor, P.T., Maddah, M., Philippakis, A., Hornsby, Wh., Yu, Zh., Natarajan, P., (2024). Deep learning-derived splenic radiomics, genomics, and coronary artery disease. medRxiv.

## Author
Meghana Kamineni, MD

Harvard Medical School, Massachusetts General Hospital 

## References
1. I. Lavdas, B. Glocker, D. Rueckert, S. A. Taylor, E. O. Aboagye, A. G. Rockall, Machine learning in whole-body MRI: experiences and challenges from an applied study using multicentre data. Clin. Radiol. 74, 346–356 (2019).
2. T. Kart, M. Fischer, S. Winzeck, et al., Automated imaging-based abdominal organ segmentation and quality control in 20,000 participants of the UK Biobank and German National Cohort Studies. Sci. Rep. 12, 18733 (2022). 
3. T. Kart, M. Fischer, T. Küstner, et al., Deep Learning-Based Automated Abdominal Organ Segmentation in the UK Biobank and German National Cohort Magnetic Resonance Imaging Studies. Invest. Radiol. 56, 401–408 (2021). 
4. S. Gatidis, T. Kart, M. Fischer, et al., Better Together: Data Harmonization and Cross-Study Analysis of Abdominal MRI Data From UK Biobank and the German National Cohort. Invest. Radiol. 58, 346–354 (2023). 



