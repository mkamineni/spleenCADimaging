
  #!/bin/bash

## bash /home/muddin/ukb200k/run_regenie_step1_LOOCV_Jun23_2021.sh /home/muddin/ukb200k/regenie_step1_LOOCV_Jun23_2021.sh UKB200k_23Jun2021 "hasCHIP,hasExpandedCHIP,hasDNMT3A,hasTET2,hasASXL1" "AGE_assessment,Age2,Sex_Genetic,GenoBatchUKBiLEVE,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10" "ever_nerver_smoked,Ethnic_Background" gcr.io/ukbb-analyses/regenie:v2.0.2.gz

plink_path="$(dirname "${PLINK_FILES}")"
plink_base="$(basename "${PLINK_FILES//\*}")"
PLINK_Prefix=${plink_path}/${plink_base}
## set -euo pipefail
## Running with "--loocv" to avoid convergence issue, due to highly unbalanced case control ratio (source:https://rgcgithub.github.io/regenie/faq/ )
regenie \
	--step 1 \
	--loocv \
	--print-prs \
	--bed ${PLINK_Prefix} \
	--extract ${KEEP_SNPs} \
	--keep ${KEEP_SAMPLE} \
	--phenoFile ${PHENO_FILE} \
	--phenoColList ${PHENO_COL_LIST} \
	--covarFile ${PHENO_FILE} \
	--covarColList ${COVAR_COL_LIST} \
	--catCovarList ${CAT_COVAR_COL_LIST} \
	--bt \
	--bsize ${BIN_SIZE} \
	--lowmem tmpdir/regenie_tmp_preds \
	--threads $(nproc --all) \
	--out ${OUTPUT_PATH}/NULL_MODEL

###  
sed 's:output:input:g' ${OUTPUT_PATH}/NULL_MODEL_pred.list > ${OUTPUT_PATH}/NULL_MODEL_pred_modified.list

