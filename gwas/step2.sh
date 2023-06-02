#!/bin/bash

## Prepare loco_list file
#loco_path=$(dirname ${STEP1_LOCO_FILE})

#loco_file_name=$(basename ${STEP1_LOCO_FILE})

#Loco_File=${loco_path}/${loco_file_name}

#echo "${PHENO_COL} ${Loco_File}" >> Step1_pred.list

#STEP1_LIST_FILE="Step1_pred.list"

echo -e "Starting REGENIE GWAS Run\n"
##
regenie \
	--step 2 \
	--bgen ${GENO_PATH}/imputed/${BGEN_FILE} \
	--sample ${BGEN_SAMPLE} \
	--bt \
	--htp UKB200k \
	--phenoFile ${PHENO_FILE} \
	--phenoColList ${PHENO_COL_LIST} \
	--covarFile ${PHENO_FILE} \
	--covarColList ${COVAR_COL_LIST} \
	--catCovarList ${CAT_COVAR_COL_LIST} \
	--bsize ${GENO_BLOCK_SIZE} \
	--firth \
	--ref-first \
	--approx \
	--split \
	--gz \
	--write-samples \
	--print-pheno \
	--pThresh ${P_THRESH} \
	--pred ${STEP1_LIST_FILE} \
	--minMAC ${MIN_MAC} \
	--threads $(nproc --all) \
	--out ${GWAS_OUT_PATH}/Chr${CHR}_${GENETIC_TEST}


