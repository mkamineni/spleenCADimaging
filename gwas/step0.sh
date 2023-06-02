trait=spleen


wdir=/MRI_spleen/gwas/regenie/${trait}
phenofile=/MRI_spleen/gwas/data/${trait}_UKB_2.txt
covarfile=/MRI_spleen/gwas/data/${trait}_UKB_2.txt
covarCol=age,PC{1:10}
catcovar=batch,array
snplist=/MRI_spleen/gwas/regenie/plink_ukb/qc_pass.snplist
samplist=/MRI_spleen/gwas/regenie/plink_ukb/qc_pass.id
out=${trait}
bed=/MRI_spleen/gwas/regenie/plink_ukb/ukb_cal

mkdir -p ${wdir}
cd ${wdir}
###############################
rm mergelist.txt
# creating symbolic links between data paths and shorter names, and then writing the shorter names into a merged list
# bed files contain genotype data
# bim files contain SNP names and their locations in chromosomes
# fam files contain information about the individuals
# want to go through all the chromosomes and create these symbolic links
for chr in {1..22}; do 
  ln -s /broad/ukbb/genotype/ukb_cal_chr${chr}_v2.bed ukb_chr${chr}_v2.bed
  ln -s /broad/ukbb/genotype/ukb_snp_chr${chr}_v2.bim ukb_chr${chr}_v2.bim
  ln -s /medpop/esp2/pradeep/UKBiobank/v2data/ukb708_cal_chr1_v2_s488374.fam ukb_chr${chr}_v2.fam
  echo "ukb_chr${chr}_v2" >> mergelist.txt
done

# with make_bed, creates a new binary fileset and applies any other parameters you specify 
plink --merge-list mergelist.txt --make-bed --memory 64000 --out ukb_cal

# now applying different filters, eliminate minor alleles that have a frequency of less than 0.01, mac is minimum count, hardy weinberg equil of 1e-15 meaning that we exclude SNPs that fail HWE at a sig threshold of 1e-15, geno: only including SNPs with at least a 90% genotyping rate
# For EUR, use --keep with EUR samples
plink2 \
  --bfile ukb_cal \
  --maf 0.01 --mac 100 --geno 0.1 --hwe 1e-15 \
  --mind 0.1 \
  --write-snplist --write-samples --no-id-header \
  --out qc_pass