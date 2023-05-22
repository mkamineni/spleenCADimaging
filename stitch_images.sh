counter=434
limit=435 #make sure to do 1 more than the number of iterations you want to do
increment=100

START="$(date +%s)"

while [ $counter -ne $limit ]
do
    #rm -r segments/*
    #rm -R outputs/*
    #rm -R data/*
    #rm -R stitched_data/*
    #rm -R nnunet_data/*
    rm -R predictions/*
    #filenum=$((counter * increment))
    #echo $filenum
    #gsutil ls gs://bulkml4h/bodymri/all/raw/*_2_0.zip | head -n $filenum | tail -n $increment | gsutil -m cp -I data/
    #gsutil cp gs://bulkml4h/bodymri/all/raw/1579710_20201_2_0.zip data/
    #gsutil cp gs://bulkml4h/bodymri/all/raw/1686757_20201_2_0.zip data/
    #gsutil cp gs://bulkml4h/bodymri/all/raw/2352592_20201_2_0.zip data/
    #gsutil cp gs://bulkml4h/bodymri/all/raw/2521842_20201_2_0.zip data/
    #gsutil cp gs://bulkml4h/bodymri/all/raw/2748543_20201_2_0.zip data/
    #gsutil cp gs://bulkml4h/bodymri/all/raw/2756845_20201_2_0.zip data/
    #gsutil cp gs://bulkml4h/bodymri/all/raw/4895744_20201_2_0.zip data/
    #gsutil cp gs://bulkml4h/bodymri/all/raw/4931981_20201_2_0.zip data/
    #gsutil cp gs://bulkml4h/bodymri/all/raw/5117700_20201_2_0.zip data/

    #COPY=$[ $(date +%s) - ${START} ]
    #START="$(date +%s)"
    #echo "Time to Copy (s): ${COPY}"

    #stitching
    #python extract_ukbb.py --zip_folder data/ --nifti_folder stitched_data/
   
    #formatting for nnunet
    #python convert2nnunet.py --nifti_folder stitched_data/ --nnunet_folder nnunet_data2/ --dataset_name ukbb --num_channels 4 < answers.txt

    #STITCH=$[ $(date +%s) - ${START}]
    START="$(date +%s)"
    #echo "Time to Stitch (s): ${STITCH}"

    #predictions
    #export CUDA_VISIBLE_DEVICES=0
    #export RESULTS_FOLDER=models/
    #python predict.py --nnunet_folder nnunet_data2/ --prediction_folder predictions/ --dataset_name ukbb --num_channels 4 \
    #--num_threads_preprocessing 12 --num_threads_nifti_save 4
    
    #convert to outputs
    #python convert2original.py --prediction_folder predictions/ --output_folder outputs/
    
    #PREDICT=$[ $(date +%s) - ${START} ]
    #START="$(date +%s)"
    #echo "Time to predict (s): ${PREDICT}"

    #extract voxels
    python extractvoxel.py --increment $increment --counter $counter
    #VOXEL=$[ $(date +%s) - ${START} ]
    #START="$(date +%s)"
    #echo "Time to extract voxels (s): ${VOXEL}"
    

    pyradiomics radiomics/radiomics_filenames_1.csv -o radiomics/radiomics_results_1_${counter}.csv -f csv --jobs 16
    pyradiomics radiomics/radiomics_filenames_2.csv -o radiomics/radiomics_results_2_${counter}.csv -f csv --jobs 16
    
    rm radiomics/radiomics_filenames_1.csv
    rm radiomics/radiomics_filenames_2.csv
    #RAD=$[ $(date +%s) - ${START} ]
    #START="$(date +%s)"
    #echo "Time to apply radiomics (s): ${RAD}"

    counter=$((counter+1))
    echo $counter
done
echo $dt