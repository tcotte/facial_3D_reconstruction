%%shell

s="georgios"
timestamp="D0"
rig_filepath="/content/facial_3D_reconstruction/artifacts/rig.json"
image_dir="/content/drive/MyDrive/IEC_Lyon/"
output_folder="/content/output_visia"
mkdir $output_folder

for m in "Standard 1" "Cross-Polarized" "Raked" "Parallel-Polarized"; do

  python /content/facial_3D_reconstruction/00_pipeline/roma_v2_export.py \
    --img-dir $image_dir \
    --subject $s \
    --session $timestamp \
    --modality "$m" \
    --step 1 \
    --mode tiled \
    --out "$output_folder"/roma2_"$s"_"$timestamp"_"$m".npz
done

python /content/facial_3D_reconstruction/00_pipeline/run_calibrate_roma.py \
  --roma "$output_folder"/roma2_"$s"_"$timestamp"_Standard 1.npz \
  --out rig_"$s".json

python /content/facial_3D_reconstruction/00_pipeline/fuse_modalities.py --subject $s --session $timestamp \
    --img-dir $image_dir \
    --roma-dir "/content" \
    --rig rig_"$s".json \
    --epi-max 5 \
    --texture-min 0.1 \
    --roma-fill

python /content/facial_3D_reconstruction/00_pipeline/outlier_filter.py \
  --fusion fusion_"$s"_"$timestamp".npz

python /content/facial_3D_reconstruction/04_validation/export_pointcloud.py \
  --fusion fusion_"$s"_"$timestamp".npz \
  --frontal "$image_dir""$s"_"$timestamp"_Frontal_"Standard 1".jpg \
  --out nuage_"$s"_"$timestamp"_fusion_mult_m_tiled.ply \
  --clip 0
