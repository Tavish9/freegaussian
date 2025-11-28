DATA_PATH=/mnt/hwfile/optimal/chenqizhi/LiveScene
CONTAINER=/mnt/hwfile/optimal/qudelin/apptainer/livescene/livescene

configs=(
  outputs/seq001_Rs_int/freegaussian/2024-09-30_165153.845860/config.yml
  # outputs/seq002_Rs_int/freegaussian/2024-09-27_193221.829371/config.yml
  # outputs/seq003_Ihlen_1_int/freegaussian/2024-09-27_193224.101855/config.yml
  # outputs/seq004_Ihlen_1_int/freegaussian/2024-09-28_111040.158757/config.yml
  # outputs/seq005_Beechwood_0_int/freegaussian/2024-09-27_193230.077012/config.yml
  # outputs/seq006_Beechwood_0_int/freegaussian/2024-09-28_172748.016888/config.yml
  # outputs/seq007_Beechwood_0_int/freegaussian/2024-09-27_193236.764279/config.yml
  # outputs/seq008_Benevolence_1_int/freegaussian/2024-09-28_101908.854325/config.yml
  # outputs/seq009_Benevolence_1_int/freegaussian/2024-09-28_185503.152006/config.yml
  # outputs/seq010_Merom_1_int/freegaussian/2024-09-27_193251.767337/config.yml
  # outputs/seq011_Merom_1_int/freegaussian/2024-09-24_112448.767837/config.yml
  # outputs/seq012_Pomaria_1_int/freegaussian/2024-09-24_131101.927284/config.yml
  # outputs/seq013_Pomaria_1_int/freegaussian/2024-09-24_021910.768802/config.yml
  # outputs/seq014_Wainscott_0_int/freegaussian/2024-09-28_150636.778591/config.yml
  # outputs/seq015_Wainscott_0_int/freegaussian/2024-09-28_142140.810479/config.yml
  # outputs/seq016_Wainscott_0_int/freegaussian/2024-09-28_224021.749738/config.yml
  # outputs/seq017_Benevolence_1_int/freegaussian/2024-09-25_151402.826792/config.yml
  # outputs/seq018_Benevolence_1_int/freegaussian/2024-09-27_193316.077428/config.yml
  # outputs/seq019_Rs_int/freegaussian/2024-09-24_102447.064826/config.yml
  # outputs/seq020_Merom_1_int/freegaussian/2024-09-28_171258.134659/config.yml

)

for config in "${configs[@]}"; do
  echo "Processing $config"
  sbatch -p optimal -o outputs/%j.out \
        --wrap="PYTHONUNBUFFERED=1 \
                apptainer exec --nv --bind /mnt:/mnt --writable $CONTAINER \
                python preprocess/knn_gaussian.py --cfg $config"
  sleep 1
done