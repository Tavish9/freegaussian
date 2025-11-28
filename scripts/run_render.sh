CONTAINER=/mnt/hwfile/optimal/qudelin/apptainer/livescene/livescene

configs=(
    outputs/seq001_Rs_int/freegaussian/2024-09-27_193219.020895/config.yml
    outputs/seq002_Rs_int/freegaussian/2024-09-27_193221.829371/config.yml
    outputs/seq003_Ihlen_1_int/freegaussian/2024-09-27_193224.101855/config.yml
    outputs/seq004_Ihlen_1_int/freegaussian/2024-09-28_111040.158757/config.yml
    outputs/seq005_Beechwood_0_int/freegaussian/2024-09-27_193230.077012/config.yml
    outputs/seq006_Beechwood_0_int/freegaussian/2024-09-28_172748.016888/config.yml
    outputs/seq007_Beechwood_0_int/freegaussian/2024-09-27_193236.764279/config.yml
    outputs/seq008_Benevolence_1_int/freegaussian/2024-09-28_101908.854325/config.yml
    outputs/seq009_Benevolence_1_int/freegaussian/2024-09-28_185503.152006/config.yml
    outputs/seq010_Merom_1_int/freegaussian/2024-09-27_193251.767337/config.yml
    outputs/seq011_Merom_1_int/freegaussian/2024-09-24_112448.767837/config.yml
    outputs/seq012_Pomaria_1_int/freegaussian/2024-09-24_131101.927284/config.yml
    outputs/seq013_Pomaria_1_int/freegaussian/2024-09-24_021910.768802/config.yml
    outputs/seq014_Wainscott_0_int/freegaussian/2024-09-28_150636.778591/config.yml
    outputs/seq015_Wainscott_0_int/freegaussian/2024-09-28_142140.810479/config.yml
    outputs/seq016_Wainscott_0_int/freegaussian/2024-09-28_224021.749738/config.yml
    outputs/seq017_Benevolence_1_int/freegaussian/2024-09-25_151402.826792/config.yml
    outputs/seq018_Benevolence_1_int/freegaussian/2024-09-27_193316.077428/config.yml
    outputs/seq019_Rs_int/freegaussian/2024-09-24_102447.064826/config.yml
    outputs/seq020_Merom_1_int/freegaussian/2024-09-28_171258.134659/config.yml

    outputs/seq001_transformer/freegaussian/2024-09-29_013427.816548/config.yml
    outputs/seq002_transformer/freegaussian/2024-09-29_015450.813095/config.yml
    outputs/seq003_door/freegaussian/2024-09-27_015108.884247/config.yml
    outputs/seq004_dog/freegaussian/2024-09-27_022657.786000/config.yml
    outputs/seq005_sit/freegaussian/2024-09-26_101527.879910/config.yml
    outputs/seq006_stand/freegaussian/2024-09-26_101528.923589/config.yml
    outputs/seq007_flower/freegaussian/2024-09-27_023041.896386/config.yml
    outputs/seq008_office/freegaussian/2024-09-28_133944.140764/config.yml
)


for config in "${configs[@]}"; do
  echo "Processing $config"
  IFS='/'
  read -ra ADDR <<< "$config"

  seq_name=${ADDR[1]}
  method_name=${ADDR[2]}

  sbatch -p optimal2 --gres=gpu:1 -o outputs/%j.out \
  --wrap="PYTHONUNBUFFERED=1 \
    apptainer exec --nv --bind /mnt:/mnt --writable $CONTAINER \
    ns-render dataset \
    --load-config $config \
    --image-format png \
    --split train+test \
    --rendered-output-names depth \
    --output-path renders/${seq_name}/${method_name}"
  sleep 3
done
