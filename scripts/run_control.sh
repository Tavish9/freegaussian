DATA_PATH=/mnt/hwfile/optimal/chenqizhi/LiveScene
CONTAINER=/mnt/hwfile/optimal/qudelin/apptainer/livescene/livescene
export PATH=$CONTAINER/opt/conda/bin:$PATH

ckpts=(
  # seq001_transformer
  # seq002_transformer
  # seq003_door # 30
  # seq004_dog # 30
  # seq005_sit
  # seq006_stand
  # "seq007_flower outputs/seq007_flower/freegaussian/2024-09-27_023041.896386/nerfstudio_models/step-000060000.ckpt"
  # seq008_office

  
  "seq001_Rs_int outputs/seq001_Rs_int/freegaussian/2024-09-30_165153.845860/nerfstudio_models/step-000030000.ckpt" 
  "seq004_Ihlen_1_int outputs/seq004_Ihlen_1_int/freegaussian/2024-09-28_111040.158757/nerfstudio_models/step-000030000.ckpt"
  "seq015_Wainscott_0_int outputs/seq015_Wainscott_0_int/freegaussian/2024-09-28_142140.810479/nerfstudio_models/step-000030000.ckpt"

  "seq002_Rs_int outputs/seq002_Rs_int/freegaussian/2024-09-27_193221.829371/nerfstudio_models/step-000030000.ckpt"
  "seq003_Ihlen_1_int outputs/seq003_Ihlen_1_int/freegaussian/2024-09-27_193224.101855/nerfstudio_models/step-000030000.ckpt"
  "seq005_Beechwood_0_int outputs/seq005_Beechwood_0_int/freegaussian/2024-09-27_193230.077012/nerfstudio_models/step-000030000.ckpt"
  "seq006_Beechwood_0_int outputs/seq006_Beechwood_0_int/freegaussian/2024-09-28_172748.016888/nerfstudio_models/step-000030000.ckpt"
  "seq007_Beechwood_0_int outputs/seq007_Beechwood_0_int/freegaussian/2024-09-27_193236.764279/nerfstudio_models/step-000030000.ckpt"
  "seq008_Benevolence_1_int outputs/seq008_Benevolence_1_int/freegaussian/2024-09-28_101908.854325/nerfstudio_models/step-000030000.ckpt"

  "seq009_Benevolence_1_int outputs/seq009_Benevolence_1_int/freegaussian/2024-09-28_185503.152006/nerfstudio_models/step-000030000.ckpt"
  "seq010_Merom_1_int outputs/seq010_Merom_1_int/freegaussian/2024-09-27_193251.767337/nerfstudio_models/step-000040000.ckpt"
  "seq011_Merom_1_int outputs/seq011_Merom_1_int/freegaussian/2024-09-24_112448.767837/nerfstudio_models/step-000030000.ckpt"
  "seq012_Pomaria_1_int  outputs/seq012_Pomaria_1_int/freegaussian/2024-09-24_131101.927284/nerfstudio_models/step-000030000.ckpt"
  "seq013_Pomaria_1_int outputs/seq013_Pomaria_1_int/freegaussian/2024-09-24_021910.768802/nerfstudio_models/step-000030000.ckpt"
  "seq014_Wainscott_0_int outputs/seq014_Wainscott_0_int/freegaussian/2024-09-28_150636.778591/nerfstudio_models/step-000030000.ckpt"
  "seq016_Wainscott_0_int outputs/seq016_Wainscott_0_int/freegaussian/2024-09-28_224021.749738/nerfstudio_models/step-000030000.ckpt"
  "seq017_Benevolence_1_int outputs/seq017_Benevolence_1_int/freegaussian/2024-09-25_151402.826792/nerfstudio_models/step-000030000.ckpt"
  "seq018_Benevolence_1_int outputs/seq018_Benevolence_1_int/freegaussian/2024-09-27_193316.077428/nerfstudio_models/step-000030000.ckpt"
  "seq019_Rs_int outputs/seq019_Rs_int/freegaussian/2024-09-24_102447.064826/nerfstudio_models/step-000030000.ckpt"
  "seq020_Merom_1_int outputs/seq020_Merom_1_int/freegaussian/2024-09-28_171258.134659/nerfstudio_models/step-000030000.ckpt"
)

for ckpt in "${ckpts[@]}"; do
    cur_date=$(date "+%H-%M-%S")
    date_dir=$(date "+%Y-%m-%d")
    set -- $ckpt
    scene_name=$1
    ckpt_path=$2
    trainer_config=$(python scripts/parse_config.py $DATA_PATH control_config $scene_name $ckpt_path)

    note=default
    mkdir -p ./outputs/$date_dir

    sbatch -p optimal --gres=gpu:1 -J ctl_${scene_name} \
        -o ./outputs/${date_dir}/${cur_date}_${scene_name}_${note}_%j.out \
        --wrap="PYTHONUNBUFFERED=1 \
                apptainer exec --nv --bind /mnt:/mnt --writable $CONTAINER \
                ns-train freegaussian-control $trainer_config"
    sleep 3
done
