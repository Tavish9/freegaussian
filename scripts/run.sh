DATA_PATH=/mnt/hwfile/optimal/chenqizhi/LiveScene
CONTAINER=/mnt/hwfile/optimal/qudelin/apptainer/livescene/livescene
export PATH=$CONTAINER/opt/conda/bin:$PATH

scene_names=(
  seq001_transformer
  seq002_transformer
  seq003_door
  seq004_dog
  seq005_sit
  seq006_stand
  seq007_flower
  seq008_office

  blender
  face-2-attributes
  face-3-attributes
  face-3-attributes-bigger-annotations
  face-3-attributes-smaller-annotations
  fast-blinking
  metronome
  transformer
  two-metronomes

  seq001_Rs_int
  seq002_Rs_int
  seq003_Ihlen_1_int
  seq004_Ihlen_1_int
  seq005_Beechwood_0_int
  seq006_Beechwood_0_int
  seq007_Beechwood_0_int

  seq008_Benevolence_1_int
  seq009_Benevolence_1_int
  seq010_Merom_1_int

  seq011_Merom_1_int
  seq012_Pomaria_1_int 
  seq013_Pomaria_1_int
  seq014_Wainscott_0_int
  seq015_Wainscott_0_int
  seq016_Wainscott_0_int

  seq017_Benevolence_1_int
  seq018_Benevolence_1_int
  seq019_Rs_int
  seq020_Merom_1_int
)

for scene_name in "${scene_names[@]}"; do
  cur_date=$(date "+%H-%M-%S")
  date_dir=$(date "+%Y-%m-%d")
  trainer_config=$(python scripts/parse_config.py $DATA_PATH config $scene_name)

  note=default
  mkdir -p ./outputs/$date_dir

  sbatch -p optimal --gres=gpu:1 -J ${scene_name} \
      -o ./outputs/${date_dir}/${cur_date}_${scene_name}_${note}_%j.out \
      --wrap="PYTHONUNBUFFERED=1 \
              apptainer exec --nv --bind /mnt:/mnt --writable $CONTAINER \
              ns-train freegaussian $trainer_config"
  sleep 1
done
