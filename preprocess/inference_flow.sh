# config=../pretrained/raft_8x2_100k_flyingthings3d_sintel_368x768.py
config=../pretrained/raft_8x2_100k_mixed_368x768.py
# config=../pretrained/rgma_plus-p_8x2_120k_flyingthings3d_400x720.py
# config=../pretrained/rgma_plus-p_8x2_120k_mixed_368x768.py

# ckpt=../pretrained/raft_8x2_100k_flyingthings3d_sintel_368x768.pth
ckpt=../pretrained/raft_8x2_100k_mixed_368x768.pth
# ckpt=../pretrained/gma_plus-p_8x2_120k_flyingthings3d_400x720.pth
# ckpt=../pretrained/gma_plus-p_8x2_120k_mixed_368x768.pth

DATA_PATH=/DATA/LiveScene/Sim
scenes=(
    # seq001_Rs_int
    # seq001_Rs_int
    # seq003_Ihlen_1_int
    # seq004_Ihlen_1_int
    # seq005_Beechwood_0_int
    # seq006_Beechwood_0_int
    # seq007_Beechwood_0_int
    # seq008_Benevolence_1_int
    # seq009_Benevolence_1_int
    # seq010_Merom_1_int
    # seq011_Merom_1_int
    # seq012_Pomaria_1_int
    # seq013_Pomaria_1_int
    # seq014_Wainscott_0_int
    # seq015_Wainscott_0_int
    # seq016_Wainscott_0_int
    # seq017_Benevolence_1_int
    # seq018_Benevolence_1_int
    # seq019_Rs_int
    # seq020_Merom_1_int

    # demo002
    # opticalflow
    # static
    # static_rot
    # Beechwood_0_int_value_0.5
    # rot_xyz
    # rot_zxy
    # trans_xyz
    # one_point
    # seq001_room_norm
    # eular45
)


for scene_name in "${scenes[@]}"; do
  echo "Processing $scene_name"
  # python preprocess/epipolar_flow_bp.py \
  python preprocess/epipolar_flow.py \
    --cfg $config \
    --ckpt $ckpt \
    --data $DATA_PATH/$scene_name \
    --int 1 \
    --save
done
