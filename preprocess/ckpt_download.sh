configs=(
  https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_flyingthings3d_sintel_368x768.py
  https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_mixed_368x768.py
  https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_flyingthings3d_400x720.py
  https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_mixed_368x768.py
)

ckpts=(
  https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_flyingthings3d_sintel_368x768.pth
  https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_mixed_368x768.pth
  https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_flyingthings3d_400x720.pth
  https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_mixed_368x768.pth
)

mkdir -p ../pretrained

for i in "${!configs[@]}"; do
  config=${configs[$i]}
  ckpt=${ckpts[$i]}

  echo "Downloading $config"
  wget -c -P ../pretrained $config --no-check-certificate
  wget -c -P ../pretrained $ckpt --no-check-certificate
done
