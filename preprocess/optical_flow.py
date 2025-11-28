import argparse
from pathlib import Path
from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow
import numpy as np
from tqdm import tqdm

argparser = argparse.ArgumentParser(description='Inference a model')
argparser.add_argument('--cfg', type=str, default="pretrained/gma_plus-p_8x2_120k_flyingthings3d_400x720.py", help='The config file of the model')
argparser.add_argument('--ckpt', type=str, default="pretrained/gma_plus-p_8x2_120k_flyingthings3d_400x720.pth", help='The checkpoint file of the model')
argparser.add_argument('--path', type=str, default="/DATA/LiveScene/Sim/seq001_Rs_int/images", help='The path of the image files')
argparser.add_argument('--interval', type=int, default=1, help='The interval of the image files')
argparser.add_argument('--viz', action='store_true', help='Visualize the flow')
args = argparser.parse_args()

if __name__ == '__main__':
    device = 'cuda:0'
    flow_path = Path(args.path).parent / 'opticalflow'
    flow_path.mkdir(exist_ok=True)

    image_files = sorted(Path(args.path).iterdir())[::args.interval]
    image1_files, image2_files  = image_files[:-1], image_files[1:]

    # in batch
    model = init_model(args.cfg, args.ckpt, device=device)
    for i, (image1, image2) in tqdm(enumerate(zip(image1_files, image2_files))):
        data = inference_model(model, image1, image2)
        # __import__('ipdb').set_trace()
        if args.viz: visualize_flow(data, f"{flow_path}/{image1.stem}-{image2.stem}.png")
        np.save(f"{flow_path}/{image1.stem}-{image2.stem}.npy", data)

