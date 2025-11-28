from __future__ import annotations

from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.configs.base_config import ViewerConfig

from freegaussian.freegaussian_pipeline import FreeGaussianPipelineConfig
from freegaussian.freegaussian_model import FreeGaussianModelConfig
from freegaussian.freegaussian_control_model import FreeGaussianControlModelConfig, FreeGaussianModelConfig
from freegaussian.datamanager.freegaussian_datamanager import FreeGaussianImageDatamanagerConfig
from freegaussian.datamanager.freegaussian_dataparser import (
    FreeGaussianDataParserConfig,
    FreeGaussianSyntheticDataParserConfig,
)

freegaussian_method = MethodSpecification(
    TrainerConfig(
        method_name="freegaussian",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=FreeGaussianPipelineConfig(
            datamanager=FreeGaussianImageDatamanagerConfig(
                dataparser=FreeGaussianSyntheticDataParserConfig(
                    load_flow=True,
                    load_mask=True,
                    includes_time=True,
                ),
                cache_images_type="uint8",
            ),
            model=FreeGaussianModelConfig(),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4 * 5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6 * 5,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.001 * 5, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "deform": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4 * 5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=30000),
            },
            "control": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4 * 5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=15000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="FreeGaussian model for dynamic scenes with lang control",
)

