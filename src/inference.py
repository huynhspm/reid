import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchreid.reid.utils import FeatureExtractor


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    reid: FeatureExtractor = instantiate(cfg.reid)
    import torch
    input = torch.randn(10, 3, 256, 256)
    features = reid(input)
    print(features.shape)

if __name__ == "__main__":
    main()