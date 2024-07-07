import mmgclip
from mmgclip import seeding
import hydra
from omegaconf import DictConfig, OmegaConf
from attrdict import AttrDict

@hydra.main(version_base=None, config_path="configs", config_name="train_binary_class_clf")
def extract(cfg : DictConfig) -> None:
    OmegaConf.resolve(cfg)

    # convert the config dict to a class object
    mmgconfig = AttrDict(cfg)

    # set the seed value
    seeding(mmgconfig.base.seed)

    # structure the dataset path from the json annotated dataset
    dataset_sample_json = mmgclip.create_dataset_df(config=mmgconfig)

    # feature extraction
    mmgclip.image_feature_extractor(config=mmgconfig, dataset=dataset_sample_json).extract()
    
if __name__ == "__main__":
    extract()