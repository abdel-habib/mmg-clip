import mmgclip
from mmgclip import seeding
import hydra
from omegaconf import DictConfig, OmegaConf
from attrdict import AttrDict
import pandas as pd

@hydra.main(version_base=None, config_path="configs", config_name="train_exam_reports_clf")
def extract(cfg : DictConfig) -> None:
    OmegaConf.resolve(cfg)

    # convert the config dict to a class object
    mmgconfig = AttrDict(cfg)

    # set the seed value
    seeding(mmgconfig.base.seed)

    # load the post-processed translated dataset
    mmgclip.logger.info(f"Loading {mmgconfig.dataset.config.post_translation_dataset_path} file...")
    postprocessed_tr_dataset = pd.read_csv(mmgconfig.dataset.config.post_translation_dataset_path, encoding="latin1", index_col=0, dtype = str)
    # print(postprocessed_tr_dataset.iloc[0])
    # print(postprocessed_tr_dataset.iloc[0]['study_path'])

    # feature extraction
    # mmgclip.study_feature_extractor(config=mmgconfig, dataset=postprocessed_tr_dataset).extract()

    processed_dataset = mmgclip.map_path_to_features(df=postprocessed_tr_dataset, 
                                                     config=mmgconfig, 
                                                     export_dir=f'data/{mmgconfig.dataset.config.post_translation_fileid}/', 
                                                     export=True)
    mmgclip.logger.info(f"Final dataset shape: {processed_dataset.shape}")
    
if __name__ == "__main__":
    extract()