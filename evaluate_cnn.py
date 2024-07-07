import mmgclip
from mmgclip import seeding
import hydra
from omegaconf import DictConfig, OmegaConf
from attrdict import AttrDict
import torch

@hydra.main(version_base=None, config_path="configs", config_name="evaluate_cnn_clf")
def run(cfg : DictConfig) -> None:
    OmegaConf.resolve(cfg)

    # convert the config dict to a class object
    mmgconfig = AttrDict(cfg)

    # set the seed value
    seeding(mmgconfig.base.seed)
    
    # create a dataset
    clf_dataset = mmgclip.get_dataset(dataset_name=mmgconfig.dataset.eval.dataset.name)(config=mmgconfig)
    mmgclip.logger.info(f"Description Example: {clf_dataset[0]['image_description']}")
    
    # split the dataset for train, val, and test, we will use the same test split to compare this supervised learning with the clip approach
    _, val_clf_split = clf_dataset.random_split(dataset=clf_dataset, split='train')
    _, test_clf_split = clf_dataset.random_split(dataset=val_clf_split, split='test')
    mmgclip.logger.info(f"Test split len ({len(test_clf_split)})")

    # Preparing dataloader with test split for inference with the image encoder
    test_dataloader = mmgclip.DataLoaders(config=mmgconfig, dataset_split=test_clf_split).get_dataloader(
        **mmgconfig.dataloader.test,
        collate_fn=clf_dataset.collate_fn,
        )
    
    # load the model classifier
    model = torch.jit.load(mmgconfig.networks.image_encoder.convnext_tiny_clf_path)
    model.eval()

    # create a new evaluator and pass the model to it
    results = mmgclip.Evaluator(
        config=mmgconfig, 
        test_dataloader=test_dataloader,
        tokenizer = clf_dataset.tokenizer,
        cnn_eval=True).evaluate_cnn(cnn=model)

    mmgclip.logger.info(f"Results:\n{results}")


if __name__ == "__main__":
    run()