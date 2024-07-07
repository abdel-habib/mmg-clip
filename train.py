import mmgclip
from mmgclip import seeding
import hydra
from omegaconf import DictConfig, OmegaConf
from attrdict import AttrDict
import torch
# import pandas as pd

@hydra.main(version_base=None, config_path="configs", config_name="train_binary_class_clf")
def run(cfg : DictConfig) -> None:
    OmegaConf.resolve(cfg)

    # convert the config dict to a class object
    mmgconfig = AttrDict(cfg)

    # set the seed value
    seeding(mmgconfig.base.seed)
    
    # create a dataset
    clf_dataset = mmgclip.get_dataset(dataset_name=mmgconfig.dataset.name)(config=mmgconfig)
    mmgclip.logger.info(f"Description Example: {clf_dataset[0]['image_description']}")
    mmgclip.logger.info(f"Features Shape: {clf_dataset[0]['image_features'].shape}")

    # split the dataset for train, val, and test
    train_clf_split, val_clf_split = clf_dataset.random_split(dataset=clf_dataset, split='train')
    mmgclip.logger.info(f"Train split len: ({len(train_clf_split)}), Valid split len ({len(val_clf_split)}).")

    if mmgconfig.dataset.name == mmgconfig.dataset.eval.dataset.name:
        # if we train and evaluate using the same type of dataset, then split the valid with the test
        val_clf_split, test_clf_split = clf_dataset.random_split(dataset=val_clf_split, split='test')
        mmgclip.logger.info(f"Test split len ({len(test_clf_split)}).")

    else:
        mmgclip.logger.info("Using different dataset for testing, thus not splitting validation dataset.")

    # export = []

    # for data in clf_dataset:
    #     image_id = data['image_id']
    #     image_description = data['image_description']

    #     if image_description != "":

    #         export.append({
    #             "image_id": image_id, 
    #             "image_description": image_description, 
    #             "file_path": mmgclip.create_path(image_id, base_dataset_path='/storage/Features/features/png_archive/2D_100micron/0/')
    #         })
        
    # # Convert the list of dictionaries to a DataFrame
    # df = pd.DataFrame(export)

    # # Export the DataFrame to a CSV file
    # csv_file = "santiago_image-prompts_experiment_data.csv"
    # df.to_csv(csv_file, index=False)

    # create dataloaders for train and val splits
    train_dataloader = mmgclip.DataLoaders(config=mmgconfig, dataset_split=train_clf_split).get_dataloader(
        **mmgconfig.dataloader.train,
        collate_fn=clf_dataset.collate_fn,
        )
    
    val_dataloader = mmgclip.DataLoaders(config=mmgconfig, dataset_split=val_clf_split).get_dataloader(
        **mmgconfig.dataloader.valid,
        collate_fn=clf_dataset.collate_fn,
        )

    test_dataloader = mmgclip.DataLoaders(config=mmgconfig, dataset_split=test_clf_split).get_dataloader(
        **mmgconfig.dataloader.test,
        collate_fn=clf_dataset.collate_fn,
        ) if mmgconfig.dataset.name == mmgconfig.dataset.eval.dataset.name else None # have a test dataloader only when we split the valid
    
    # if we were to train with p percentage of the training dataset, the config controls the percentage
    if mmgconfig.dataset.percentage.name != "100percent":
        mmgclip.logger.info(f"Using only {mmgconfig.dataset.percentage.config.percentage}% of training data >> Initial train_dataloader length: {len(train_dataloader)*mmgconfig.dataloader.train.batch_size}.")
        
        train_dataloader = mmgclip.dataloader_percentage(train_dataloader, mmgconfig, collate_fn=clf_dataset.collate_fn)

    mmgclip.logger.info(f"train_dataloader length: {len(train_dataloader)*mmgconfig.dataloader.train.batch_size}, val_dataloader length: {len(val_dataloader)*mmgconfig.dataloader.valid.batch_size}.")

    # create a new experiment and run it
    experiment_class = mmgclip.create_experiment(experiment_name=mmgconfig.experiments.config.experiment_name) # get the experiment class
    
    # initialize the class
    experiment = experiment_class(config = mmgconfig, 
                                  train_dataloader=train_dataloader, 
                                  valid_dataloader=val_dataloader,
                                  test_dataloader=test_dataloader,
                                  tokenizer=clf_dataset.tokenizer)
    experiment.run()

if __name__ == "__main__":
    run()