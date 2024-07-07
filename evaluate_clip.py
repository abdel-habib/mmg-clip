import mmgclip
from mmgclip import seeding
import argparse
import os
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from attrdict import AttrDict
from loguru import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_path', type=str, help='Path to hydra experiment folder that contains an inner checkpoint folder. Start after `outputs/yyyy-mm-dd/XX-XX-XX`.', required=True)
    parser.add_argument('--train_split', type=bool, help="A boolean flag to use the same split during training, for splitting the test dataset as it was in the training. Default = True (only supported for now).", default=True)
    parser.add_argument('--run_name', type=str, help="The name of the folder to create where the new results will be saved in.", required=True)

    # get cmd args from the parser 
    logger.info(f"Excuting inference script running on the same dataset used in the experiment...")    
    args = parser.parse_args()

    # include `/outputs` in the experiment path
    args.experiment_folders_names = args.experiment_path
    args.experiment_path = os.path.join('outputs', args.experiment_folders_names)

    # validate if the experiment_path contains necessary folders and files.
    if not os.path.isdir(args.experiment_path) or 'checkpoints' not in os.listdir(args.experiment_path):
        raise ValueError("Wrong value passed for `experiment_path`. Pass the folder path insides the outputs directory, for instance 'yyyy-mm-dd/XX-XX-XX' Don't type `outputs/` in your path.")

    assert args.train_split == True, "`train_split`={args.train_split} is not supported. Only True value is supported."

    # create necessary paths
    args.export_dir = os.path.join(args.experiment_path, 'results')
    args.config_path = os.path.join(args.experiment_path, '.hydra')

    # read the config file information
    with initialize(config_path=args.config_path):
        cfg = compose(config_name="config")

    try:
        cfg['base']['export_dir'] = f'outputs/{args.experiment_folders_names}'
        cfg['base']['features_export_dir'] = 'outputs/dataset' # HAS TO POINT TO THE IMAGE-LABEL DATASET
        cfg['base']['results_export_dir'] = f'outputs/{args.experiment_folders_names}/{args.run_name}'
        # cfg['base']['tensorboard_export_dir'] = f'outputs/{args.experiment_folders_names}'
        cfg['checkpoints']['checkpoints_export_dir'] = f'outputs/{args.experiment_folders_names}/checkpoints'        
        mmgconfig = AttrDict(cfg)

    except Exception as e:
        print("An error occurred:", e)
        
    # set the seed same as the experiment
    seeding(mmgconfig.base.seed)

    if args.train_split:
        # create a dataset
        clf_dataset = mmgclip.get_dataset(dataset_name=mmgconfig.dataset.eval.dataset.name)(config=mmgconfig)
        logger.info(f"Description Example: {clf_dataset[0]['image_description']}")

        # split the dataset for train, val, and test
        _, val_clf_split = clf_dataset.random_split(dataset=clf_dataset, split='train')
        _, test_clf_split = clf_dataset.random_split(dataset=val_clf_split, split='test')
        logger.info(f"Test split len ({len(test_clf_split)})")

    # Preparing for evaluation with test split
    test_dataloader = mmgclip.DataLoaders(config=mmgconfig, dataset_split=test_clf_split).get_dataloader(
        **mmgconfig.dataloader.test,
        collate_fn=clf_dataset.collate_fn,
        )

    # create a new evaluator pointing to model checkpoint
    evaluator = mmgclip.Evaluator(
        config=mmgconfig, 
        test_dataloader=test_dataloader,
        tokenizer = clf_dataset.tokenizer)
    evaluator.evaluate_experiment()

