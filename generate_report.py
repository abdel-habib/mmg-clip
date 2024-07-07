import warnings
warnings.filterwarnings("ignore")

import mmgclip
import argparse
import os
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from attrdict import AttrDict
from loguru import logger
from PIL import Image
import torchvision.transforms as transforms
import torch
import re

torch.cuda.empty_cache()

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, help='Path to hydra experiment folder that contains a checkpoint folder. Start after `outputs/yyyy-mm-dd/XX-XX-XX`.', required=True)
    parser.add_argument('--image_id', type=str, help='Image filename to run inference on. It has a format of `p/{10 letters}/{cl/cr/ml/mr}`.', default=None)
    parser.add_argument('--exam_id', type=str, help='Exam folder to run inference on. It has a format of `10 letters`.', default=None)

    # get cmd args from the parser 
    logger.info(f"Generating report script excuted.")    
    args = parser.parse_args()

    # include `/outputs` in the experiment path
    args.experiment_folders_names = args.experiment_path
    args.experiment_path = os.path.join('outputs', args.experiment_folders_names)

    # create necessary paths
    args.export_dir = os.path.join(args.experiment_path, 'results')
    args.config_path = os.path.join(args.experiment_path, '.hydra')

    # read the config file information
    with initialize(config_path=args.config_path):
        cfg = compose(config_name="config")

    try:
        cfg['base']['export_dir'] = f'outputs/{args.experiment_folders_names}'

        n_images_per_study = cfg['dataset']['config']['n_images_per_study']
        concatenate_features_method = cfg['dataset']['config']['concatenate_features_method']
        cfg['base']['features_export_dir'] = f'outputs/dataset/reports_studies/{n_images_per_study}_{concatenate_features_method}'
        cfg['base']['results_export_dir'] = args.export_dir
        # cfg['base']['tensorboard_export_dir'] = f'outputs/{args.experiment_folders_names}'
        cfg['checkpoints']['checkpoints_export_dir'] = f'outputs/{args.experiment_folders_names}/checkpoints'        
        mmgconfig = AttrDict(cfg)

    except Exception as e:
        print("An error occurred:", e)

    # Image aug (transform)
    transform = transforms.Compose([transforms.ToTensor()])

    # model loading, setting to eval, and to device
    assert os.path.isdir(mmgconfig.checkpoints.checkpoints_export_dir), f"Couldn't find the checkpoint directory {mmgconfig.checkpoints.checkpoints_export_dir}."
    model_ckp_path = os.path.join(mmgconfig.checkpoints.checkpoints_export_dir, mmgconfig.checkpoints.checkpoints_file_name)
    model_ckp_file = torch.load(model_ckp_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mmgclip.model(config=mmgconfig).to(device)
    model.load_state_dict(model_ckp_file['model_state_dict'])
    model.eval()

    # prompt classifier
    clf   = mmgclip.PromptClassifier(model=model)

    logger.warning("Feature extraction runs on the CPU..")
    # load the image encoder and set to eval mode (feature extraction)
    feature_extractor  = torch.jit.load(mmgconfig.networks.image_encoder.convnext_tiny_clf_path)
    feature_extractor = feature_extractor #.to(device)
    feature_extractor.eval()

    # create path and validate it
    if args.image_id:
        # validate the format
        if not (len(args.image_id) == 13 and args.image_id[0] == 'p' and args.image_id[-2:] in ['cl', 'cr', 'ml', 'mr']): raise ValueError(f"Wrong Value passed to image_id: {args.image_id}.")
        
        # create a path
        path = mmgclip.create_path(args.image_id, base_dataset_path=mmgconfig.dataset.config.base_dataset_path)
        assert os.path.isfile(path), f"Either path doesn't exist or no image views found inside `{path}`."

        image_raw = Image.open(path)
        image_tensor = transform(image_raw) #.to(device)

        # Dicom images in 16bits, while pngs are 8 bits to save space
        image_tensor = 65535 * image_tensor
        image_tensor = image_tensor.unsqueeze(0)

        # Apply Forward pass in stages
        image_tensor_norm = (image_tensor - 32767.5) / 32767.5
        feature_map = feature_extractor.features((image_tensor_norm))
        image_embeddings = feature_extractor.avgpool(feature_map).squeeze()

    elif args.exam_id:
        if not len(args.exam_id) == 10: raise ValueError(f"Wrong Value passed to exam_id {args.exam_id}.")
        
        # create a path
        path = mmgclip.create_exam_path(args.exam_id, base_dataset_path=mmgconfig.dataset.config.base_dataset_path)
        assert os.path.isdir(path) and (len(os.listdir(path)) > 0), f"Either path doesn't exist or no exam found inside `{path}`."

        # extract the features, has to follow the model config
        try:
            # will hold all features extracted from each view inside a study
            study_views_feature_vector = []

            # iterate over study views
            for _, study_view_filename in enumerate(os.listdir(path)):
                # create a path for the image view
                study_view_filepath = os.path.join(path, study_view_filename) 
                
                image_raw = Image.open(study_view_filepath)
                image_tensor = transform(image_raw) #.to(device)

                # Dicom images in 16bits, while pngs are 8 bits to save space
                image_tensor = 65535 * image_tensor
                image_tensor = image_tensor.unsqueeze(0)

                # Apply Forward pass in stages
                image_tensor_norm = (image_tensor - 32767.5) / 32767.5
                feature_map = feature_extractor.features((image_tensor_norm))
                features = feature_extractor.avgpool(feature_map)

                study_views_feature_vector.append(features.squeeze()) # each features vector has shape of [768]

            # perform the feature vector concatenation method
            if mmgconfig.dataset.config.concatenate_features_method == "maxpool":
                # stack them on the first axis
                stacked_embeddings = torch.stack(study_views_feature_vector, dim= 0)    # [n_files, 768]
                
                # Apply max pooling along the batch dimension
                image_embeddings, _ = torch.max(stacked_embeddings, dim=0)              # [768]

            elif mmgconfig.dataset.config.concatenate_features_method == "concat":
                # NOTE: DON'T USE, not fully implemented
                # NOTE: here the config.networks.image_features_dimension has to be changed, thus we can't
                # use this approach unless we concat [0, 0, 0, .., 0] and make all embedding vectors has 
                # same shape
                image_embeddings = torch.cat(study_views_feature_vector, dim=0)         # [n_files * 768]

            elif mmgconfig.dataset.config.concatenate_features_method == "stack":
                image_embeddings = torch.stack(study_views_feature_vector, dim=0)       # [n_files, 768]

            elif mmgconfig.dataset.config.concatenate_features_method == "avgpool":
                stacked_embeddings = torch.stack(study_views_feature_vector, dim= 0)    # [n_files, 768]
                image_embeddings = torch.mean(stacked_embeddings, dim=0)                # [768]
            
            else:
                raise ValueError("Not implemented feature vector concatenation method")
                        
            # detach the features from cuda device to allow pin_memory=True when other objects are not on cuda and avoid torch multiprocessing 'spawn'
            image_embeddings = image_embeddings.detach().cpu()
            
        except Exception as e:
            failed_txt_filepath = os.path.join(args.experiment_path, 'failed_inference.txt')
            with open(failed_txt_filepath, "a") as myfile:
                logger.error(str(e))
                myfile.write(path + '\n' + str(e) + '\n\n')
    
    # stack to correct the dimension for the network
    image_embeddings = torch.stack([image_embeddings], dim=0)

    # # Report procedure (create pseudocode)
    # # 1. Obtain mass or calc or no findings
    # # 2. if mass:
    #     # 2.1 mass shapes
    #     # 2.2 mass margins
    #     # 2.3 mass density
    #     # 2.4 malignancy
    #     # 2.5 BIRADS score (based on malignancy and score list range)
    # # 3. if calc
    #     # 3.1 calc density
    #     # 3.2 calc morphology
    #     # 3.3 malignancy
    #     # 3.4 BIRADS score (based on malignancy and score list range)
    # # 4. architectural distortion
    # # 5. 

    report = {
        "mass_type": None,
        "mass_malignancy": None,
        "mass_shape": None,
        "mass_margin": None,

        "calc_malignancy": None,
        "calc_distribution": None,

        "arch_distortion": None,
        "birads": None,

        # we store the reports separately our of the prompt classifier wrapper, and construct report sentences using the generator and random sampling
        # the reports are saved in the keys below
        "no_findings_report": None,
        "mass_report": None,
        "calc_report": None,
        "arch_dist_report": None,
    }    

    # Compare between mass, calcification, or no findings
    # TODO: assign multi-label here with a threshold instead of argmax, as it could contain mass and calc,
    #       that is better than adding a prompt for both classes togather "Mammogram revealed mass and calcifications."
    clf_mass_type = clf(
        image_features = image_embeddings,
        class_list = ["Mammogram revealed a mass.", 
                      "Mammogram revealed calcifications.",
                      "No findings are present."], 
        visualize = False
    )
    report['mass_type'] = clf_mass_type['class_list'][int(clf_mass_type['similarities_argmax'])]

    # if no findings, 
    # The breasts are symmetric and no masses, architectural distortion or suspicious calcifications are present.
    if int(clf_mass_type['similarities_argmax']) in [2]:
        report['arch_distortion'] = 'Mammography showed no evidence of architectural distortion.'
        report['birads'] = 'BI-RADS score 1.'

        report['no_findings_report'] = clf_mass_type['class_list'][int(clf_mass_type['similarities_argmax'])] + ' ' + report['arch_distortion'] + ' ' + report['birads']

    else:
        # if there are findings, we distinguish between them
        # Mass findings
        if int(clf_mass_type['similarities_argmax']) in [0]: # 0 here is the indice for the two prompts of the mass type, mainly on mass
            # mass malignancy
            clf_mass_malignancy = clf(
                image_features = image_embeddings,
                class_list = ["Mass suggestive of benign pathology.", 
                            "Mass suggestive of malignant pathology."],
                visualize = False
            )
            report['mass_malignancy'] = clf_mass_malignancy['class_list'][int(clf_mass_malignancy['similarities_argmax'])]

            # mass shape
            clf_mass_shape = clf(
                image_features = image_embeddings,
                class_list = ["Mass shape is oval.", 
                            "Mass shape is round.", 
                            "Mass shape is irregular."],
                visualize = False
            )
            report['mass_shape'] = clf_mass_shape['class_list'][int(clf_mass_shape['similarities_argmax'])]

            # mass margin
            clf_mass_margin = clf(
                image_features = image_embeddings,
                class_list = ["Mass margin is circumscribed.", 
                            "Mass margin is obscured.", 
                            "Mass margin is spiculated.", 
                            "Mass margin is ill defined."],
                visualize = False
            )
            report['mass_margin'] = clf_mass_margin['class_list'][int(clf_mass_margin['similarities_argmax'])]

            # for BI-RADS score, we can make the prediction more accurate based on the malignancy findings and
            # prior knowledge to the problem

            # Benign mass from BI-RADS template can fall under BI-RADS 0 (as additional assesment is needed), 
            # BI-RADS 2, or BI-RADS 3 (Probably Benign Finding) 
            if int(clf_mass_malignancy['similarities_argmax']) == 0:
                clf_birads = clf(
                    image_features = image_embeddings,
                    class_list = ["BIRADS score of 0.", 
                                "BIRADS score of 2.",
                                "BIRADS score of 3."],
                    visualize = False
                )
            else:
                # Malignancy can be BI-RADS 0, BI-RADS 4, 5, 6
                clf_birads = clf(
                    image_features = image_embeddings,
                    class_list = ["BIRADS score of 0.", 
                                "BIRADS score of 4.",
                                "BIRADS score of 5.",
                                "BIRADS score of 6."],
                    visualize = False
                )

            report['birads'] = clf_birads['class_list'][int(clf_birads['similarities_argmax'])]

            # report construction
            M_MALIG_match = re.search(r"\b(benign|malignant)\b", report['mass_malignancy'], re.IGNORECASE)
            M_MARG_match = re.search(r"\b(circumscribed|obscured|spiculated|ill defined)\b", report['mass_margin'], re.IGNORECASE)
            M_SHAPE_match = re.search(r"\b(oval|round|irregular)\b", report['mass_shape'], re.IGNORECASE)
            B_SCORE_match = re.search(r"\b(0|1|2|3|4|5|6)\b", report['birads'], re.IGNORECASE)

            report['mass_report'] = mmgclip.generate_gtr_prompt_sentence(
                key='gtr_mass:True', n=1, M_MALIG=M_MALIG_match.group() if M_MALIG_match else 'unknown', 
                M_MARG=M_MARG_match.group() if M_MARG_match else 'unknown', 
                M_SHAPE=M_SHAPE_match.group() if M_SHAPE_match else 'unknown')

            report['mass_report'] = report['mass_report'][:-1] + ", " \
                + mmgclip.generate_gtr_prompt_sentence(key="row.labels['birads']:True", n=1, B_SCORE=B_SCORE_match.group() if B_SCORE_match else 'unknown')

        # Calcification findings
        if int(clf_mass_type['similarities_argmax']) in [1]: # 1 here is the indice for the two prompts of the mass type, mainly on calcifications
            # calc malignancy
            clf_calc_malignancy = clf(
                image_features = image_embeddings,
                class_list = ["Calcifications suggestive of benign pathology.", 
                            "Calcifications suggestive of malignant pathology."],
                visualize = False
            )
            report['calc_malignancy'] = clf_calc_malignancy['class_list'][int(clf_calc_malignancy['similarities_argmax'])]

            clf_calc_distribution = clf(
                image_features = image_embeddings,
                class_list = ["Mammogram revealed calcifications with diffuse distribution.", 
                            "Mammogram revealed calcifications with regional distribution.",
                            "Mammogram revealed calcifications with grouped distribution.",
                            "Mammogram revealed calcifications with linear distribution.",
                            "Mammogram revealed calcifications with segmental distribution."],
                visualize = False
            )
            report['calc_distribution'] = clf_calc_distribution['class_list'][int(clf_mass_type['similarities_argmax'])]

            # for BI-RADS score, we can make the prediction more accurate based on the malignancy findings and
            # prior knowledge to the problem

            # Benign mass from BI-RADS template can fall under BI-RADS 0 (as additional assesment is needed), 
            # BI-RADS 2, or BI-RADS 3 (Probably Benign Finding) 
            if int(clf_calc_malignancy['similarities_argmax']) == 0:
                clf_birads = clf(
                    image_features = image_embeddings,
                    class_list = ["BIRADS score of 0.", 
                                "BIRADS score of 2.",
                                "BIRADS score of 3."],
                    visualize = False
                )
            else:
                # Malignancy can be BI-RADS 0, BI-RADS 4, 5, 6
                clf_birads = clf(
                    image_features = image_embeddings,
                    class_list = ["BIRADS score of 0.", 
                                "BIRADS score of 4.",
                                "BIRADS score of 5.",
                                "BIRADS score of 6."],
                    visualize = False
                )

            report['birads'] = clf_birads['class_list'][int(clf_birads['similarities_argmax'])]


            # report construction
            C_MALIG_match = re.search(r"\b(benign|malignant)\b", report['calc_malignancy'], re.IGNORECASE)
            C_DIST_match = re.search(r"\b(diffuse|regional|grouped|linear|segmental)\b", report['calc_distribution'], re.IGNORECASE)
            B_SCORE_match = re.search(r"\b(0|1|2|3|4|5|6)\b", report['birads'], re.IGNORECASE)

            report['calc_report'] = mmgclip.generate_gtr_prompt_sentence(key='gtr_calc:True', n=1, 
                                                                         C_MALIG=C_MALIG_match.group() if C_MALIG_match else 'unknown', 
                                                                         C_DIST=C_DIST_match.group() if C_DIST_match else 'unknown')
            
            report['calc_report'] = report['calc_report'][:-1] + ", " \
                + mmgclip.generate_gtr_prompt_sentence(key="row.labels['birads']:True", n=1, B_SCORE=B_SCORE_match.group() if B_SCORE_match else 'unknown')
                        
        # arch distortion score
        clf_arch_distortion = clf(
            image_features = image_embeddings,
            class_list = ["Mammogram displayed architectural distortion.",
                        "Mammography showed no evidence of architectural distortion."],
            visualize = False
        )
        report['arch_distortion'] = clf_arch_distortion['class_list'][int(clf_arch_distortion['similarities_argmax'])]

        # report construction
        report['arch_dist_report'] = mmgclip.generate_gtr_prompt_sentence(
            key=f'gtr_is_architectural_distortion:True' if int(clf_arch_distortion['similarities_argmax']) == 1 else 'gtr_is_architectural_distortion:False', n=1)

    report_keys = ['no_findings_report', 'mass_report', 'calc_report', 'arch_dist_report']
    report_text = ' '.join(report[key] for key in report_keys if report[key] is not None)
    
    print('Generated Report: ', report_text)
         