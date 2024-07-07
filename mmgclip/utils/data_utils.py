import os
import pandas as pd
from pathlib import Path
import json
from fuzzywuzzy import process
from pathlib import Path
import re
import time
from .global_utils import create_directory_if_not_exists
import difflib
import nltk
nltk.download('punkt')

def get_project_root() -> Path:
    '''Returns the project root directory.'''
    return Path(__file__).parent.parent


def find_similar_item(search_text, items):
    """
    Find the item in the list that is most similar to the search text.

    Parameters:
        search_text (str): The text to search for.
        items (list of str): The list of strings to search through.

    Returns:
        str: The item from the list that is most similar to the search text.
    """
    return process.extractOne(search_text, items)[0]

def create_path(id, base_dataset_path):
    '''A callback function to create a string path to the image with the specific id passed.
    It can handle receiving a view id or a patient id.

    Args:
        id: a string id for either the patient or the image view.

    Returns: 
        A list of string paths for the given id. 
    
    '''    
    if isinstance(id, str) and id[0][0] == 'p':
        # check the first id if it has letter 'p' indicating a specific study view.
        path = os.path.join(base_dataset_path, id[1:3], id[1:9], f"st{id[9:11]}", f"{id}.png")
        return path
    
    elif isinstance(id, str) and len(id)== 8:
        # this indicates a patient id
        studies_path = os.path.join(base_dataset_path, id[0:2], id)
        views_paths = []

        for study in os.listdir(studies_path):
            study_path = os.path.join(studies_path, study)

            for view in os.listdir(study_path):
                view_path = os.path.join(study_path, view)
                views_paths.append(view_path)

        return views_paths 

def create_exam_path(id, base_dataset_path):
    '''A callback function to create a string path to the exam with the specific id passed.
    It can handle receiving a view id or a patient id.

    Args:
        id: a string id for the exam.

    Returns: 
        A string representing the path to the exam folder. 
    '''    
    if isinstance(id, str):
        return os.path.join(base_dataset_path, id[0:2], id[0:8], f"st{id[8:10]}")

# def create_dataset_list(path_list):
#     '''
#     This function is mean to create a dataframe containing details about the .txt list dataset. 
#     It receives the folder directory path, and returns a dataframe with as structured below.

#     This function follows the implementation of `create_dataset_df`, even though there are no annotations for 
#     `mass_margin` and `mass_shape`, it returns those values.

#     Args:
#         path_list (list): list of paths to the dataset folder directory.

#     Returns:
#         dataset (pd.dataframe)

#     ```
#     >> dataset
#     >>           image_id  image_label                     mass_margin                  mass_shape       image_path     uncertain
#     >> 0     p02********mr          1                            [-1]                        [-1]  /***/***.png...           True
#     >> 1     p02********mr          1        [-1, -1, -1, -1, -1, -1]    [-1, -1, -1, -1, -1, -1]  /***/***.png...           True
#     >> 2     p02********ml          1                [Spiculated, -1]             [Irregular, -1]  /***/***.png...          False```   

#     The `image_label` could be 0 for benign, or 1 for malignant.
#     The `image_path` contains the full path of the image data file stored on the server. 

#     Note that some values are stored with a value of -1, which means it was missing in the json file.

#     '''
#     # placeholder for iteration
#     df_row_list = []

#     for file_path in path_list:
#         txt_dataset = pd.read_csv(file_path, sep=" ", dtype=str)
        
#         for index, row in txt_dataset.iterrows():
#             view_path_list = create_path(row[txt_dataset.columns.tolist()[0]])

#             # make it as a list if it is not the case, when the id is a string
#             if not isinstance(view_path_list, list): view_path_list = [view_path_list]

#             for image_view_path in view_path_list:
#                 image_id = image_view_path.split('/')[-1].replace('.png', '')
                
#                 view_info = [
#                     image_id, 
#                     0 if 'normal'in file_path else 1,
#                     [-1],
#                     [-1],
#                     image_view_path,
#                     True if ('malignant' in file_path or 'stl' in file_path) and ("cr" in image_id or "mr" in image_id) else False
#                 ]
#                 df_row_list.append(view_info)

#         # create dataset 
#     dataset = pd.DataFrame(df_row_list, 
#                            columns=['image_id', 'image_label', 'mass_margin', 'mass_shape', 'image_path', 'uncertain'])

#     return dataset

def validate_file_type(self, filepath, filetype = '.pth'):
    '''
    Validates if the filepath ends with a specific file format, given as filetype.

    Args:
        filepath (str): a string representing the filepath.
        filetype (str): a string representing the filetype to be validated with.

    Returns:
        boolean based on the condition.
    '''
    return filepath.lower().endswith(filetype)

def create_dataset_path(path):
    '''
    Creates a dataset dataframe containing both `image_id` and `image_path` for the given folder directory.

    Args:
        path (str): path to the folder to iterate inside

    Returns:
        dataset (pd.DataFrame): a dataframe containing both `image_id` and `image_path` files.
    '''
    dataset_list = []
    for root, dirs, files in os.walk(path):
        for patient_dir in dirs:
            patient_dirpath = os.path.join(root, patient_dir)

            for patient_path, study_folders, study_files in os.walk(patient_dirpath):
                
                for study_path in study_folders:
                    studypath = os.path.join(patient_path, study_path)

                    for file in os.listdir(studypath):
                        filepath = os.path.join(studypath, file)
                                            
                        if validate_file_type(filepath, '.pth'):
                            image_id = os.path.basename(filepath).replace('.pth', '')
                            dataset_list.append([image_id, filepath])
                                                
    return pd.DataFrame(dataset_list, columns=['image_id', 'image_path'])

def create_dataset_df(config):
    '''
    This function is mean to create a dataframe containing details about the JSON annotated dataset. 
    It receives the folder directory path, and returns a dataframe with as structured below.

    Args:
        config (object): an object that holds the experiment variables, including the dataset folder directory.

    Returns:
        dataset (pd.dataframe)

    ```
    >> dataset
    >>           image_id  image_label                     mass_margin                  mass_shape    has_mass        image_path
    >> 0     p02********mr          1                            [-1]                        [-1]            0   /***/***.png...
    >> 1     p02********mr          1        [-1, -1, -1, -1, -1, -1]    [-1, -1, -1, -1, -1, -1]            0   /***/***.png...
    >> 2     p02********ml          1                [Spiculated, -1]             [Irregular, -1]            1   /***/***.png...```

    The `image_label` could be 0 for benign, or 1 for malignant.
    The `image_path` contains the full path of the image data file stored on the server. 

    Note that some values are stored with a value of -1, which means it was missing in the json file.
    '''
    # placeholder for iteration
    df_row_list = []

    dataset_path = config.dataset.config.annotated_dataset_path
    list_dataset_path = config.dataset.config.lists_dataset_path
    list_dataset_files = [f for f in os.listdir(list_dataset_path) if f.endswith('.txt')]

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder) # ../data/02_data_T_regions/02_stl

        # label validation file
        list_dataset_file = find_similar_item("normal", list_dataset_files) if 'benign' in folder_path else find_similar_item("malignant", list_dataset_files)
        list_dataset_filepath = os.path.join(list_dataset_path, list_dataset_file)
        list_dataset = pd.read_csv(list_dataset_filepath, sep=" ", dtype=str)
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            image_id = os.path.basename(file_path).replace('.json', '')
            patient_id = image_id[1:9]

            # validating that the file exists in the patient labelled dataset
            if patient_id in list_dataset['patient_id'].values:
                
                # View information placeholder to append into the dataframe
                # we initialize it with the first 2 columns representing the image view
                view_info = [image_id]

                with open(file_path) as f:
                    d = json.load(f)

                    # get the regions object, could contain more than one entry
                    mass_regions_dict = d[f'{image_id}_png']['regions']

                    if len(mass_regions_dict) > 0:
                        # could have more than one regions, we store them in a single image row
                        region_mass_margins = []
                        region_mass_shapes = []
                        region_is_malig = []
                        region_has_mass = []
                        region_has_arch_distortion = []
                        region_has_calc = []

                        # iterate over all regions available
                        for _, region in mass_regions_dict.items():
                            region_properties = region['properties']

                            region_has_mass.append(region['is_mass'])
                            region_is_malig.append(region['is_malign'])
                            region_has_arch_distortion.append(region['is_architectural_distortion'])
                            region_has_calc.append(region['is_calcification_cluster'] or region['is_individual_calcification'])
                            
                            region_mass_margins.append(region_properties["mass_margin"] if "mass_margin" in region_properties.keys() else -1)
                            region_mass_shapes.append(region_properties["mass_shape"] if "mass_shape" in region_properties.keys() else -1)

                        is_malign = any(region_is_malig)
                        has_mass  = any(region_has_mass)
                        has_architectural_distortion = any(region_has_arch_distortion)
                        has_calc = any(region_has_calc)

                        # assign image_label 0 to benign views
                        # assign image_label 1 to malignant views with at least one is_malig
                        # assign image_label 2 to malignant views with no is_malig Flag set to True
                        if 'benign' in folder_path:
                            view_info.append(0)
                        else:
                            view_info.append(1 if is_malign else 2)

                        view_info.extend([region_mass_margins, region_mass_shapes, has_mass, has_architectural_distortion, has_calc])
                            
                    else:
                        # assign image_label 0 to benign views
                        # assign image_label 1 to malignant views with at least one is_malig
                        # assign image_label 2 to malignant views with no is_malig Flag set to True
                        if 'benign' in folder_path:
                            view_info.append(0)
                        else:
                            view_info.append(1 if is_malign else 2)

                        view_info.extend([[-1], [-1], False, False, False])

                # structure the image view path: p/patient[8]/study[2]/view[2]
                # view_path = os.path.join(base_dataset_path, image_id[1:3], image_id[1:9], f"st{image_id[9:11]}", f"{image_id}.png")
                view_path = create_path(image_id, base_dataset_path=config.dataset.config.base_dataset_path)

                view_info.append(view_path)

                if os.path.isfile(view_path):
                    df_row_list.append(view_info)
            else:
                # print(f"File Not found: {image_id}")
                pass

    # create dataset 
    dataset = pd.DataFrame(df_row_list, 
                           columns=['image_id', 'image_label', 'mass_margin', 'mass_shape', 'has_mass', 'has_architectural_distortion', 'has_calc', 'image_path'])
    return dataset

def preprocess_reports_csv(df = None, config = None, export = False):
    '''
    This function pre-processes the raw dataset csv to be adequate with the code structure and sutable to
    be translated. It also handles the export of the pre-processed dataframe.

    Args:
        df (pd.DataFrame): a pandas dataframe for the translated file object.
        config (object): mmgclip config object.
        export (bool): a boolean flag to export the processed file.

    Returns:
        df (pd.DataFrame): the processed dataframe.
    '''
    def remove_text_before_word(sentence, word):
        index = sentence.find(word)
        if index != -1:  # Check if the word is found in the sentence
            return sentence[index + len(word) + 1:] # + 1 to remove the preceeding space
        else:
            return sentence  # If the word is not found, return the original sentence

    def extract_report(report):
        'This function selects specific reports sections, and removes reports headers.'
        # KEPT SECTIONS: Current Study, MR, MG
        pattern_remove_section = r'Report\s(?:US|OTUS|MROT|MGOT)\s\d{4}-\d{2}-\d{2}(?::\s##)?\s[\d.]+\s\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2}\s(?:READ|ARRIVED)?\s?Finalized\s(.+?)(?=##)'
        # pattern_remove_section = r'Report\s(?:US|MR|OTUS|MROT|MGOT)\s\d{4}-\d{2}-\d{2}(?::\s##)?\s[\d.]+\s\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2}\s(?:READ|ARRIVED)?\s?Finalized\s([\d.]+.*?)##'

        # in the following pattern, add all the report headers to be eliminated, main report text will be kept
        # default_pattern = r'Report\s(?:current\sstudy|US|MR|OTUS|MROT|MGOT|MG)\s\d{4}-\d{2}-\d{2}(?::\s##)?\s[\d.]+\s\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2}\s(?:READ\s)?Finalized\s'
        default_pattern = r'Report\s(.+?)Finalized\s'

        # remove unnecessary sections
        cleaned_report = re.sub(pattern_remove_section, '', report, flags=re.MULTILINE)

        # remove unnecessary headers from kept sections
        cleaned_report = re.sub(default_pattern, '', cleaned_report, flags=re.MULTILINE)

        return cleaned_report
    
    def find_sentences_with_keyword(text, keywords, return_str = False):
        # Split the text into individual sentences
        sentences = text.split('.')
        
        # List to store sentences containing the keyword
        matched_sentences = []

        for keyword in keywords:    
            # Iterate through each sentence
            for sentence in sentences:
                # Check if the keyword exists in the sentence
                if keyword.lower() in sentence.lower():
                    # If found, append the sentence to the list
                    if sentence not in matched_sentences:
                        matched_sentences.append(sentence)

        if return_str:
            if len(matched_sentences) == 0: 
                return "Unknown"
            
            return ' '.join(matched_sentences)
        
        if len(matched_sentences) == 0: 
            matched_sentences = ["Unknown"]  
        
        return matched_sentences
            
    def remove_extra_spaces(text):
        # Use regular expression to replace multiple spaces with a single space
        return re.sub(r'\s+', ' ', text)
    
    def dutch_number_to_integer(word):
        dutch_numbers = {
            "nul": 0,
            "een": 1,
            "twee": 2,
            "drie": 3,
            "vier": 4,
            "vijf": 5,
            "zes": 6
        }
        return dutch_numbers.get(word.lower(), word)

    def replace_dutch_numbers(sentence):
        words = sentence.replace('.', '').split() # cases like BI-RADS twee. (ends with dot)
        replaced_sentence = []
        for word in words:
            replaced_sentence.append(str(dutch_number_to_integer(word)))
        return ' '.join(replaced_sentence)
    
    def roman_to_int(text):
        '''Some of the BIRAD values were reported in roman numerals, this method casts them to an integer.
        
        Args:
            text (str): The text to search into.

        Returns:
            replaced_string (str): The same text, replacing the roman values to integers.
        '''
        values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        
        def replace_roman_with_int(match):
            # Convert the Roman numeral to integer
            roman_numeral = match.group(0)
            result = 0
            prev_value = 0  
            for char in roman_numeral:
                value = values[char.upper()]
                if value > prev_value:
                    result += value - 2 * prev_value
                else:
                    result += value
                prev_value = value
            return str(result)
        
        # Replace Roman numerals with their integer values
        replaced_string = re.sub(r'\b([IVXLCDM]+)\b', replace_roman_with_int, text) #  \b([IVXLCDM]+)(?![\s])
        
        return replaced_string
    
    def find_similar_words(paragraph:str, keywords:list = [], threshold:float=0.7):
        tokens = paragraph.split()  # Tokenize the paragraph
        similar_words = []

        special_keywords_casting = {
            # special mass shapes wording
            "ovaalvormige": "ovaal",

            # special calcifications distribution wording
            "diffuus verspreid": "verspreid",

            # not always the case that the two words are mentioned togther in the report in morphology
            "grof heterogeen": "heterogeen",
            "heterogene": "heterogeen",
            "fine pleomorphic": "pleomorphic",
        }
        
        for keyword in keywords:
            for token in tokens:
                similarity = difflib.SequenceMatcher(None, keyword, token).ratio()
                if similarity >= threshold:

                    if keyword in special_keywords_casting.keys():
                        keyword = special_keywords_casting[keyword]

                    return (token, similarity, keyword)

        return ("", "","unknown")
    
    def extract_labels(**kwargs):
        birad_match = re.search(r'\b(?:birads|bi[-\s]rads)[-a-zA-Z]*\b(?:\s+\w+)*?[-\s]*(\d+)(?:\s*([A-Z])\b)?', replace_dutch_numbers(kwargs['birads']), re.IGNORECASE)
        
        if birad_match:
            birads_number = birad_match.group(1)

            # to add the letters following the numbers as a separate category
            # if birad_match.group(2):
            #     birads_number += birad_match.group(2).upper()
        else:
            birads_number = "unknown"

        labels = {}
        labels['birads'] = birads_number
        labels['malignancy'] = kwargs['malig']

        labels['masses'] = {}
        labels['masses']['shapes'] = find_similar_words(paragraph=kwargs['report'], keywords=["ovaal", "ovaalvormige", "irregulair", "rond"], threshold=0.7)[-1]
        labels['masses']['density'] = find_similar_words(paragraph=kwargs['report'], keywords=["hyperdens", "isodens", "hypodens", "lucent"], threshold=0.7)[-1]

        labels['calcifications'] = {}
        labels['calcifications']['distribution'] = find_similar_words(paragraph=kwargs['report'], keywords=["diffuus", "diffuus verspreid", "regionaal", "gegroepeerd", "lineair", "segmenteel"], threshold=0.65)[-1]
        labels['calcifications']['morphology'] = find_similar_words(paragraph=kwargs['report'], keywords=["amorf", "grof heterogeen", "heterogeen", "heterogene", "Fine pleomorphic", "pleomorphic", "fijn lineair", "lineair vertakkend"], threshold=0.65)[-1]

        return labels
    
    def translate_labels(labels):
        '''
        This function translates the obtained labels from the reports to match BIRADs wording. This is to avoid wrong 
        translation and to ensure correct matching from dutch to english 
        
        Args:
            labels (dict): labels to be translated from NL to EN.

        Returns:
            translated_labels (dict): same input dict with labels translated to BIRADS EN wording.
        '''
        def _replace_values(original, translation):
            for key, value in original.items():
                if isinstance(value, dict):
                    _replace_values(value, translation.get(key, {}))
                else:
                    translated_value = translation.get(key, {}).get(value.lower() if not isinstance(value, int) else value, value)
                    original[key] = translated_value

            return original

        translation = {
            "masses":{
                "shapes":{
                    "ovaal": "oval",
                    "rond": "round",
                    "irregulair": "irregular",
                    "unknown": "unknown"
                },
                "density":{
                    "hyperdens": "high density",
                    "isodens": "equal density",
                    "hypodens": "low density",
                    "lucent": "fat-containing",
                    "unknown": "unknown"
                }
            },
            "calcifications":{
                "distribution":{
                    "diffuus": "diffuse",
                    "verspreid": "diffuse",
                    "regionaal": "regional",
                    "gegroepeerd": "grouped",
                    "lineair": "linear",
                    "segmenteel": "segmental",
                    "unknown": "unknown"
                },
                "morphology": {
                    "amorf": "amorphous",
                    "heterogeen": "coarse heterogeneous",
                    "pleomorphic": "fine pleomorphic",
                    "fijn lineair": "fine linear",
                    "lineair vertakkend": "fine-linear branching",
                    "unknown": "unknown"
                }
            }
        }

        return _replace_values(labels, translation)
    
    def validate_report(row):
        # malignancy section
        if (row.malignancy_benign_section_nl.lower() != "unknown") and (row.malignancy_benign_section_nl.lower() not in row.report_preprocessed.lower()):
            row.report_preprocessed += f" {row.malignancy_benign_section_nl}"
        
        # conclusion section
        if (row.conclusion_nl.lower() != "unknown") and ((row.conclusion_nl.lower() not in row.report_preprocessed.lower()) and ("conclusie" not in row.report_preprocessed.lower())):
            row.report_preprocessed += f" {row.conclusion_nl}"

        # birads section
        if (row.birads_section_nl.lower() != "unknown") and (row.birads_section_nl.lower() not in row.report_preprocessed.lower()):
            row.report_preprocessed += f" {row.birads_section_nl}"

        # add a condition to check if still the row.report_preprocessed is an empty string, take the label from the 
        # malignancy column is_malig and add a prompt "proven {classname}" etc..
        # TODO: Keep this up to date with the labels
        if row.report_preprocessed == "":
            row.report_preprocessed = None
            # malig_label = "malignant" if row.labels['malignancy'] == 1 else "benign"
            # row.report_preprocessed += f"Potential to be {malig_label} pathology."

        return row

    _keywords_to_remove = [
        'Medische gegevens:', # Medical data:
        ' Medische gegevens:', # Medical data:
        'Medische gegevens:   ',
        '-------------------------------------------------Addendum   start---------------------------------------------',
        '-------------------------------------------------Addendum   einde----------------------------------------',
        '   -------------------------------------------------Addendum   einde-------------------------------------------- ',
        '##',
        '## ##'
        'ADDENDUM', # ADDENDUM, appendix/add-on
        'ADDENDUM:',
        '----',
        '   /',
        'Addendum: ',
        'Addendum:   ',
        'ANON Klinische gegevens', # ANON Clinical data
        'HITGE-BOETESC Medische gegevens:', # HITGE-FINESC Medical data:
        'HITGE-BOETESC Medische gegevens:   ',  # HITGE-FINESC Medical data:
        'HITGE-BOETESC', # HITGE-BOETESC
        'MRW ENGELBRECHT', # MRW ENGELBRECHT
        'ANON Medische gegevens', # ANON Medical data
        'AARTS Medische gegevens Bij bevolkingsonderzoek afwijkingen links.', # AARTS Medical data Abnormalities on the left during population screening,
        'AARTS Medische gegevens PatiÃ«nt overgekomen uit Veghel.', # AARTS Medical data Patient transferred from Veghel.
        'AARTS Medische gegevens via bevolkingsonderzoek in verband met afwijking in de linkermamma.', # AARTS Medical data via population survey in connection with an abnormality in the left breast.
        'ANON ', # ANON
        'ANON Klinische gegevens ', # ANON Clinical data
        'BOKHOVEN VSC Medische gegevens. ', # BOKHOVEN VSC Medical data.
        'Medische gegevens', # Medical data
        'Medische gegevens.', # Medical data.,
        'Medische gegevens. ', # Medical data.,
        'WILLIAMSVAN Klinische informatie ', # WILLIAMSVAN Clinical information
        'WILLIAMSVAN Klinische ', # WILLIAMSVAN Clinical,
        'WILLIAMSVAN Medische gegevens ', # WILLIAMSVAN Medical data
        'WILLIAMSVAN ', # WILLIAMSVAN 
        'IMHOF-TASMW ', # IMHOF-TASMW
        'MUSRDM ', # MUSRDM ,
        'VELTMANJ ', # VELTMANJ
        'MEIJERFJA ', # MEIJERFJA
        'HITGE-BOETESC ',
        'JAFARIK ', # JAFARIK
        "This is a summary report. The complete report is available in the patient''s medical record. If you cannot access the medical record, please contact the sending organization for a detailed fax or copy. ",
        'FÃTTERERJJ', # FÃTTERERJJ
        "PLOEGMAKERSM ",
        "FÜTTERERJJ Medische gegevens:",
        "FÃTTERERJJ",
        "DIE VCE",
        "false false Digital ",
        "IMHOF-TASMW",
        "Specimen   opnamen ten behoeve van pathologie.", # Specimen recordings for pathology
        "Specimen opnamen ten behoeve van pathologie.", # Specimen recordings for pathology
        "DIJK VANR",
        "IMHOF-TASMW",
        "Specimen opnamen ten behoeve van pathologie.", # specimin recording for pathology,
        "Specimen opname ten behoeve van   pathologie",
        "Addendum start",
        "-Addendum start-",

        "STOUTJESDIJKMJ",
        "SPAARGARENGJ",

        "Specimen opnamen ten behoeve van de PA.",
        "Specimen opnamen ten behoeve van de   PA.",
        "Specimen opnamen ten behoeve van pathologie"
        "Specimen opnamen ten behoeve van   pathologie",
        '-- ',
        '--',
        ' -- ',
        'Controle.',
        'Familieanamnese negatief.',
        'FEUTHL',
        'FA /'
        
        # the next line causes some processed text to be an empty string as it is the only text after the report extraction
        # TODO: Make sure it is meaningful to remove it
        # "Het betreft hier externe beelden voor radiologisch of nucleaire geneeskundig onderzoek. Deze beelden zijn op aanvraag ingelezen in Impax. Er heeft geen revisie plaatsgevonden. Indien u alsnog een revisie wenst, dient u hiervoor een aanvraag in te dienen samen met het originele papieren verslag. Echografie is een dynamisch onderzoek en is derhalve voor herbeoordeling niet geschikt.",
        # "Specimen opnamen ten behoeve van de PA.", # a lot of reports were excluded before adding this line here with the same sentences, TODO: invistigate the impact of keeping them and adding text based on the labels
        ]

    _keywords_to_replace = [
        '   ',
        '  ',
        '    ',
        ' . ',
        ' .',
        '>>',
        '  >>  ',
        ': ',
    ]

    # remove eliminated reports lists, note that this list might include rows that were already dropped when reading
    # the original csv as bad lines
    eliminated_reports_path = config.dataset.config.eliminated_reports_path
    eliminated_reports_data = pd.read_csv(eliminated_reports_path, names=['raw_id', 'patient_id', 'report_date'], dtype = str)
    eliminated_reports_cols = ['raw_id', 'patient_id', 'report_date']

    # Get the common rows between df1 and df2 based on specific columns
    common_rows = df.merge(eliminated_reports_data, on=eliminated_reports_cols, how='inner')

    # Get the indices of the common rows in df1
    indices_to_drop = df.index[df.isin(common_rows.to_dict('list')).all(axis=1)]
    df = df.drop(indices_to_drop)

    # remove samples that dont have patient id
    df = df[df['patient_id'].notna()]

    # select only mg modality
    df = df[df['modality'] == "MG"]

    # drop dates columns
    df = df.drop(columns=['modality'])

    # validate if the patient id is in the list of ids for normal or malignancy, remove if not in either one of them
    normal_list = pd.read_csv(config.dataset.config.lists_dataset_path + '/normal_patients.txt', sep=" ", dtype=str)['patient_id'].values
    malignant_list = pd.read_csv(config.dataset.config.lists_dataset_path + '/malignant_patients.txt', sep=" ", dtype=str)['patient_id'].values

    df = df[df['patient_id'].isin(normal_list) | df['patient_id'].isin(malignant_list)]
    df.reset_index(drop=True, inplace=True)

    # replace "malignant" with label 1, and "NotMalignant" with label 0
    df['is_malig'] = df['is_malig'].apply(lambda x: 1 if x == "malignant" else 0)

    # in the pathology column, remove <st0> and </st0>
    df['pathology'] = df['pathology'].apply(lambda x: x.replace('<st0>', '').replace('</st0>', '').replace('<st>', '').replace('</st>', ''))
    
    # in the impression column, replace "*" with a space
    df['impression'] = df['impression'].apply(lambda x: x.replace('*', ' '))

    # replcae multiple spaces with a single space
    df['report_preprocessed'] = df['report'].apply(remove_extra_spaces)

    # remove starting template sentence of each of the reports, there could be several reports
    df['report_preprocessed'] = df['report_preprocessed'].str.replace('READFinali zed', ' READ Finalized ')
    df['report_preprocessed'] = df['report_preprocessed'].str.replace('READFinal ized', ' READ Finalized ')
    df['report_preprocessed'] = df['report_preprocessed'].str.replace('READFinaliz ed', ' READ Finalized ')
    df['report_preprocessed'] = df['report_preprocessed'].str.replace('ARRIVEDFi nalized', ' ARRIVED Finalized ')
    df['report_preprocessed'] = df['report_preprocessed'].str.replace('00Finalized', '00 Finalized ')
    df['report_preprocessed'] = df['report_preprocessed'].str.replace('00Finaliz   ed', '00 Finalized ')
    df['report_preprocessed'] = df['report_preprocessed'].str.replace('00F   inalized', '00 Finalized ')
    df['report_preprocessed'] = df['report_preprocessed'].str.replace('00F inalized', '00 Finalized ')
    df['report_preprocessed'] = df['report_preprocessed'].str.replace('00Finaliz ed', '00 Finalized ')
    
    df['report_preprocessed'] = df['report_preprocessed'].apply(extract_report) # using regex pattern
    df['report_preprocessed'] = df['report_preprocessed'].str.replace(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{7}\+\d{2}:\d{2}\s0', '', regex=True) # noticed this pattern in some reports
    
    df['report_preprocessed'] = df['report_preprocessed'].str.replace(r'(?:Zie ook\s)?T\d{2}-\d{3}\s?\(?\w*\)?', '', regex=True) # removes any raw id from the text
    df['pathology'] = df['pathology'].str.replace(r'(?:Zie ook\s)?T\d{2}-\d{3}\s?\(?\w*\)?', '', regex=True) # removes any raw id from the text
    
    # separate reports to sections
    df['report_preprocessed'] = df['report_preprocessed'].apply(lambda x: roman_to_int(x))

    # Steps
    # Find sentences with keywords first from the original report, it could be missed from the sub report type
    # roman to int if any
    # remove text before Finalized, given that it could select sentences with reports headers READ Finalized etc..
    df['malignancy_benign_section_nl'] = df.apply(lambda x: remove_text_before_word(find_sentences_with_keyword(text=x.report, keywords=['maligniteit', 'benigne'], return_str = False)[-1], "Finalized"), axis=1)
    df['birads_section_nl'] = df.apply(lambda x: remove_text_before_word(roman_to_int(find_sentences_with_keyword(text=x.report, keywords=['BI-RADS', 'BIRADS', 'BIRAD'], return_str = False)[-1]), "Finalized"), axis=1)
    df['conclusion_nl'] = df.apply(lambda x: remove_text_before_word(find_sentences_with_keyword(text=x.report, keywords=['Conclusie'], return_str = False)[-1], "Finalized"), axis=1)

    # remove those keywords in the list to filter
    # for keyword in _keywords_to_remove:
    #     df['report_preprocessed'] = df['report_preprocessed'].str.replace(keyword, '')
    #     df['pathology'] = df['pathology'].str.replace(keyword, '')
    #     df['impression'] = df['impression'].str.replace(keyword, '')

    #     df['malignancy_benign_section_nl'] = df['malignancy_benign_section_nl'].str.replace(keyword, '')
    #     df['birads_section_nl'] = df['birads_section_nl'].str.replace(keyword, '')
    #     df['conclusion_nl'] = df['conclusion_nl'].str.replace(keyword, '')

    # # replace those keywords in the list to replace
    # for keyword in _keywords_to_replace:
    #     df['report_preprocessed'] = df['report_preprocessed'].str.replace(keyword, ' ')
    #     df['pathology'] = df['pathology'].str.replace(keyword, ' ')
    #     df['impression'] = df['impression'].str.replace(keyword, ' ')

    #     df['malignancy_benign_section_nl'] = df['malignancy_benign_section_nl'].str.replace(keyword, ' ')
    #     df['birads_section_nl'] = df['birads_section_nl'].str.replace(keyword, ' ')
    #     df['conclusion_nl'] = df['conclusion_nl'].str.replace(keyword, ' ')

    # Define columns and operations
    columns_to_process = {
        'report_preprocessed': [_keywords_to_remove, _keywords_to_replace],
        'pathology': [_keywords_to_remove, _keywords_to_replace],
        'impression': [_keywords_to_remove, _keywords_to_replace],
        'malignancy_benign_section_nl': [_keywords_to_remove, _keywords_to_replace],
        'birads_section_nl': [_keywords_to_remove, _keywords_to_replace],
        'conclusion_nl': [_keywords_to_remove, _keywords_to_replace]
    }

    # Iterate over columns and perform operations
    for column, (keywords_to_remove, keywords_to_replace) in columns_to_process.items():
        for keyword in keywords_to_remove:
            df[column] = df[column].str.replace(keyword, '')
        for keyword in keywords_to_replace:
            df[column] = df[column].str.replace(keyword, ' ')

    # if the final sentence starts with a space, strip it
    df['report_preprocessed'] = df['report_preprocessed'].apply(lambda x: x.lstrip() if x.startswith(" ") else x)                   # sentences that starts with space, remove it
    df['report_preprocessed'] = df['report_preprocessed'].apply(lambda x: x.replace('. ', '', 1) if x.startswith(".") else x)       # remove '.' if the sentence starts with it.
    
    df['malignancy_benign_section_nl'] = df['malignancy_benign_section_nl'].apply(lambda x: x.lstrip() if x.startswith(" ") else x)
    df['birads_section_nl'] = df['birads_section_nl'].apply(lambda x: x.lstrip() if x.startswith(" ") else x)

    # labels
    df['labels'] = df.apply(lambda x: extract_labels(birads=x.birads_section_nl, malig=x.is_malig, report=x.report), axis=1)
    df['labels'] = df['labels'].apply(lambda x: translate_labels(x)) # translate labels from NL to EN, google translate is random and doesn't follow BIRADs report template, thus cast manually
    

    # ensure certain information are found in the report
    df = df.apply(lambda x: validate_report(x), axis=1)

    # ensure again there is no double spaces after all processing
    df['report_preprocessed'] = df['report_preprocessed'].apply(remove_extra_spaces)
    
    # the report text could have different types of reports
    df['has_report_current'] = df['report'].apply(lambda x: "Report current " in x)
    df['has_report_US'] = df['report'].apply(lambda x: "Report US " in x)
    df['has_report_MG'] = df['report'].apply(lambda x: "Report MG " in x)
    df['has_report_MR'] = df['report'].apply(lambda x: "Report MR " in x)
    df['has_report_others'] = df['report'].apply(lambda x: any([report_type in x for report_type in ["OTUS", "MROT", "MGOT"]])) # OTUS|MROT|MGOT

    df = df.sort_values(['patient_id', 'exam_date'], ascending=[True, True]).reset_index(drop=True)

    if export:
        export_dir = os.path.join(os.getcwd(), '../data', time.strftime("%Y-%m-%d/%H-%M-%S", time.gmtime()))
        filename_csv = os.path.join(export_dir, 'processed_reports.csv')
        filename_txt = os.path.join(export_dir, 'nl_reports_only.txt')

        create_directory_if_not_exists(export_dir)

        # export the entire df to a csv
        df.to_csv(filename_csv, encoding="latin1")

        # export the reports column to a txt file
        df['report'].to_csv(filename_txt, index=False, header=False, sep=' ', mode='a')

    return df

def remove_duplicate_sentences(text):
    '''Removes duplicate sentences from a string text.

    Args:
        text (str): text to search through.

    Returns:
        cleaned_text (str): original text withou any duplicates.
    
    '''
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Remove duplicates while preserving the order
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        if sentence not in seen_sentences:
            unique_sentences.append(sentence)
            seen_sentences.add(sentence)

    # Join the unique sentences back into a single string
    cleaned_text = ' '.join(unique_sentences)
    return cleaned_text


def post_process_translated_report(df= None, config = None, export = False, export_dir = None):
    '''
    This function post-processes the translated csv to be adequate with the code structure and sutable to
    be used as a dataset for study-report task. It also handles the export of the processed dataframe.

    Args:
        df (pd.DataFrame): a pandas dataframe for the translated file object.
        config (object): mmgclip config object.
        export (bool): a boolean flag to export the processed file.
        export_dir (str): the folder path to which the processed dataframe will be exported.

    Returns:
        df (pd.DataFrame): the processed dataframe.
    '''
    def create_study_path(patient_id, study_id):
        return os.path.join(config.dataset.config.base_dataset_path, patient_id[0:2], patient_id, study_id) 
    
    # rename the last column to `image_description`
    last_col = df.columns[-1]                           # translated processed reports 
    second_last_col = df.columns[-2]                    # impression section
    df = df.rename(columns={second_last_col: "image_impression",last_col:'image_description'}, inplace=False)
    
    # zero-fill the patient id column to have 8 digits as string, the leading 0 is eliminated once it
    # is uploaded for translation
    df['patient_id'] = df['patient_id'].apply(lambda x: '{0:0>8}'.format(x))

    # create the path for the study
    df['study_path'] = df.apply(lambda x: create_study_path(x.patient_id, x.study_id), axis=1)

    # Remove characters that are not part of the ASCII character set
    df['image_description'] = df['image_description'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))

    # remove any duplicate sentences to reduce the input length
    df['image_description'] = df['image_description'].apply(lambda x: remove_duplicate_sentences(x))

    # remove any row without a desciption, as a result from the translation function returning '#VALUE!'
    df = df[df['image_description'] != '#VALUE!']

    # to make the validation consistent, cast special keywords
    df['image_description'] = df['image_description'].apply(lambda x: x.replace('malignancy', 'malignant'))
    df['image_description'] = df['image_description'].apply(lambda x: x.replace('BI-RADS', 'BIRADS'))
    df['image_description'] = df['image_description'].apply(
        lambda x: x.replace(':', ' ')
                    .replace(',', ' ')
                    .replace('-', ' ')
                    .replace('""','')
                    .replace('...', ''))

    if export:
        export_dir = os.path.join(os.getcwd(), export_dir)
        filename_csv = os.path.join(export_dir, 'postprocessed_tr_dataset.csv')
        reports_filename_txt = os.path.join(export_dir, 'en_reports_only.txt')
        impression_filename_txt = os.path.join(export_dir, 'en_impression_only.txt')

        create_directory_if_not_exists(export_dir)

        # export the entire df to a csv
        df.to_csv(filename_csv) # encoding="latin1"

        # export the reports column to a txt file
        df['image_description'].to_csv(reports_filename_txt, index=False, header=False, sep=' ', mode='a')
        df['image_impression'].to_csv(impression_filename_txt, index=False, header=False, sep=' ', mode='a')

    return df

def map_path_to_features(df, config, export = False, export_dir = None):
    '''After extracting the features of the studies, we need to update the postprocessed translated file to point 
    to the extracted features path and not to the storage.
    
    Args:
        df (pd.DataFrame): a pandas dataframe for the translated file object.
        config (object): mmgclip config object.
    
    Returns:
        df (pd.DataFrame): same dataset preprocessed.
    '''

    def _get_patient_id(path):
        match =  re.search(r'\d{8}', path)

        if match:
            return match.group()
    
    def _modify_study_path(study_path):
        return os.path.join(config.base.features_export_dir, 
                            study_path.split('2D_100micron/')[-1], 
                            '{}.pth'.format(_get_patient_id(study_path))) 

    # modify the study path to point to the extracted directory
    df['study_path'] = df.apply(lambda x: _modify_study_path(x.study_path), axis=1)

    # drop if the file of the study path doesn't exist in the extracted feature
    df = df[df['study_path'].apply(lambda x: os.path.isfile(x))]
    df.reset_index(drop=True, inplace=True)

    if export:
        export_dir = os.path.join(os.getcwd(), export_dir)
        filename_csv = os.path.join(export_dir, 'final_reports_dataset.csv')

        create_directory_if_not_exists(export_dir)

        # export the entire df to a csv
        df.to_csv(filename_csv, encoding="latin1")

    return df

def process_class_list(class_list:list):
    '''
    Pre-processing function mainly useful to match the training labels with the inference labels.
    This is in the case that there is a mis-match between the labels trained with those in the enums
    file.

    Args:
        class_list (list): a list of labels / classes that will be fed to the text encoder.

    Returns:
        class_list (list): the same list passed, but processed. 
    '''
    if not isinstance(class_list, list):
        raise ValueError("`class_list` has to be a list of classes.")
    
    # # remove undefined mainly in sentence training
    # if 'undefined' in class_list:
    #     class_list = [item for item in class_list if item != 'undefined']

    # For mass margins
    class_list = ['ill defined' if x == 'illdefined' else x for x in class_list] if 'illdefined' in class_list else class_list

    # for mass
    class_list = ['no mass' if x == 'nomass' else x for x in class_list] if 'nomass' in class_list else class_list

    # for calc
    class_list = ['non-calcified' if x == 'noncalcified' else x for x in class_list] if 'noncalcified' in class_list else class_list
    class_list = ['has calcification' if x == 'hascalcification' else x for x in class_list] if 'hascalcification' in class_list else class_list


    class_list = ['no architectural distortion' if x == 'noarchitecturaldistortion' else x for x in class_list] if 'noarchitecturaldistortion' in class_list else class_list
    class_list = ['displayed architectural distortion' if x == 'displayedarchitecturaldistortion' else x for x in class_list] if 'displayedarchitecturaldistortion' in class_list else class_list

    # For mass shape
    # lobular is merged in the image-label training to the oval
    # if 'lobular' in class_list:
    #     class_list = [item for item in class_list if item != 'lobular']

    # Make them all as capital letters
    # class_list = [l.capitalize() for l in class_list]

    return class_list



# if __name__ == '__main__':
#     # create_dataset_df('')
#     dataset = create_dataset_df(os.path.join(os.getcwd(), '../../data/02_data_T_regions'))
#     print(dataset)
