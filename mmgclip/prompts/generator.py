from random import sample
import pandas as pd
import random

BENIGN_MALIG = {
    "benign":[
        "This mammogram is benign.",
        "This is a benign mammogram.",
        "Benign.",
        "This is a benign finding.",
        "Probably benign.",
        "Benign appearance.",
        "Benign evaluation.",
        "There are evidence of benign."
    ],
    "malignant": [
        "This mammogram is malignant.",
        "This is a malignant mammogram.",
        "Malignant.",
        "This is a malignant finding.",
        "Probably malignant.",
        "Malignant appearance.",
        "Malignant evaluation.",
        "There are evidence of malignancy."
    ]
}

HAS_MASS = {
    "positive": [
        "{E} is {R}.",
        "The presence of {E} is {R}.",
        "findings are suggesting {E}.",
        "findings are suggestive of {E}.",
        "findings are representing {E}.",
        "There are evidence of {E}"
    ],
    "negative": [
        "There is no {E}.",
        "No radiographic evicence for {E}.",
        "No {R} {E}.",
        "No {R}, {R} evidence of {E}.",
        "No convincing signs of {E}.",
        "No {E} is {R}.",
        "no {E}.",
        "There is no convincing signs of {E}."
    ]
}

MASS_SHAPE = {
    "has_single_shape": [
        "Has single mass shape that is {E}.",
        "Has one mass shape {E}.",
        "Mass shape is {E}.",
        "Found one mass shape that is {E}.",
        "mass shape {E} is {R}.",
        "There are evidence of {E} mass.",
        "There is a {E} mass."
    ],
    "no_shape": {
        "Doesn't have mass shape, it is {E}.",
        "Mass shape is {E}.",
        "{E} mass shape.",
        "there is no mass shape, it is {E}.",
        "No evidence of mass, it is {E}."
    },
    "has_many_shapes": [
        "Has several mass shapes, which are {E}.",
        "Has multiple mass shapes, that are {E}.",
        "Has more than one mass shape, {E}.",
        "multiple mass shapes were found such as {E}.",
        "Masses with {E} shapes"
    ]
}

MASS_MARGIN = {
    "has_single_margin": [
        "Has single mass margin that is {E}.",
        "Has one mass margin {E}.",
        "Mass margin is {E}.",
        "{E} mass margin.",
        "There are evidence of {E} margin for the {R} mass."
    ],
    "no_margin": {
        "Doesn't have mass margin, it is {E}.",
        "{E} mass margin {R}.",
        "Mass margin is {E}."
    },
    "has_many_margins": [
        "Has several mass, which are {E}.",
        "Has multiple mass, that are {E}.",
        "Has more than one mass margin, {E}.",
        "The {R} masses has {E}."
    ]
}


def generate_gtr_prompt_sentence(key, n=1, **kwargs):
    # KEYS:
    # gtr_mass:True
    # gtr_calc:True
    
    # gtr_is_architectural_distortion:True (in general, not related to specific group of mass or calc)
    # gtr_is_architectural_distortion:False

    # gtr_is_asymmetry:True (in general, not related to specific group of mass or calc), indicate whether asymetry is observed or not excluding its type
    # gtr_is_asymmetry:False

    REPORT = {
        # M_MARG    : Mass Margin
        # M_SHAPE   : Mass Shape
        # M_MALIG   : Mass Malignancy, note that calc can also be malignant
        # Separate with a comma between the mass shape/margin, and benign/malig classification
        "gtr_mass:True":[
            "The mass was characterized by {M_MARG} {M_SHAPE} on imaging, suggesting a potential {M_MALIG} etiology.",
            "The observed mass demonstrated {M_MARG} {M_SHAPE}, necessitating consideration of its {M_MALIG} characteristics.",
            "The mass exhibited {M_MARG} {M_SHAPE}, suggesting potential {M_MALIG} pathology.",
            "The mass noted {M_MARG} {M_SHAPE}, raising suspicion for {M_MALIG}.",
            "The mass demonstrated {M_MARG} {M_SHAPE}, prompting concern for underlying {M_MALIG}.",
            "The mass indicated {M_MARG} {M_SHAPE}, prompting concern for potential {M_MALIG}.",
            "The imaging revealed a mass with {M_MARG} {M_SHAPE}, warranting further investigation for {M_MALIG}.",
            "The mass displayed {M_MARG} {M_SHAPE}, suggestive of {M_MALIG} features upon imaging.",
            "The of the mass seen on imaging were {M_MARG} {M_SHAPE}, prompting concern for {M_MALIG}.",
            "The imaging findings revealed a mass with {M_MARG} {M_SHAPE}, indicating further evaluation for {M_MALIG}.",
            "The mass demonstrated {M_MARG} {M_SHAPE}, prompting further evaluation for {M_MALIG} features.",
            "The mass displayed {M_MARG} {M_SHAPE}, suggestive of a {M_MALIG} lesion.",
            "Imaging revealed a mass with {M_MARG} {M_SHAPE}, indicative of possible {M_MALIG}.",
            "The imaging findings showed a mass with {M_MARG} {M_SHAPE}, indicative of potential {M_MALIG}.",
            "The found mass appeared {M_MARG} {M_SHAPE}, raising suspicion for {M_MALIG} etiology.",
            "Imaging revealed a mass with {M_MARG} {M_SHAPE}, suggestive of {M_MALIG} pathology.",
            "The mass demonstrated {M_MARG} {M_SHAPE}, indicating a likely {M_MALIG} etiology.",
            "The present mass appeared {M_MARG} {M_SHAPE}, indicating potential {M_MALIG} characteristics.",
            "The observed mass shown {M_MARG} {M_SHAPE}, indicative of {M_MALIG} behavior.",
        ],
        "gtr_calc:True":[
            "The mammogram revealed calcifications {C_DIST}, suggesting potential {C_MALIG} pathology.",
            "Identified calcifications exhibit features indicative of {C_MALIG} {C_DIST}.",
            "The mammography report notes the presence of calcifications with concerning features, suggesting a potential for {C_MALIG} {C_DIST}.",
            "Calcifications observed raised suspicion for {C_MALIG} {C_DIST}.",
            "Reported calcifications display {C_MALIG} characteristics {C_DIST}.",
            "Calcifications identified present with suspicious features, warranting further evaluation for {C_MALIG} {C_DIST}.",
            "The calcifications visualized exhibit {C_MALIG} characteristics {C_DIST}.",
            "Calcifications noted suggesting a higher likelihood of {C_MALIG} {C_DIST}.",
            "Observed calcifications appear {C_MALIG} {C_DIST}.",
            "The calcifications identified present with {C_MALIG} characteristics {C_DIST}."
        ],
        "gtr_mass:True&gtr_calc:True":[
            "The mammography report highlights the presence of a {M_MARG} {M_SHAPE} accompanied by calcifications, indicative of {M_MALIG}.",
            "The mammography findings reveal masses with {M_MARG} {M_SHAPE} and calcifications, suggesting a low likelihood of {M_MALIG}.",
            "Masses identified exhibit {M_MARG} {M_SHAPE} and calcifications with concerning features, raising suspicion for {M_MALIG}.",
            "Observed masses demonstrate a {M_MARG} {M_SHAPE}, along with calcifications, indicating a likelihood of {M_MALIG}.",
            "The mammography report notes mass with {M_MARG} {M_SHAPE} and calcifications demonstrating suspicious features, suggesting {M_MALIG}.",
            "Detected masses display {M_MARG} {M_SHAPE} and associated calcifications, indicating a low probability of {M_MALIG}.",
            "Masses visualized exhibit {M_MARG} {M_SHAPE} and calcifications with concerning features, indicating {M_MALIG}."
        ],
        "suspicious": [ # note that those must be assigned birads 0 
            "Additional imaging or information is needed to make an assessment."
        ],
        "no_gtr": [ # we can assign birads 1, 2, or 3; assigned to birads 1
            "No finding is present in the imaging.",
            "Mammogram shows no evidence of any abnormalities.",
            "Mammogram shows no abnormal findings.",
            "Radiologist assessment reveals no evidence of abnormalities.",
            "Breast tissue appears unremarkable with no signs of pathology.",
            "No suspicious lesions or abnormalities are observed.",
            "Mammogram shows no significant findings."
        ],
        "row.labels['birads']:True":[ # this will come after malig or calc or both togather prompts, within the same sentence 
            "assigning BIRADS score of {B_SCORE} based on the findings.",
            "the mammography report assigns a BIRADS score of {B_SCORE} to guide further clinical decisions.",
            "this concludes assigning a BIRADS score of {B_SCORE}.",
            "a BIRADS score of {B_SCORE} is assigned to effectively communicate the mammography findings.",
            "therefor assigning BIRADS score of {B_SCORE} that serves an importance in the interpretation of mammography results.",
            "as a conclusion BIRADS score {B_SCORE} guides clinical decisions.",
            "thus BIRADS score {B_SCORE} communicates findings effectively.",
            "BIRADS score {B_SCORE} reflects radiologist's assessment.",
            "assigned BIRADS score {B_SCORE} for clinical management."
        ],
        "gtr_histology>0":[ # we use values > 0
            "Histological analysis confirmed the presence of {HISTOLOGY}.", # consistent with the suspicious findings observed on mammography
            "{HISTOLOGY} histology is reported.",
            "The histology examination revealed {HISTOLOGY}.",
            "Histology {HISTOLOGY} is noted.",
            "The histological findings are consistent with {HISTOLOGY}.",
            "Histological analysis reported {HISTOLOGY}.",
            "The histological features are {HISTOLOGY}."
        ],
        "gtr_is_architectural_distortion:True":[
            "The mammogram displayed architectural distortion, indicating possible disruption or retraction of breast tissue.",
            "Architectural distortion was noted on mammography, suggestive of underlying changes in breast tissue organization, necessitating further assessment.",
            "The presence of architectural distortion on the mammogram raised concern.",
            "Mammographic findings revealed architectural distortion.",
            "The observed architectural distortion on mammography warranted correlation with clinical findings.",
            "The presence of architectural distortion on mammography necessitated careful evaluation.",
            "Mammographic evaluation revealed diffuse architectural distortion, prompting consideration for comprehensive breast imaging.",
            "Architectural distortion observed on mammography may indicate localized breast tissue changes.",
            "Mammographic findings showed architectural distortion, suggesting alterations in breast tissue architecture.",
            "Architectural distortion identified on mammography may represent focal tissue changes.",
            "The observed architectural distortion on mammography warranted consideration.",
            "Mammographic evaluation revealed architectural distortion.",
            "Architectural distortion noted on mammography may indicate localized tissue abnormalities.",
            "The presence of architectural distortion on mammography prompted consideration.",
            "The presence of architectural distortion on mammography warranted further investigation.",
        ],
        "gtr_is_architectural_distortion:False":[
            "Mammography showed no evidence of architectural distortion.",
            "No architectural distortion was noted on mammography.",
            "Mammographic evaluation revealed no architectural distortion.",
            "No evidence of architectural distortion was observed on mammography.",
            "Mammography showed no architectural distortion.",
            "No architectural distortion was identified on mammography.",
            "Mammography revealed no evidence of architectural distortion.",
            "Mammography showed no architectural distortion, consistent with normal tissue appearance.",
            "Mammography showed no architectural distortion, suggesting preserved tissue organization.",
            "No evidence of architectural distortion was noted on mammography.",
        ]
    }

    # prompts list
    prompts_report = [] 

    if key == "no_gtr":
        prompts_random = sample(REPORT[key], n)
        prompts_report = prompts_random

    if key == 'gtr_mass:True' or key == "gtr_mass:True&gtr_calc:True":
        M_MARG = kwargs.get('M_MARG', 'unknown')
        M_SHAPE = kwargs.get('M_SHAPE', 'unknown')
        M_MALIG = kwargs.get('M_MALIG')

        prompts_random = sample(REPORT[key], n)
        prompt_replacement = "{M_MARG} {M_SHAPE}"

        for prompt in prompts_random:
            if M_MARG == "unknown":
                # if no margin passed, no need to add unknown label to the sentence
                prompt_replacement = prompt_replacement.replace("{M_MARG} ", "")
            else:
                # if margin passed, add margin info
                prompt_replacement = prompt_replacement.replace("{M_MARG}", f"{M_MARG} margins")

            if M_SHAPE == "unknown":
                # if no shape passed, no need to add shape information to the sentence
                prompt_replacement = prompt_replacement.replace(" {M_SHAPE}", "")
            else:
                prompt_replacement = prompt_replacement.replace("{M_SHAPE}", f"and {M_SHAPE} shape" if M_MARG != "unknown" else f"{M_SHAPE} shape")

            if M_MARG == "unknown" and M_SHAPE == "unknown":
                # if both are unknown, split the sentence from the comma, and remove that part
                prompt = prompt.split(', ')[-1].replace("{M_MALIG}", M_MALIG).capitalize()
            else:
                prompt = prompt.replace("{M_MALIG}", M_MALIG).replace("{M_MARG} {M_SHAPE}", prompt_replacement)
            prompts_report.append(prompt)

    if key == 'gtr_calc:True':
        C_MALIG = kwargs.get('C_MALIG')
        C_DIST = kwargs.get('C_DIST', 'unknown')

        prompts_random = sample(REPORT[key], n)
        prompt_replacement = " {C_DIST}"

        for prompt in prompts_random:
            if C_DIST == "unknown":
                prompt_replacement = prompt_replacement.replace(" {C_DIST}", "")
            else:
                prompt_replacement = prompt_replacement.replace(" {C_DIST}", f" with {C_DIST} distribution")

            prompt = prompt.replace("{C_MALIG}", C_MALIG).replace(" {C_DIST}", prompt_replacement)
            
            prompts_report.append(prompt)

    if 'birads' in key:
        B_SCORE = kwargs.get('B_SCORE')
        prompts_random = sample(REPORT[key], n)

        for prompt in prompts_random:
            prompt = prompt.replace("{B_SCORE}", B_SCORE)
            
            prompts_report.append(prompt)

    if key == "gtr_histology>0":
        HISTOLOGY = kwargs.get('HISTOLOGY')
        prompts_random = sample(REPORT[key], n)

        for prompt in prompts_random:
            prompt = prompt.replace("{HISTOLOGY}", HISTOLOGY)
            prompts_report.append(prompt)
            
    if 'gtr_is_architectural_distortion' in key:
        prompts_random = sample(REPORT[key], n)
        prompts_report = prompts_random


    return ' '.join(prompts_report)

def available_prompts_templates():
    "Returns a list of templates to generate prompts and their respective keys."

    _templates = {
        "BENIGN_MALIG": BENIGN_MALIG,
        "HAS_MASS": HAS_MASS,
        "MASS_SHAPE": MASS_SHAPE,
        "MASS_MARGIN": MASS_MARGIN
    }
    return _templates

def generate_label_prompt_sentence(label_name, label_type, n=20, template=None):
    ''''
    Generates a sentece based on a given label name. 
    It searches in the global constants for a template sentence and random words selection.

    >> sentence = generate_label_prompt_sentence(label_name="mass", label_type="positive", n=1)
    >> print(sentence)
    >> [The presence of mass is obvious.]

    Args:
        label_name (str, list): A string representing the label, that is the expression or a list of expressions.
        label_type (str): The type of the label to differenciate the sentence, either positive or negative.
        n (int): number of prompts to be generated.
        template (dict): The constant dictionary that holds the information to generate from. Default is HAS_MASS

    Returns:
        prompt_sentences (list): A list of prompt/s.

    '''
    # get the dictionary to look from
    _dictionary = globals().get(template) if globals().get(template) is not None else HAS_MASS
    _random_selection = ["present", "seen", "noted", "visible", "obvious", "appreciable", "evident", "found"]

    # Create a prompt sentence and replace the expression with the label name passed to the function
    prompt_sentences = sample(_dictionary[label_type], n)
    prompts_selected = []

    for prompt_sentence in prompt_sentences:
        # replace the expression
        prompt_sentence = prompt_sentence.replace("{E}", label_name) if isinstance(label_name, str) else prompt_sentence.replace("{E}", ", ".join(label_name))

        # count the number of random selection words
        n_random_selection = prompt_sentence.count("{R}")
    
        # selects and replaces all required random selections placeholders
        random_selections = sample(_random_selection, n_random_selection)
        
        for selection in random_selections:
            # We replace the first finding for every iteration
            prompt_sentence = prompt_sentence.replace("{R}", selection, 1)

        prompts_selected.append(prompt_sentence)
        
    return prompts_selected

def generate_label_prompt_report(dataset: pd.DataFrame, new_col: str):
    
    assert isinstance(dataset, pd.DataFrame), "`dataset` has to be a pandas dataframe."
    
    for i, row in dataset.iterrows():
        report = []

        # benign vs malignant
        report.append(sample(BENIGN_MALIG["benign"], 1)[0] if row['image_label'] == 0 else sample(BENIGN_MALIG["malignant"], 1)[0])

        # has mass
        report.append(generate_label_prompt_sentence("mass", "positive" if row['has_mass'] else "negative", n=1, template="HAS_MASS")[0])

        # mass shape
        mass_shape_list = list(set([val.lower() for val in row['mass_shape'] if val != -1]))
        if len(mass_shape_list) == 0:
            mass_shape_search = "no_shape"
            mass_shape_list = ["unknown"]
        elif len(mass_shape_list) == 1:
            mass_shape_search = "has_single_shape"
        else:
            mass_shape_search = "has_many_shapes"
        report.append(generate_label_prompt_sentence(mass_shape_list, mass_shape_search, n=1, template="MASS_SHAPE")[0])

        # mass margin
        mass_margin_list = list(set([val.lower() for val in row['mass_margin'] if val != -1]))
        if len(mass_margin_list) == 0:
            mass_margin_search = "no_margin"
            mass_margin_list = ["unknown"]
        elif len(mass_margin_list) == 1:
            mass_margin_search = "has_single_margin"
        else:
            mass_margin_search = "has_many_margins"
        report.append(generate_label_prompt_sentence(mass_margin_list, mass_margin_search, n=1, template="MASS_MARGIN")[0])

        # random shuffle the sentences
        random.shuffle(report)

        # construct the report
        report = " ".join(report)

        dataset.at[i, new_col] = report

    return dataset