# This can be improved by having all labels in a single dict

from enum import Enum

''''
- To get the class label names from an enum, use:
    `class_names = [label.name for label in EnumClass]`

- To get the class labels values from an enum, use:
    `class_names = [label.value for label in EnumClass]`
'''

class HasArchDistortion(Enum):
    noarchitecturaldistortion = 0
    displayedarchitecturaldistortion = 1

class BenignMalignantDatasetLabels(Enum):
    benign = 0
    malignant = 1

class HasMassLabels(Enum):
    nomass = 0
    mass = 1 # try with "hasmass"?

class HasCalcification(Enum):
    negative = 0
    hascalcification  = 1

class MassShapeLabels(Enum):
    unknown = 0
    oval = 1
    round = 2
    irregular = 3
    # lobular = 4

class MassMarginLabels(Enum):
    unknown = 0
    circumscribed = 1
    obscured = 2
    spiculated = 3
    illdefined = 4
    # indistinct = 2
    # microlobulated = 2 


# '''Below are the enums of the gtr cpp file.'''
class gtr_Malign(Enum):
    '''If False was found, then benign'''
    malignant = True

class gtr_Mass(Enum):
    '''If False was found, then no mass'''
    mass = True

class gtr_MassMargin(Enum):
    circumscribed = 1
    illdefined = 2
    spiculated = 3
    obscured = 4
   
gtr_Histology = {
    1:  "ductal carcinoma in situ (DCIS)",
    2:  "invasive ductal carcinoma (IDC)",
    3:  "lobular carcinoma in situ (LCIS)",
    4:  "invasive lobular carcinoma (ILC)",
    5:  "papilloma in situ",
    6:  "infiltrative papilloma, intracystic carcinoma",
    7:  "medullar carcinoma",
    8:  "adenoid-cystic carcinoma",
    9:  "mucinous/colloid carcinoma",
    10: "tubular carcinoma",
    11: "plaveiselcel carcinoma",
    12: "M. Paget",
    13: "sarcoma",
    14: "Non Hodgkin lymphoma",
    15: "metastasis from elsewhere",
    20: "fibroadenoma",
    21: "solitary cyst",
    22: "radial scar, complex sclerosing lesion",
    23: "phyllodes tumor",
    24: "single papilloma",
    25: "multiple intraductal papillomatosis",
    26: "fibrocystic change",
    # 27: "benign microcalcification",
    # 28: "other benign lesion",
    29: "atypical ductal hyperplasia",
    30: "normal tissue",
    # 31: "unknown benign",
    99: "invasive carcinoma"
    }

def get_key_from_value(enum_class, value):
    for key, enum_value in enum_class.__members__.items():
        if enum_value.value == value:
            return key
    # raise ValueError("Value not found in the enum")
    return "unknown"