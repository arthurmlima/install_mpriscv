"""
Criado por: Ivan C Perissini
Data: 12/08/2020
Função: Parametrization file with the default value of many used functions
Última alteração: 25/07/2020
"""

import matplotlib.colors as mcolors


def __version__():
    return 'parameters version: v0.2'


# ======================== GET FOLDERS PATH =========================
# Used for user output, ML metrics calculation and graphics color scheme
def get_result_folder():
    return r"C:\\Projetos\\Demeter\\codes\\results\\"


def get_model_folder():
    return r"C:\\Projetos\\Demeter\\codes\\models\\"


def get_graphviz_folder():
    return r"C:\\Projetos\\Demeter\\codes\\core\\bin\\Graphviz\\bin"


# ======================== LABEL PARAMETERS =========================
# Used for user output, ML metrics calculation and graphics color scheme
def get_label_info():
    # Label name: [label number, label color, label shortcut]
    return {'Solo': [0, mcolors.CSS4_COLORS['brown'], 's'],
            'Massa 1': [1, mcolors.CSS4_COLORS['yellow'], '1'],
            'Massa 2': [2, mcolors.CSS4_COLORS['greenyellow'], '2'],
            'Massa 3': [3, mcolors.CSS4_COLORS['limegreen'], '3'],
            'Massa 4': [4, mcolors.CSS4_COLORS['green'], '4'],
            'Massa 5': [5, mcolors.CSS4_COLORS['darkgreen'], '5'],
            'Animal': [6, mcolors.CSS4_COLORS['royalblue'], 'a'],
            'Outros': [7, mcolors.CSS4_COLORS['gray'], 'r'],
            'Sombra': [8, mcolors.CSS4_COLORS['black'], 'e'],
            'Daninha': [9, mcolors.CSS4_COLORS['blue'], 'd'],
            'Arvore': [10, mcolors.CSS4_COLORS['blue'], 't'],
            'coleta1': [11, mcolors.CSS4_COLORS['yellow'], 'c1'],
            'coleta2': [12, mcolors.CSS4_COLORS['greenyellow'], 'c2'],
            'coleta3': [13, mcolors.CSS4_COLORS['limegreen'], 'c3'],
            'coleta4': [14, mcolors.CSS4_COLORS['green'], 'c4'],
            'coleta5': [15, mcolors.CSS4_COLORS['darkgreen'], 'c5']}


# ======================== SPECIES GROUP =========================
def get_species_group():
    # Label name: [label number, label color, label shortcut]
    return {'ESTRELAAFRICANA': 'short',
            'DECUMBENS': 'short',
            'RUZIZIENSIS': 'short',
            'CAYANA': 'average',
            'SABIA': 'average',
            'PIATA': 'average',
            'PIQUETE156': 'average',
            'PIQUETE153': 'average',
            'PIQUETE154': 'average',
            'TAMANI': 'average',
            'XARAES': 'average',
            'MOBAÇA': 'tall',
            'PANICUM': 'tall',
            'TANZANIA': 'tall',
            'QUENIA': 'tall',
            'ELEFANTE': 'tall',
            'CYNODON': 'tall',
            'ZURI': 'tall',
            'ZURI': 'tall',
            }


# ======================== CAMERA PARAMETERS =========================
# Used for area coverage calculation and other geometric related functions
def get_camera_info():
    return {'f': 4.5,
            'widht_pixels': 3968,
            'heigth_pixels': 2976,
            'widht_sensor': 6.17,
            'heigth_sensor': 4.55,
            'fov': 81.9}


# ======================== CONVERSION USED FOR DESCRIPTORS =========================
def get_conversion_list():
    return ['XYZ',
            'YUV',
            'LAB',
            'LUV',
            'HSV',
            'YCrCb',
            'CrCgCb',
            'rgb',
            'I1I2I3',
            'l1l2l3',
            'TSL']


# ======================== INDEXES USED FOR DESCRIPTORS =========================
def get_index_list():
    return ['NDI',
            'ExG',
            'ExR',
            'ExGR',
            'CIVE',
            'VEG',
            'MExG',
            'COM1',
            'COM2',
            'YmX',
            'mA',
            'BmA',
            'mExU',
            'VmU',
            'mCrpCb',
            'Cg',
            'ExCg',
            'l3',
            'I3',
            'VARI',
            'TGI',
            'NGRDI',
            'RGBVI',
            'GLI',
            'canny',
            'dft']


# ======================== INDEXES USED FOR DESCRIPTORS =========================
def get_old_index_list():
    return ['NDI',
            'ExG',
            'ExR',
            'ExGR',
            'CIVE',
            'VEG',
            'MExG',
            'COM1',
            'COM2',
            'YmX',
            'mA',
            'BmA',
            'mExU',
            'VmU',
            'mCrpCb',
            'Cg',
            'ExCg',
            'l3',
            'I3']


# ======================== TEXTURE MARGIN PARAMETER =========================
def get_default_margin():
    return 20


# ======================== HOG FUNCTION PARAMETERS =========================
# Format: (Grid size, orientations, margin, mode)
def get_hog_parameters():
    return ((5, 4, 0, 'default'), (5, 4, get_default_margin(), 'default'),
            (5, 8, 0, 'statistics'), (5, 8, get_default_margin(), 'statistics'),
            (10, 4, 0, 'statistics'), (10, 4, get_default_margin(), 'statistics'))


# ======================== MODELS LIST =========================
def get_models_list():
    # Model: RFC
    # Period: Day / Dawn / Dusk / Night
    # Light: cloudy / sunny
    # Season: water/ dry / transition
    # Quality: low / medium / high
    # Species group: short / average / tall
    # Response: label_best / label_mean / label_rank / label_ROGESTER GOMES
    # Classes: complete / simple / basic
    # Balance: true / false
    return [
        {
            'model': 'RFC_n10d05',
            'inf_period': '',
            'inf_light': '',
            'inf_season': '+dry',
            'inf_quality': '-low',
            'inf_species_group': '+average',
            'label': 'label_best',
            'class': 'complete',
            'balance': 'false'
        },
        {
            'model': 'RFC_n10d05',
            'inf_period': '+Day',
            'inf_light': '-cloudy',
            'inf_season': '+dry',
            'inf_quality': '-low',
            'inf_species_group': '+tall',
            'label': 'label_best',
            'class': 'basic',
            'balance': 'true'
        },
        {
            'model': 'RFC_n20d09',
            'inf_period': '',
            'inf_light': '-cloudy',
            'inf_season': '+dry',
            'inf_quality': '-low',
            'inf_species_group': '+average',
            'label': 'label_best',
            'class': 'simple',
            'balance': 'true'
        },
        {
            'model': 'RFC_n20d09',
            'inf_period': '+Day',
            'inf_light': '-cloudy',
            'inf_season': '+dry',
            'inf_quality': '-low',
            'inf_species_group': '+average',
            'label': 'label_best',
            'class': 'simple',
            'balance': 'false'
        }
    ]
