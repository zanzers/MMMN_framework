from algorithm.image_forensics.image_forensics import *
from algorithm.image_forensics.global_algo import *
from algorithm.image_forensics.extract_features import *
from algorithm.helper_tools.helper import *

# Main Caller and process

def start_analyze(user_input: np.ndarray, input_type: str) -> np.ndarray:
    
    input_type = check_input(input_type)
    create_saveImg([
        ('original.jpg', user_input)
    ], remove_after=True)


    if input_type == "image":
        start_IF(user_input)
    elif input_type == "video":
        pass
    else:
        pass


def start_IF(img: np.ndarray):

    image_forinsics = Image_Forensics(img)

    ela = image_forinsics.ela_anaylze(85, 40)
    ela_features = extractELA_features(ela)
    # initial_report(ela_features)

    norm, prnu, noise = image_forinsics.noise_analyze(7, 49)
    noise_features = extractNOISE_features(norm, prnu, noise)
    # initial_report(noise_features)

    
    copy_move = image_forinsics.compyMove(16, 8, 1000)
    copyMove_features = extractCOPYMOVE_features(copy_move)
    # initial_report(copyMove_features)

    resample = image_forinsics.resample_detect()
    resample_features = extractResample_features(resample)



def  proced_frequecny(img):
    pass


def deepfake(img):
    pass


