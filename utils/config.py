from pathlib import Path

CLASS_NAMES = ["Engaged", "Not_Engaged", "Drowsy"]
DEFAULT_IMAGE_SIZE = 224
DEFAULT_MODEL_NAME = "mobilenet_v2"
DEFAULT_MODEL_PATH = Path("models") / "best_model.pt"
DEFAULT_NUM_WORKERS = 0
DEFAULT_SEED = 42
FER2013_EMOTION_TO_CLASS = {
    0: 1,  # angry -> Not_Engaged
    1: 1,  # disgust -> Not_Engaged
    2: 1,  # fear -> Not_Engaged
    3: 0,  # happy -> Engaged
    4: 2,  # sad -> Drowsy
    5: 0,  # surprise -> Engaged
    6: 0,  # neutral -> Engaged
}
