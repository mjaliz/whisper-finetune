import random
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment

from app.stt.stt import SpeechToText

CURRENT_DIR = Path(__file__).parent
AUDIO_DIR = CURRENT_DIR.joinpath("audios")
DATASET_DIR = CURRENT_DIR.joinpath("leranit")
DATASET_DIR.joinpath("data").joinpath("train").mkdir(parents=True, exist_ok=True)
DATASET_DIR.joinpath("data").joinpath("test").mkdir(parents=True, exist_ok=True)


def get_random_split_dir() -> str:
    splits = ["data/train"] * 8 + ["data/test"] * 2
    return random.choice(splits)


def transcribe_data():
    stt = SpeechToText()
    audios = []
    texts = []
    duratoin = 0
    for audio in tqdm(list(AUDIO_DIR.glob("*.mp3"))):
        aus = AudioSegment.from_file(audio)
        txt = stt.whisper_func(str(audio))
        if txt.strip() == "":
            continue
        duratoin += aus.duration_seconds
        split = get_random_split_dir()
        shutil.copy(audio, DATASET_DIR.joinpath(split).joinpath(audio.name))
        audios.append(f"{split}/{audio.name}")
        texts.append(txt.strip())

    df = pd.DataFrame({"file_name": audios, "sentence": texts})
    df.to_csv(DATASET_DIR.joinpath("metadata.csv"), index=None)
    print(duratoin)


if __name__ == "__main__":
    transcribe_data()
