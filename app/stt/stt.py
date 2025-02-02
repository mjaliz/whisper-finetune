import os
import whisper


class SpeechToText:
    def __init__(self):
        self.model = whisper.load_model(
            "medium.en",
            device="cuda",
            in_memory=False,
            download_root="/home/mjaliz/whisper-finetune/whisper_models",
        )

    def whisper_func(self, audio_file):
        result = self.model.transcribe(audio_file)
        return result["text"]
