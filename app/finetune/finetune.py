import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path
from datasets import load_dataset, DatasetDict
from datasets import Audio

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


CURRENT_DIR = Path(__file__).parent
# AUDIO_DIR = CURRENT_DIR.joinpath("audios")
DATASET_DIR = CURRENT_DIR.joinpath("leranit")
# DATASET_DIR.joinpath("data").joinpath("train").mkdir(parents=True, exist_ok=True)
# DATASET_DIR.joinpath("data").joinpath("test").mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = "openai/whisper-tiny.en"

learnit = load_dataset(
    path="audiofolder",
    data_dir="/home/mjaliz/whisper-finetune/app/finetune/leranit",
)

feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)

tokenizer = WhisperTokenizer.from_pretrained(
    WHISPER_MODEL, language="English", task="transcribe"
)

processor = WhisperProcessor.from_pretrained(
    WHISPER_MODEL, language="English", task="transcribe"
)


learnit = learnit.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


learnit = learnit.map(prepare_dataset, num_proc=4)


model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english", task="transcribe"
)

model.config.forced_decoder_ids = forced_decoder_ids
model.config.suppress_tokens = []


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper_tiny-lr",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=learnit["train"],
    eval_dataset=learnit["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)
trainer.train()
