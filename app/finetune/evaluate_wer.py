import torch
import evaluate
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and processor
MODEL_NAME = "openai/whisper-small.en"
# MODEL_NAME_TINY = "openai/whisper-tiny.en"
# CHECK_POINT = "/home/mjaliz/whisper-finetune/whisper_tiny-lr/checkpoint-600"

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(
    device
)  # Move model to GPU

# Load WER metric
wer_metric = evaluate.load("wer")

# Load dataset (Common Voice English test set)
dataset = load_dataset(
    "audiofolder",
    data_dir="/home/mjaliz/whisper-finetune/app/finetune/leranit",
    split="train",
)

# Resample audio to 16kHz (required by Whisper)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


def transcribe_and_evaluate(batch):
    # Extract audio array and sampling rate
    audio = batch["audio"]

    # Convert to log-Mel spectrogram
    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features  # This should already be [num_features, time_steps]

    # Convert to tensor and move to GPU
    input_features = torch.tensor(input_features).unsqueeze(0).to(device)

    # Ensure the input is in the correct format (should be 3D: [1, 80, T])
    input_features = input_features.squeeze(1)  # Remove extra dimension if present

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english", task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids
        )

    # Decode predicted text
    batch["transcription"] = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]

    return batch


# Apply transcription
dataset = dataset.map(transcribe_and_evaluate)

# Extract predictions and references
predictions = dataset["transcription"]
references = dataset["sentence"]  # Assuming 'sentence' contains ground truth text

# Compute WER
wer_score = wer_metric.compute(predictions=predictions, references=references)
print(f"Word Error Rate (WER): {wer_score * 100:.2f}%")
