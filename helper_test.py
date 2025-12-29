import ast
import concurrent.futures
import gc
import io
import json
import logging
import os
import re
import shutil
import smtplib
import ssl
import uuid
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from urllib.parse import unquote, urlparse

import httpx
import librosa
import nltk
import pytz
import requests
import soundfile as sf
import torch
import wget

# import whisperx
# from ctc_forced_aligner import (
#     generate_emissions,
#     get_alignments,
#     get_spans,
#     load_alignment_model,
#     postprocess_results,
#     preprocess_text,
# )
import whisper

# from nemo.collections.asr.parts.utils.speaker_utils import (
#     labels_to_pyannote_object, rttm_to_labels)
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    FileSource,
    PrerecordedOptions,
)

# from deepmultilingualpunctuation import PunctuationModel
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from keybert import KeyBERT
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from omegaconf import OmegaConf
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import logging as trans_logging

from slack_handler import post_error

# Set verbose only for NeMo errors
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
trans_logging.set_verbosity(trans_logging.FATAL)

AGENT_NAMES = [
    "abdul",
    "abhay",
    "aftab",
    "akash",
    "ameen",
    "animesh",
    "arbab",
    "ayush",
    "ayushi",
    "chinmay",
    "hamid",
    "hasan",
    "india",
    "laksh",
    "manisha",
    "masarrat",
    "mohammad",
    "nadeemulla",
    "naman",
    "nitin",
    "prem",
    "priya",
    "rakesh",
    "sarafarz",
    "satyendra",
    "sheetal",
    "shubham",
    "syed",
    "tarun",
    "utkarsh",
    "vedansh",
    "vishwajeet",
    "yash",
]

# #################### Load the environment variables
load_dotenv()

# Deepgram config
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

# OpenAI Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")

#Gemini Config
# Google Gemini Config (replace OpenAI)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro-latest")  # Default to latest stable; set in .env if needed

# MongoDB Config
MONGO_DB_HOST = os.environ.get("MONGO_DB_HOST")
MONGO_DB_PORT = os.environ.get("MONGO_DB_PORT", None)
MONGO_DB_USER = os.environ.get("MONGO_DB_USER")
MONGO_DB_PASSWORD = os.environ.get("MONGO_DB_PASSWORD")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME")
MONGO_DB_ASR_COLLECTION = "asr_feeds"
MONGO_DB_CDR_COLLECTION = "cdr"

if not MONGO_DB_PORT:
    mongo_db_uri = (
        f"mongodb+srv://{MONGO_DB_USER}:{MONGO_DB_PASSWORD}@{MONGO_DB_HOST}"
        f"/{MONGO_DB_NAME}"
    )
else:
    mongo_db_uri = (
        f"mongodb://{MONGO_DB_USER}:{MONGO_DB_PASSWORD}@{MONGO_DB_HOST}"
        f":{MONGO_DB_PORT}/{MONGO_DB_NAME}"
    )
print("MongoDB URI:", mongo_db_uri)

# Email Config
SENDER_MAIL = os.environ.get("SENDER_MAIL")
SENDER_PWD = os.environ.get("SENDER_PWD")
# #################### Environment variables loaded

# #################### [Optional] File to store failed JSON strings
os.makedirs("debug", exist_ok=True)
debug_file = open("debug/invalid_ai_responses.txt", "w")
debug_file.close()
# ####################

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1
print(f"\n\nWorking on {'GPU' if device >= 0 else 'CPU'}")
if device == -1:
    print("Warning: No CUDA GPU detected. Running on CPU may be slow or fail for some models.")

# #################### Configure Whisper
whisper_model_dir = "model"
whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model_name = "medium.en" if whisper_device == "cuda" else "small.en"
whisper_floating_point = whisper_device == "cuda"

# Initialize whisper model
whisper_model = whisper.load_model(
    whisper_model_name, whisper_device, download_root=whisper_model_dir
)
# #################### Whisper model initialized

# #################### Initialize Deepgram model
config = DeepgramClientOptions(verbose=False)
deepgram_model = DeepgramClient(DEEPGRAM_API_KEY, config)
deepgram_model_options = PrerecordedOptions(
    model="nova-2",
    smart_format=True,
    utterances=True,
    punctuate=True,
    diarize=True,
    language="hi",
)

# # Initialize the ChatOpenAI model
# llm = ChatOpenAI(
#     model=OPENAI_MODEL,
#     max_tokens=None,
#     timeout=None,
#     temperature=1,
#     max_retries=5,
#     api_key=OPENAI_API_KEY,
#     # model_kwargs={"response_format": {"type": "json_object"}},
# )

# Initialize the ChatGoogleGenerativeAI model
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=1,  
    max_tokens=None,  # No limit (Gemini caps at ~8K-32K tokens/model)
    max_retries=5, 
    timeout=None,
    safety_settings=None #to disable content filters if calls have sensitive topics (e.g., complaints)
)

# Initialize SentenceTransformer & KeyBERT model
st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
kw_model = KeyBERT(st_model)


# #################### START: Download Audio
def is_valid_url(url: str) -> bool:
    """
    Check if the URL is valid and contains a scheme.

    Args:
        url (str): URL to be validated

    Return:
        bool: True if url is valid, else False
    """
    parsed_url = urlparse(url)
    return parsed_url.scheme in ("http", "https") and "XXX" not in url


def extract_filename_from_url(url: str) -> str:
    """
    Extract the filename from the URL.

    Args:
        url (str): URL from which filename will be extracted

    Returns:
        str: Extracted filename with file type
    """
    parsed_url = urlparse(url)
    return os.path.basename(unquote(parsed_url.path))


def load_audio_from_url(url: str) -> io.BytesIO | None:
    """
    Download audio data from the given URL

    Args:
        url (str): URL for the audio to download

    Returns:
        io.BytesIO or None: Audio stream for the audio file if fetched,
            otherwise None
    """
    try:
        print(f"\n\nAttempting to download audio from: {url}")
        response = requests.get(url, stream=True)
        print(f"\n\nHTTP Response Status Code: {response.status_code}")
        if response.status_code == 200:
            audio_data = io.BytesIO(response.content)
            print("Audio successfully loaded")
            return audio_data
        else:
            print(f"Failed to download audio. Status code: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        post_error()
        print("Connection failed")
        return None
    except Exception as err:
        post_error()
        print(f"Error occurred while downloading {url}: {err}")
        return None


def save_audio_file(audio_data: io.BytesIO, filename: str, download_dir: str) -> str | None:
    """
    Save the audio data to the specified file path.

    Args:
        audio_data (io.BytesIO): Stream bytes of the audio
        filename (str): Filename for audio
        download_dir (str): Directory path to store the file

    Returns:
        str or None: Returns the file path if saved, otherwise None
    """
    try:
        # Extract the filename from the URL
        save_path = os.path.join(download_dir, filename)

        print(f"\n\nAttempting to save audio to: {save_path}")
        with open(save_path, "wb") as f:
            f.write(audio_data.getbuffer())
        print(f"\n\nAudio successfully saved to {save_path}")
        return save_path

    except Exception as err:
        post_error()
        print(f"Failed to save audio: {err}")
        return None


def download_audio_file(recording_url: str, download_dir: str) -> str | None:
    """
    Download the audio stream and save it to local system

    Args:
        recording_url (str): URL to fetch audio data
        download_dir (str): Directory path to store the file

    Returns:
        str or None: Return audio path if file is downloaded properly,
            otherwise None
    """
    audio_data = load_audio_from_url(recording_url)
    if not audio_data:
        return None

    filename = extract_filename_from_url(recording_url)
    # Save the downloaded audio in local storage
    saved_path = save_audio_file(audio_data, filename, download_dir)
    return saved_path


# #################### END: Download Audio


# #################### START: Process Audio
def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> None:
    """
    Convert MP3 file to .WAV format

    Args:
        mp3_path (str): Path to .mp3 file
        wav_path (str): Path to store .wav file after conversion

    Returns:
        None
    """
    y, sr = librosa.load(mp3_path, sr=None)
    sf.write(wav_path, y, sr)
    print("\n\nMP3 file converted to WAV file")


def deepgram_transcribed_aligned(audio_path):
    global deepgram_model, deepgram_model_options

    try:
        print("\n\nDEEPGRAM == Laod audio")
        with open(audio_path, "rb") as file:
            buffer_data = file.read()

        print("\n\nDEEPGRAM == Transcribe audio")
        payload = {"buffer": buffer_data}
        response = deepgram_model.listen.rest.v("1").transcribe_file(
            payload,
            deepgram_model_options,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )
        response_data = response.to_dict()
        # print("\n\n============ DEEPGRAM transcription Result")
        # print(response_data)

        if "results" in response_data:
            print("\n\nDEEPGRAM == Generate transcript")
            # trans_segments = response_data["results"]["utterances"]
            response_data = response_data["results"]["channels"][0]["alternatives"][0]
            full_transcript = response_data["transcript"]

            trans_segments = response_data["words"]
            # Sort the segments based on start time
            # trans_segments = sorted(trans_segments, key=lambda x: x['start'])

            print(full_transcript)
            return trans_segments, full_transcript

        # If Deepgram API won't work
        assert False, "Unable to fetch data from DEEPGRAM"
    except Exception as e:
        post_error()
        print(f"**ERROR**: {type(e)}: {str(e)}")


def whisperx_transcribed_aligned(audio_file: str, language: str | None = None):
    global whisper_model, whisper_device, whisper_floating_point

    try:
        print("\n\nWHISPERX == Transcribe Audio")

        # Transcribe Audio
        result = whisper_model.transcribe(
            audio_file, fp16=whisper_floating_point, word_timestamps=True
        )
        print(result)

        # Sort the segments based on start time
        trans_segments = sorted(result["segments"], key=lambda x: x["start"])

        # Clean unnecessary data & Store word list
        keys_to_remove = [
            "id",
            "seek",
            "tokens",
            "temperature",
            "avg_logprob",
            "compression_ratio",
            "no_speech_prob",
        ]
        word_bucket = []
        for segment in trans_segments:
            word_bucket.extend(segment["words"])
            for key in keys_to_remove:
                segment.pop(key, None)

        # Collect garbage [Optional]
        gc.collect()
        torch.cuda.empty_cache()

        print("Transcription completed")
        return word_bucket, result["text"]  # trans_segments, result["text"]
    except Exception as e:
        post_error()
        print(f"**ERROR**: {type(e)}: {str(e)}")


def create_dummy_rttm(wav_path, rttm_path):
    duration = librosa.get_duration(filename=wav_path)
    with open(rttm_path, "w") as f:
        f.write(
            f"SPEAKER {os.path.basename(wav_path)} 1 0.000 {duration:.3f} <NA> <NA> SPEAKER_00 <NA> <NA>"
        )
    print("\n\nDummy RTTM file created")


def create_manifest_for_nemo(manifest_path, wav_path, rttm_path):
    meta = {
        "audio_filepath": wav_path,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": None,
        "rttm_filepath": rttm_path,
        "uem_filepath": None,
    }

    with open(manifest_path, "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")
    print("\n\nManifest file created")


def create_new_config_for_nemo(request_dir, manifest_path, output_dir, derived):
    # Load settings
    model_config = os.path.join(request_dir, "diar_infer_telephonic.yaml")
    if not os.path.exists(model_config):
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
        model_config = wget.download(config_url, request_dir)

    # Update settings
    config = OmegaConf.load(model_config)

    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"

    config.diarizer.num_workers = 1
    config.diarizer.manifest_filepath = manifest_path
    config.diarizer.out_dir = output_dir

    # Speaker Embedding Configuration
    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    if derived == "India":
        config.diarizer.speaker_embeddings.parameters.shift_length = 0.5
        config.diarizer.speaker_embeddings.parameters.window_length = 1.5

    # VAD Configuration
    config.diarizer.oracle_vad = False
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.7 if derived == "India" else 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = 0.2 if derived == "India" else -0.05

    # Clustering Configuration
    config.diarizer.clustering.parameters.oracle_num_speakers = False
    if derived == "India":
        config.diarizer.clustering.parameters.oracle_num_speakers = False
        config.diarizer.clustering.parameters.max_num_speakers = 2
        config.diarizer.clustering.parameters.enhanced_count_threshold = 0.4

    # MSDD Model Configuration
    config.diarizer.msdd_model.model_path = "diar_msdd_telephonic"
    if derived == "India":
        config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.6, 1.0]

    print("\n\nNeMo config created")
    return config


def get_diarization_results(pred_rttm_path):
    speaker_ts = []
    with open(pred_rttm_path, "r") as f:
        for line in f:
            fields = line.split()
            start = float(fields[3])
            end = start + float(fields[4])
            speaker = int(fields[7].split("_")[-1])
            speaker_ts.append(
                {"start_time": start, "end_time": end, "speaker": speaker}
            )

    print("\n\nDiarization Result:")
    print(speaker_ts)
    return speaker_ts


def run_diarization(wav_path, request_dir, file_name_no_ext, derived: str | None):
    try:
        enable_stemming = True

        print("\n\nStarting diarization")
        output_dir = os.path.join(request_dir, "oracle_vad")
        os.makedirs(output_dir, exist_ok=True)

        # Generate dummy RTTM file for oracle diarization
        rttm_path = os.path.join(request_dir, "audio.rttm")
        create_dummy_rttm(wav_path, rttm_path)

        # Create manifest file
        manifest_path = os.path.join(request_dir, "input_manifest.json")
        create_manifest_for_nemo(manifest_path, wav_path, rttm_path)

        # Load and modify config
        config = create_new_config_for_nemo(
            request_dir, manifest_path, output_dir, derived
        )

        # Run Neural Diarizer
        print("\n\nStarting diarization")
        msdd_model = NeuralDiarizer(cfg=config).to("cuda:0")
        msdd_model.diarize()
        print("\n\nDiarization completed")

        # Fetch and parse diarization results
        pred_rttm_path = f"{output_dir}/pred_rttms/{file_name_no_ext}.rttm"
        diarized_result = get_diarization_results(pred_rttm_path)

        return diarized_result
    except Exception as e:
        post_error()
        print(f"**ERROR**: {type(e)}: {str(e)}")


def map_diarization_with_segments(word_bucket, segments):
    print("\n\nMapping words with diarization")
    print("Length of words: ", len(word_bucket))
    print("Length of segments: ", len(segments))

    ctr = 0
    speaker_statements = []
    for segment in segments:
        # print(segment)
        sentence_words = []

        while ctr < len(word_bucket):
            word = word_bucket[ctr]
            word["word"] = word["word"].strip()

            if "start" not in word:
                # print(ctr, "--", word)
                sentence_words.append(word)
                ctr += 1
            elif segment["start_time"] <= word["start"] <= segment["end_time"]:
                # print(ctr, "--", word)
                sentence_words.append(word)
                ctr += 1
            elif word["start"] < segment["start_time"]:
                # print(ctr, "--", "SKIP", "--", word)
                sentence_words.append(word)
                ctr += 1
            else:
                # print("\n\n")
                break

        # Add all the words under single segment
        speaker_statements.append({"segment": segment, "words": sentence_words})

    return speaker_statements


def realign_mapped_segments(word_segments):
    print("\n\nStarting realignment")

    aligned_statements = []
    new_statement = []

    last_statement_start = 0.0
    last_statement_end = 0.0
    last_speaker = None

    for i in word_segments:
        start_time = i["segment"]["start_time"]
        end_time = i["segment"]["end_time"]
        curr_speaker = i["segment"]["speaker"]

        if last_speaker != curr_speaker:
            if last_speaker is None:
                # print("Speaker:", curr_speaker)
                pass
            else:
                # print("\n\n")
                # print("Speaker changed from --", last_speaker, "-- to:", curr_speaker)
                aligned_statements.append(
                    {
                        "speaker": last_speaker,
                        "start_time": last_statement_start,
                        "end_time": last_statement_end,
                        "text": " ".join(new_statement),
                    }
                )

            new_statement = []
            last_statement_start = start_time
            last_statement_end = 0.0

        last_speaker = curr_speaker
        # print(i['segment'])

        new_statement.append(" ".join(word["word"].strip() for word in i["words"]))

        if len(i["words"]) > 0:
            last_word_end = i["words"][-1]["end"] if "end" in i["words"][-1] else 0.0
            last_statement_end = last_word_end if last_word_end > end_time else end_time

    if len(aligned_statements) == 0:
        aligned_statements.append(
            {
                "speaker": last_speaker,
                "start_time": last_statement_start,
                "end_time": last_statement_end,
                "text": " ".join(new_statement),
            }
        )

    # Remove the data not containing any text
    aligned_statements = [i for i in aligned_statements if len(i["text"]) > 0]

    return aligned_statements


def load_deviations_list_from_folder():
    extracted_name_mistakes = {}

    deviation_list_dir = "deviations"
    if not os.path.exists(deviation_list_dir):
        print("Skipping replace; directory not found")
        return None

    # Iterate over each file in the folder
    for filename in os.listdir(deviation_list_dir):
        if filename.endswith(".txt"):
            # Extract the key from the filename (without the .txt extension)
            key = filename.rsplit(".", 1)[0]

            # Read the deviations from the file
            with open(
                os.path.join(deviation_list_dir, filename),
                "r",
                encoding="utf-8",
            ) as file:
                known_deviations = [
                    line.strip() for line in file.readlines() if line.strip()
                ]

            extracted_name_mistakes[key] = known_deviations

    return extracted_name_mistakes


def replace_deviations_in_string(known_deviations, text):
    # Regular expression pattern to match any deviation
    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, known_deviations.keys())) + r")\b",
        re.IGNORECASE,
    )

    # Lambda function for case-insensitive replacement
    return pattern.sub(lambda match: known_deviations[match.group(0).lower()], text)


def replace_deviations(data, data_type: str = "text"):
    print(f"\n\nReplacing Deviations in data -- {data_type}")

    # Load known entity deviations
    known_name_mistakes = load_deviations_list_from_folder()
    if not known_name_mistakes:
        print("Skipping replace; no deviations found")
        return data
    # print("Deviation list:")
    # print(known_name_mistakes)

    replacement_map = {
        dev.lower(): main
        for main, deviations in known_name_mistakes.items()
        for dev in deviations
    }
    if data_type == "text":
        updated_data = replace_deviations_in_string(replacement_map, data)
    else:
        for record in data:
            record["text"] = replace_deviations_in_string(
                replacement_map, record["text"]
            )
        updated_data = data

    print("Updated Data:")
    print(updated_data)
    return updated_data


def format_conversation(conversation: dict | list) -> str:
    """
    Function to convert conversation to a readable format and merge
    consecutive statements from the same speaker

    Args:
        conversation (dict or list): Sentence speaker mapping

    Returns:
        str: Formatted transcript by speaker
    """
    formatted_conversation = ""
    previous_speaker = None
    merged_text = ""

    print("\n\nFormatting Speaker Conversion")
    # {'speaker': 1, 'start_time': 0.3, 'end_time': 0.675, 'text': 'hello'}
    print(conversation)
    for message in conversation:
        current_speaker = message["speaker"]
        current_text = message["text"]

        # If the current speaker is the same as the previous speaker,
        # merge the text
        if current_speaker == previous_speaker:
            merged_text += " " + current_text.strip()
        else:
            # If there's already merged text from the previous speaker,
            # add it to the formatted conversation
            merged_text = merged_text.strip()
            if merged_text:
                formatted_conversation += f"Speaker {previous_speaker}: {merged_text}\n"

            # Start a new line for the current speaker
            previous_speaker = current_speaker
            merged_text = current_text

    # Append the last merged text to the conversation
    merged_text = merged_text.strip()
    if merged_text:
        formatted_conversation += f"Speaker {previous_speaker}: {merged_text}\n"

    print("\n\nSpeaker conversion format completed")
    return formatted_conversation.strip()


def extract_json(text):
    json_pattern = r"(\{.*\}|\[.*\])"
    match = re.search(json_pattern, text, re.DOTALL)
    return match.group(0) if match else None


def generate_key_analysis(transcript, max_retries=3):
    global llm

    PROMPT = """

    Generate a detailed customer interaction report in the exact JSON format
    provided below. Extract complete, relevant quotes (not partial or
    single-word responses) from both agent and customer conversations.


    {
        "Summary": { "Summary of the whole conversation": "" },
        "Entities": {
            Extract key entities from the transcript, ensuring inclusion of
            'Customer Name', 'Agent Name', 'Business ID/Name',
            'Location ID/Name', 'Email ID', 'Concern', and any contact numbers.
            If a required field is missing, return it as 'NA'. Also, capture any
            other important entities present
        },
        "Key Topics": [
            List of key topics discussed during the conversation (provide only
            names)
        ],
        "Call Intent": "Intent of the call in one or two words dont include Unknown always give some intent for the calls",
        "Customer_name": "Extract the name of the Customer if present in the
        conversation otherwise return Unknown",
        "Speaker": {
            "Based on the conversation identify who is agent and customer": {
                Determine the role of each speaker in the conversation as either
                'agent' or 'customer'. If a speaker is explicitly associated
                with UrbanPiper or OrderMark, they are the agent. If not
                explicitly mentioned, determine the agent based on the content:
                agents typically provide support, ask clarifying questions, or
                help resolve issues.
                Handle any number of speakers dynamically based on the input
                conversation.
                "speaker_0": "agent/customer",
                "speaker_1": "agent/customer",
            }
        },
        "Overall_sentiment": {
            "Agent": "(Positive/Negative/Neutral)",
            "Customer": "(Positive/Negative/Neutral)",
            "call_sentiment": "(Positive/Negative/Neutral)"
        },
        "Issue": { Identify Main issue, and secondary issue if appeared },
        "First_call_resolution": "(Yes/No/Partial/Follow-up)",
        "customer_metrics": [
            {
                "name": "CSAT",
                "description": "Measures customer satisfaction from post-interaction surveys.",
                "score": "out of 10 (e.g., 4.5)",
                "formula": "(Positive experience - Negative experience) / Total interaction points * 10",
                "explanation": "Quote customer satisfaction indicators and explain score based on specific interactions"
            },
            {
                "name": "NPS",
                "description": "Measures customer loyalty and likelihood of recommending the service.",
                "score": "out of 10 (e.g., 7.8)",
                "formula": "(Likelihood to recommend - Frustration signs) / Total sentiment * 10",
                "explanation": "Reference specific moments showing loyalty or dissatisfaction"
            },
            {
                "name": "CES",
                "description": "Assesses ease of interaction and issue resolution.",
                "score": "out of 10 (e.g., 6.3)",
                "formula": "(Ease of process - Complexity) / Total steps * 10",
                "explanation": "Quote moments showing process ease or difficulty"
            },
            {
                "name": "TSS",
                "description": "Evaluates satisfaction with a specific transaction.",
                "score": "out of 10 (e.g., 5.2)",
                "formula": "(Transaction satisfaction - Delays) / Total steps * 10",
                "explanation": "Evidence of transaction efficiency or issues"
            },
            {
                "name": "Greeting & Introduction",
                "description": "Measures politeness, tone, and clarity at the start of interaction.",
                "score": "out of 10 (e.g., 8.0)",
                "formula": "(Politeness + Clarity + Tone) / 3 * 10",
                "explanation": "Quote opening exchange and its effectiveness"
            },
            {
                "name": "Active Listening",
                "description": "Evaluates attentiveness and acknowledgment of concerns.",
                "score": "out of 10 (e.g., 7.4)",
                "formula": "(Acknowledgment + Response relevance + Attentiveness) / 3 * 10",
                "explanation": "Evidence of listening quality from transcript"
            },
            {
                "name": "Clear Communication",
                "description": "Assesses clarity and professionalism in responses.",
                "score": "out of 10 (e.g., 6.9)",
                "formula": "(Clarity + Simplicity + Professionalism) / 3 * 10",
                "explanation": "Quote examples of clarity or confusion"
            },
            {
                "name": "Problem Solving",
                "description": "Measures effectiveness in issue resolution.",
                "score": "out of 10 (e.g., 9.1)",
                "formula": "(Resolution speed + Solution effectiveness + Satisfaction) / 3 * 10",
                "explanation": "Evidence of solution quality and timing"
            }
        ],
        "Recommendation": [
            Provide 5 ways to improve agent performance based on script
            adherence
        ],
        "Missed Out Opportunities": [
            List 5 opportunities the agent missed during the call
        ],
        "Additional Suggestion": [
            Suggest up to 7 improvements for a better conversation
        ]
    }

    Instructions:

    - Cover all categories using the given structure.
    - Provide full spoken sentences for quotes, ensuring depth and contextual
    accuracy.
    - Avoid short replies (e.g., "okay," "yes") and generic pleasantries in
    Personalization.
    - Each explanation should analyze what the agent did or missed, justifying
    the score.
    - Do not include "description" and "formula" in the output for
    "customer_metrics".
    - In Rephrase_Paraphrase and Understanding_Empathy_Evaluation, include both
    the customer’s statement and the agent’s response.
    - For Probing_Procedures, include multiple relevant quotes when necessary.
    - Validate the Json structure Before output.
    - Output only the JSON object with no additional commentary.
    - Maintain the exact JSON structure and key names.
    """
    key_analysis = None

    messages = [
        ("system", PROMPT),
        ("human", f"Full Transcript:\n{transcript}"),
    ]

    for attempt in range(max_retries):
        print(f"\n\nSending *Key Analysis* prompt to OpenAI model (Attempt {attempt + 1}/{max_retries})")
        try:
            ai_response = llm.invoke(messages)
            ai_response_content = ai_response.content

            ai_response_content = extract_json(ai_response_content)

            # Check if JSON extraction was successful
            if ai_response_content is None:
                raise ValueError("No JSON found in response")

            key_analysis = json.loads(ai_response_content)
            print("\n\nKey Analysis:")
            print(key_analysis)

            # If successful, break out of retry loop
            return key_analysis

        except Exception as e:
            print(f"**ERROR** (Attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}")

            # If this was the last attempt, return None
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts. Returning None.")
                return None

            # Otherwise, wait a bit before retrying (optional)
            print("Retrying...")
            # Uncomment the line below if you want to add a delay between retries
            # import time; time.sleep(1)

    return key_analysis

def generate_statement_analysis(transcript, max_retries=3):
    global llm

    PROMPT = """
    Analyze the call transcription and score the script adherence based on the following criterias .The response should be in exact json that i have provided below their should be no change in format.
{
    "script_adherence": {
        "opening_statement": {
            "greeting": {
                "criteria": "The agent introduces themselves using the company-approved script (e.g., 'Thank you for calling UrbanPiper. My name is Alex. How may I help you today?').",
                "score_evaluation": {
                    "2": "Followed the exact greeting and sounded warm and approachable.",
                    "1": "Missed minor details (e.g., slight script variation or tone issues).",
                    "0": "Did not follow the greeting script or sounded impersonal."
                },
                "quote": "Quote for greeting from the call.",
                "explanation": "Explanation for greeting adherence based on score evaluation.",
                "score": "Score based on score evaluation."
            },
            "introduction": {
                "criteria": "Agent must confirm the customer's identity (e.g., 'May I ask who I am speaking with?' or 'Am I speaking with Rachit?').",
                "score_evaluation": {
                    "2": "Properly asked for the customer's name and brand name.",
                    "1": "Asked for one but not both.",
                    "0": "Did not confirm identity."
                },
                "quote": "Quote for introduction.",
                "explanation": "Explanation for the introduction score.",
                "score": "Score based on score evaluation."
            },
            "assistance_offer": {
                "criteria": "Agent asking how they can help (e.g., 'How may I help you today?').",
                "score_evaluation": {
                    "1": "Offer of assistance was clear and appropriately phrased.",
                    "0": "No offer or unclear phrasing."
                },
                "quote": "Quote offering assistance.",
                "explanation": "Explanation for assistance offer.",
                "score": "Score based on score evaluation."
            },
            "personalisation": {
                "criteria": "Address the merchant by name rather than using Sir/Madam on call; probe for the name if required. Pleasantries to be used and personalize the call throughout the conversation / feel-good statements.",
                "score_evaluation": {
                    "3": "The agent consistently used the merchant's name and personalized the conversation appropriately. The agent asked for the name if necessary and used feel-good statements.",
                    "2": "The agent used the merchant's name but occasionally reverted to 'Sir/Madam' or did not fully personalize the interaction, and used feel-good statements.",
                    "1": "The agent asked for the name but used it infrequently, relying mostly on 'Sir/Madam' or general terms, and used feel-good statements.",
                    "0": "The agent did not use the merchant's name and only used 'Sir/Madam' throughout the conversation, and did not use any feel-good statements."
                },
                "quote": "Quote for addressing the customer by name.",
                "explanation": "Explanation of how the agent addressed the merchant and personalized the call.",
                "score": "Score based on addressing the customer by name evaluation.",
                "quote_pleasantries": "Quote demonstrating use of pleasantries.",
                "explanation_pleasantries": "Explanation of how the agent used pleasantries and feel-good statements to personalize the interaction.",
                "score_pleasantries": "Score based on using pleasantries and feel-good statements."
            }
        },
        "rephrase_paraphrase": {
            "criteria": "The agent should rephrase/paraphrase the concern to display an understanding of the issue/request, followed by the steps taken to resolve the issue.",
            "score_evaluation": {
                "1.5": "The agent effectively paraphrased the customer's concern, clearly demonstrating an understanding of the issue/request.",
                "0": "The agent did not paraphrase at all."
            },
            "quote": "Paraphrasing of what the customer told and what the agent paraphrased.",
            "explanation": "Explanation of how the agent paraphrased or rephrased the statement."
        },
        "understanding_empathy_apology_evaluation": {
            "criteria": "The agent should demonstrate a clear understanding of the merchant's issue by accurately paraphrasing or restating it. Additionally, the agent should express empathy and offer an apology at the first available opportunity, if needed.",
            "score_evaluation": {
                "1 to 5": "The agent effectively understood and acknowledged the merchant's issue by restating or summarizing it clearly, and demonstrated empathy and apologized at the correct moment.",
                "0": "The agent did not effectively understand or acknowledge the merchant's issue, and failed to express empathy or apologize where needed."
            },
            "quote": {
                "customer": "Paraphrase the merchant's concern or issue as stated by the merchant.",
                "agent": "Paraphrase how the agent demonstrated understanding, including any empathy or apologies provided."
            },
            "explanation": "Explanation of how the agent understood, empathized, or apologized.",
            "score": "Score based on score evaluation."
        },
        "acknowledgment_assurance_evaluation": {
            "criteria": "The agent should acknowledge the merchant's issue and provide assurance statements to demonstrate that the issue is being taken seriously and will be resolved.",
            "score_evaluation": {
                "1 to 5": "The agent effectively acknowledged the issue and used appropriate assurance statements to calm the merchant's concerns and provide confidence in a resolution.",
                "0": "The agent did not acknowledge the issue or failed to use appropriate assurance statements."
            },
            "quote_explanation": {
                "explanation": "Explain how the agent acknowledged the issue and whether they used appropriate acknowledgment statements.",
                "customer": "Paraphrase the merchant's concern or issue as stated by the merchant.",
                "agent": "Paraphrase how the agent acknowledged the issue and provided assurance statements."
            },
            "quote_assurance": {
                "explanation": "Describe how the agent used assurance statements to address the merchant's issue.",
                "customer": "Paraphrase the merchant's concern or issue as stated by the merchant.",
                "agent": "Paraphrase how the agent used assurance statements."
            },
            "score": "Score based on score evaluation."
        }
    },
    "call_closure_procedures": {
        "criteria": "The agent should follow the prescribed script to end the call, including reconfirming the resolution of the issue, probing for further assistance needs, and mentioning the IVR survey.",
        "score_evaluation": {
            "5.0": "The agent reconfirmed the resolution, probed for further assistance (e.g., 'Is there anything else I can help you with?'), and mentioned the IVR survey (e.g., 'After the call, there will be an IVR survey; please rate us there.').",
            "3.75": "The agent probed for further assistance needs (e.g., 'Is there anything else I can help you with?') and reconfirmed the resolution (e.g., 'The agent should say something like, 'I have now resolved your issue with the missing item. Is everything clear on your end?' or 'Just to confirm, we’ve processed the refund for the missing item and it should reflect in your account in 3-5 days.').",
            "2.50": "The agent mentioned the IVR survey and probed for further assistance needs (e.g., 'Is there anything else I can help you with?').",
            "1.25": "The agent only mentioned the IVR survey (e.g., 'After the call, there will be an IVR survey; please rate us there.').",
            "0": "The agent did not follow the call closure procedure and ended with just 'Thank you' or 'Goodbye'."
        },
        "quote": "The agent should use specific phrases for closing the call, including confirming resolution, probing for additional help, and mentioning the IVR survey.",
        "explanation": "This section should detail how the agent handled the call closure, including if they reconfirmed the resolution, if they asked if the customer needed further assistance, and if they mentioned the IVR survey. The score is based on whether the agent followed all these steps appropriately."
    },
    "hold_procedures": {
        "criteria": "The advisor must follow the prescribed hold script if a hold occurs, including seeking approval before placing the customer on hold, informing of the hold duration (2 minutes), and thanking the customer for staying connected upon resuming the call. The advisor should not place the caller on uninformed mute or hold, and the hold should be refreshed within 1 minute if more time is needed. The advisor should not be on mute and should ensure that the customer can hear the hold music.",
        "score_evaluation": {
            "5": "The hold did not happen in the conversation, or the advisor followed all the prescribed hold procedures perfectly, including seeking approval, informing the customer of the duration, refreshing the hold if needed, and thanking the customer upon resuming the call.",
            "3.32": "The hold occurred, and the advisor followed the hold procedure but missed one or more criteria, such as not refreshing the hold within 1 minute or failing to properly thank the customer upon resuming the call.",
            "1.66": "The hold occurred, and the advisor placed the caller on uninformed hold or mute, did not inform the customer of the hold duration, or failed to thank the customer upon resuming the call.",
            "0": "The hold occurred, and the advisor did not follow any part of the hold procedure, including placing the caller on uninformed hold or mute, not informing of the duration, and not thanking the customer."
        },
        "quote": "Script for placing the customer on hold, including seeking approval, informing of the hold duration, and thanking the customer upon resuming the call.",
        "explanation": "Explanation of how the advisor followed the hold procedures, including seeking approval, informing the customer of the hold duration, refreshing the hold if needed, and thanking the customer upon resuming the call.",
        "score": "Score based on score evaluation."
    },
    "call_flow_procedures": {
        "criteria": "The advisor should maintain a smooth flow during the call by avoiding unnecessary pauses, fumbling, dead air, or prolonging the call without reason. Jargon, stammering, and technical or internal terms should be avoided. The advisor should not make the merchant repeat themselves and should practice active listening without interrupting the merchant, allowing them to fully explain their issue.",
        "score_evaluation": {
            "5": "The advisor maintained a smooth and professional call flow, avoiding all unnecessary pauses, fumbling, dead air, jargon, stammering, and technical/internal terms. The merchant did not have to repeat themselves, and the advisor actively listened without interrupting.",
            "3.32": "The advisor generally followed the call flow but had some minor instances of unnecessary pauses, jargon, or requiring the merchant to repeat themselves.",
            "1.66": "The advisor frequently had dead air, stammered, used technical/internal terms, or caused the merchant to repeat themselves. Active listening was inconsistent.",
            "0": "The advisor did not follow the call flow, with frequent interruptions, fumbling, dead air, and a lack of active listening, making the merchant repeat themselves."
        },
        "quote": "Script for maintaining proper call flow, including avoiding jargon, stammering, dead air, and ensuring active listening.",
        "explanation": "Explanation of how the advisor managed the call flow, focusing on maintaining smooth communication, avoiding jargon, and actively listening to the merchant.",
        "score": "Score based on score evaluation."
    },
    "probing_procedures": {
        "definition": "Probing refers to the agent's ability to ask strategic and purposeful questions to fully understand the customer's issue. Effective probing involves asking relevant questions to gather detailed information without repeating or requesting information that is already available.",
        "criteria": "The agent should use effective probing techniques by asking clear, relevant, and purposeful questions that address the customer's issues. Questions should avoid redundancy and should not request information already available or previously provided.",
        "quote": {
            "customer_main_issue": "Customer's primary statement describing the main issue that led to the call.",
            "customer_secondary_issue": "Any additional or secondary issues mentioned by the customer during the call.",
            "agent_probing": [
                "Agent's probing question 1: A relevant question aimed at clarifying or gathering more information about the main issue.",
                "Agent's probing question 2: A follow-up question addressing any additional details related to the main issue.",
                "Agent's probing question 3: A question to clarify or gather information about any secondary issues mentioned.",
                "Agent's probing question 4: A question aimed at resolving any remaining concerns or verifying details."
            ]
        },
        "explanation": "Explanation of how the agent used probing techniques to understand and address the customer's issues and ask relevant question for understanding the issue and resolving it. This should highlight how the agent avoided redundant or irrelevant questions and focused on gathering necessary information to resolve the issues effectively.",
        "score": "Numerical score based on the effectiveness of the probing, with a maximum score of 10.",
        "score_evaluation": {
            "10": "The agent used optimal probing techniques, asking only relevant questions that were necessary to understand and resolve the customer's main and secondary issues. There was no redundancy, and all questions were purposeful and directly related to the issues at hand.",
            "7.5": "The agent mostly used effective probing techniques but included a few irrelevant or redundant questions. The probing was generally relevant and addressed the customer's main and secondary issues with minor areas for improvement.",
            "5": "The agent asked several irrelevant or redundant questions, including requests for information already available or previously provided. Despite these issues, the agent managed to gather some relevant information to address the customer's main and secondary issues.",
            "2.5": "The agent struggled with effective probing, asking numerous irrelevant or redundant questions. Many questions were not related to the customer's issues or repeated information already provided, making it difficult to fully understand and resolve the concerns.",
            "0": "The agent failed to use effective probing techniques, asking questions that were irrelevant, redundant, or unrelated to the customer's issues. The probing did not help in understanding or resolving the customer's main and secondary concerns."
        }
    },
    "ownership": {
        "definition": "Ownership involves the agent’s proactive management of the call, including using necessary tools, verifying resolution feasibility, providing additional information, and keeping the customer informed. Tools like Anydesk should be used if necessary to resolve the issue effectively.",
        "criteria": "The agent should effectively request tools like Anydesk if deemed necessary, check feasibility with L2 teams, take extra actions beyond basic resolution, provide relevant updates, maintain call control, and inform the customer about timelines.",
        "quote": {
            "request_tool": "Did the agent request tools like Anydesk if necessary to understand or resolve the issue?",
            "feasibility_check": "Did the agent check with the L2 team before committing to solutions or timelines?",
            "extra_action": "Did the agent go beyond basic resolution expectations with additional efforts?",
            "additional_info": "Did the agent provide necessary updates or information about recent changes?",
            "call_control": "Did the agent maintain control over the call and address concerns promptly?",
            "TAT_information": "Did the agent inform the customer about the expected turnaround time (TAT), if known?"
        },
        "explanation": "Evaluate how well the agent demonstrated ownership, including the use of tools if necessary, feasibility checks, extra efforts, information provision, call control, and TAT communication.",
        "score": "Score out of 10 based on overall ownership demonstrated.",
        "score_evaluation": {
            "10": "Exceptional ownership with effective use of tools (if necessary), thorough feasibility checks, extra efforts, clear information, strong call control, and accurate TAT communication.",
            "7.5": "Strong ownership with minor gaps in tool usage (if necessary), feasibility checks, extra effort, information provision, or TAT communication.",
            "5": "Basic ownership with several gaps in tool usage (if necessary), feasibility checks, extra efforts, information provision, call control, or TAT communication.",
            "2.5": "Limited ownership with many gaps in tool usage (if necessary), feasibility checks, extra efforts, information provision, call control, or TAT communication.",
            "0": "No effective ownership; poor tool usage (if necessary), feasibility checks, extra effort, information provision, call control, and TAT communication."
        }
    },
    "objection_handling": {
        "definition": "Handling customer objections effectively involves providing clear, layman-friendly explanations and confidently navigating the system during remote assistance sessions like Anydesk. Agents should address customer concerns with appropriate reasoning and ensure comprehensive understanding of the issue or request.",
        "criteria": "The agent should effectively handle objections, provide clear and understandable explanations, be confident while navigating the system (especially during Anydesk sessions), and ensure the customer understands the reasoning behind the issue or request.",
        "quote": {
            "customer_statement": "What did the customer say regarding their objections or concerns?",
            "agent_response": "What did the agent reply to address the customer's objections or concerns?",
            "clear_explanation": "Did the agent explain concepts from a layman’s perspective effectively? Was the explanation clear and easy to understand?",
            "effective_rebuttal": "Did the agent address and handle customer objections effectively with appropriate reasoning?",
            "confidence_in_navigation": "Did the agent demonstrate confidence while navigating the system, particularly during Anydesk sessions?",
            "detailed_reasoning": "Did the agent provide thorough and detailed reasoning regarding the issue or request?"
        },
        "explanation": "Evaluate how well the agent handled customer objections and responded to concerns. Assess the clarity of the agent's explanations, their confidence in system navigation, and the depth of reasoning provided. Consider both what the customer said and how the agent addressed those points.",
        "score": "Score out of 10 based on overall effectiveness in handling objections and providing explanations.",
        "score_evaluation": {
            "10": "Exceptional handling of objections with clear, layman-friendly explanations, confident system navigation, and detailed reasoning.",
            "7.5": "Strong handling of objections with minor gaps in explanation clarity, confidence, or detail.",
            "5": "Basic handling of objections with several gaps in clarity, confidence, or detail in explanations.",
            "2.5": "Limited handling of objections with many gaps in explanation clarity, confidence, or detail.",
            "0": "No effective handling of objections; poor explanations, lack of confidence, and insufficient detail."
        }
    },
    "escalations": {
        "definition": "Escalations involve transferring issues that cannot be resolved immediately to a higher level of support or specialized team. This ensures that complex or high-priority issues receive appropriate attention and expertise.",
        "criteria": "The agent should effectively escalate issues when needed, involving the right parties, ensuring proper documentation, and communicating clearly with the customer about next steps.",
        "quote": {
            "supervisor_contact": "Did the agent escalate the issue to a supervisor when necessary?",
            "manager_callback": "Did the agent request a callback from a manager if required?",
            "higher_authority": "Did the agent escalate to a higher authority when appropriate?",
            "expert_handling": "Did the agent involve technical experts or specialized teams for complex issues?",
            "complaint_logging": "Did the agent properly log and track formal complaints?",
            "legal_threats": "Did the agent escalate legal threats to the legal department promptly?",
            "priority_flagging": "Did the agent flag high-priority issues appropriately and take immediate action?"
        },
        "explanation": "Evaluate how well the agent managed escalations, including involving the necessary parties, documenting complaints, addressing legal threats, and handling high-priority issues. If escalation was not necessary, indicate N/A.",
        "score": "Score out of 10 based on the effectiveness of escalation management. If escalation did not occur, the score should be marked as N/A.",
        "score_evaluation": {
            "10": "Exceptional escalation handling with timely involvement of supervisors, experts, proper complaint logging, and effective management of legal threats and high-priority issues.",
            "7.5": "Strong escalation handling with minor gaps in involving the right parties, logging complaints, or managing urgent issues.",
            "5": "Basic escalation handling with several gaps in involving necessary teams, logging complaints, or addressing high-priority issues.",
            "2.5": "Limited escalation handling with many gaps in involving the right people, documenting complaints, or managing urgent matters.",
            "0": "No effective escalation management; failure to involve necessary parties, log complaints, or address legal threats and high-priority issues.",
            "N/A": "Escalation did not occur as it was not necessary for the call."
        }
    },
    "Positive Movements": {
        "Acknowledgement and Empathy": {
            "Empathetic Statements": {
                "criteria": "Assess how well the agent used empathetic statements to connect with the customer's feelings. Look for distinct statements that demonstrate genuine understanding.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote reflecting a specific empathetic statement.",
                "Explanation": "Explanation of how this statement shows empathy."
            },
            "Sympathetic Tone": {
                "criteria": "Evaluate the agent’s tone throughout the call to ensure it was consistently sympathetic and aligned with the customer's feelings.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showcasing a sympathetic tone.",
                "Explanation": "Explanation of the tone's effectiveness."
            },
            "Personalized Responses": {
                "criteria": "Assess how well the agent personalized their responses to the customer's specific situation. Ensure that responses avoid generic phrases.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote reflecting a personalized response.",
                "Explanation": "Explanation of how the response was tailored."
            }
        },
        "Active Listening": {
            "Paraphrasing": {
                "criteria": "Assess the agent’s use of paraphrasing to confirm understanding of the customer's statements and concerns. Each paraphrase should be unique.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique paraphrase demonstrating understanding.",
                "Explanation": "Explanation of how the paraphrase reflects active listening."
            },
            "Affirmative Acknowledgement": {
                "criteria": "Evaluate the agent's use of affirmative acknowledgments to show understanding and agreement with the customer's concerns.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing affirmative acknowledgment.",
                "Explanation": "Explanation of how this acknowledgment was effective."
            },
            "Non-Verbal Cues (where applicable)": {
                "criteria": "Assess the use of non-verbal cues (e.g., tone of voice, pauses) to indicate active listening. Ensure quotes reflect specific cues.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Quote demonstrating effective non-verbal cues.",
                "Explanation": "Explanation of how these cues contributed to the interaction."
            }
        },
        "Positive Language": {
            "Encouraging Words": {
                "criteria": "Evaluate how the agent used encouraging language to support and motivate the customer.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote with encouraging language.",
                "Explanation": "Explanation of how this language motivated the customer."
            },
            "Constructive Feedback": {
                "criteria": "Assess the agent’s use of constructive feedback to guide the customer towards a positive outcome.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote demonstrating constructive feedback.",
                "Explanation": "Explanation of the feedback's impact."
            },
            "Reassurance": {
                "criteria": "Evaluate how the agent provided reassurance to alleviate the customer's concerns or anxieties.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing reassurance.",
                "Explanation": "Explanation of how reassurance was provided."
            }
        },
        "Problem Identification": {
            "Accurate Diagnosis": {
                "criteria": "Assess how accurately the agent diagnosed the customer’s issue, including alignment with the customer’s description.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote illustrating accurate diagnosis.",
                "Explanation": "Explanation of how the diagnosis aligned with the customer's description."
            },
            "Probing Questions": {
                "criteria": "Evaluate the effectiveness of the probing questions used by the agent to gather information and clarify the problem.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote reflecting effective probing.",
                "Explanation": "Explanation of how probing helped clarify the issue."
            },
            "Contextual Understanding": {
                "criteria": "Assess how well the agent understood the context and relevant background information of the customer's issue.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing contextual understanding.",
                "Explanation": "Explanation of the agent's understanding of the context."
            }
        },
        "Solution Offering": {
            "Clear Solutions": {
                "criteria": "Evaluate how clearly the agent presented solutions to the customer’s issue, including clarity and comprehensibility.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote presenting a clear solution.",
                "Explanation": "Explanation of how the solution was effectively communicated."
            },
            "Alternative Options": {
                "criteria": "Assess the agent’s ability to offer alternative solutions when the initial solution is not feasible.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote offering alternative solutions.",
                "Explanation": "Explanation of how alternatives were provided."
            },
            "Detailed Instructions": {
                "criteria": "Evaluate how well the agent provided detailed, step-by-step instructions for implementing the solution.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote providing detailed instructions.",
                "Explanation": "Explanation of the clarity of the instructions."
            }
        },
        "Thanking the Customer": {
            "Gratitude for Patience": {
                "criteria": "Assess how well the agent expressed gratitude for the customer’s patience during the call.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote expressing gratitude.",
                "Explanation": "Explanation of how gratitude was conveyed."
            },
            "Appreciation for Feedback": {
                "criteria": "Evaluate how the agent showed appreciation for the customer’s feedback.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing appreciation for feedback.",
                "Explanation": "Explanation of the importance of customer feedback."
            },
            "General Thanks": {
                "criteria": "Assess how the agent thanked the customer for their time and interaction overall.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote of thanks.",
                "Explanation": "Explanation of how the thanks were given."
            }
        },
        "Quick Response Time": {
            "Immediate Acknowledgement": {
                "criteria": "Evaluate how promptly the agent acknowledged the customer’s issue or request at the start of the call.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing immediate acknowledgment.",
                "Explanation": "Explanation of the promptness of the acknowledgment."
            },
            "Swift Action": {
                "criteria": "Assess how quickly the agent took action to address the customer’s issue after initial acknowledgment.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote reflecting swift action.",
                "Explanation": "Explanation of the speed of action taken."
            },
            "Fast Issue Resolution": {
                "criteria": "Evaluate how quickly the agent resolved the customer’s issue or provided a solution after taking action.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote demonstrating fast resolution.",
                "Explanation": "Explanation of how quickly the issue was resolved."
            }
        },
        "Personalization": {
            "Using Customer's Name": {
                "criteria": "Assess how frequently and appropriately the agent used the customer's name during the call.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote using the customer's name.",
                "Explanation": "Explanation of how personalization was achieved."
            },
            "Acknowledging Customer History": {
                "criteria": "Evaluate how well the agent acknowledged the customer's history or previous interactions with the company.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote acknowledging customer history.",
                "Explanation": "Explanation of the relevance of past interactions."
            },
            "Tailored Solutions": {
                "criteria": "Assess how well the agent provided solutions that were specifically tailored to the customer’s needs and history.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote reflecting tailored solutions.",
                "Explanation": "Explanation of how the solutions were customized."
            }
        }
    },
    "Negative Movements": {
        "Interruptions": {
            "Cutting Off": {
                "criteria": "Evaluate how often the agent interrupted the customer. Each quote should highlight a distinct interruption.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote illustrating a specific interruption.",
                "Explanation": "Explanation of the impact of this interruption."
            },
            "Speaking Over": {
                "criteria": "Assess how often the agent spoke over the customer, talking over their responses.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing the agent speaking over the customer.",
                "Explanation": "Explanation of how this affected the conversation."
            },
            "Ignoring Customer Input": {
                "criteria": "Evaluate whether the agent ignored or did not acknowledge the customer’s input during the call.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing ignored input.",
                "Explanation": "Explanation of how this impacted the customer."
            }
        },
        "Incorrect Information": {
            "Providing Wrong Details": {
                "criteria": "Assess how often the agent provided incorrect or inaccurate details to the customer.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote providing incorrect details.",
                "Explanation": "Explanation of how this affected the conversation."
            },
            "Misleading Statements": {
                "criteria": "Evaluate if the agent made any misleading statements that could confuse the customer.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing a misleading statement.",
                "Explanation": "Explanation of the impact of misleading information."
            }
        },
        "Lack of Empathy": {
            "Indifference": {
                "criteria": "Evaluate if the agent showed indifference towards the customer's situation or concerns.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing indifference.",
                "Explanation": "Explanation of how indifference was demonstrated."
            },
            "Dismissive Language": {
                "criteria": "Assess whether the agent used dismissive language that minimized or ignored the customer’s concerns.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote with dismissive language.",
                "Explanation": "Explanation of the implications of dismissive language."
            },
            "Insensitive Remarks": {
                "criteria": "Evaluate if the agent made any remarks that were insensitive to the customer’s situation.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote containing insensitive remarks.",
                "Explanation": "Explanation of how these remarks affected the customer."
            }
        },
        "Long Hold Times": {
            "Extended Wait": {
                "criteria": "Assess how long the customer had to wait during holds.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote regarding extended wait times.",
                "Explanation": "Explanation of how wait times impacted the experience."
            },
            "Repeated Holds": {
                "criteria": "Assess if the customer was placed on hold multiple times and the impact on the experience.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote reflecting repeated holds.",
                "Explanation": "Explanation of the effect of repeated holds."
            }
        },
        "Unresolved Issues": {
            "Failure to Solve": {
                "criteria": "Evaluate if the agent failed to resolve the customer's issue by the end of the call.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote indicating unresolved issues.",
                "Explanation": "Explanation of the unresolved nature of the issues."
            },
            "Incomplete Solutions": {
                "criteria": "Assess if the solutions provided by the agent were incomplete or did not fully address the issue.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing incomplete solutions.",
                "Explanation": "Explanation of how the solutions fell short."
            },
            "Lack of Follow-Through": {
                "criteria": "Evaluate if the agent followed through on promises or commitments made to the customer.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing lack of follow-through.",
                "Explanation": "Explanation of the importance of follow-through."
            }
        },
        "Negative Language": {
            "Use of No/Not": {
                "criteria": "Assess how the agent used negative language, such as 'no' or 'not,' and its impact.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote demonstrating negative language.",
                "Explanation": "Explanation of how negative language affected the interaction."
            },
            "Blame Language": {
                "criteria": "Evaluate if the agent used language that placed blame on the customer or others.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing blame language.",
                "Explanation": "Explanation of the implications of blame language."
            },
            "Defensive Responses": {
                "criteria": "Assess if the agent provided defensive responses instead of addressing concerns openly.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote reflecting defensive responses.",
                "Explanation": "Explanation of how defensiveness affected the conversation."
            }
        }
    },
    "Neutral Movements": {
        "Clarification Requests": {
            "Detailed Clarification": {
                "criteria": "Evaluate how effectively the agent requested detailed clarification to understand the issue.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote requesting clarification.",
                "Explanation": "Explanation of how this request aided understanding."
            },
            "Confirming Details": {
                "criteria": "Assess how well the agent confirmed the details provided by the customer.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote confirming details.",
                "Explanation": "Explanation of the importance of confirming details."
            },
            "Probing for More Information": {
                "criteria": "Evaluate how effectively the agent probed for additional information to fully understand the customer’s needs.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote probing for more information.",
                "Explanation": "Explanation of how probing enhanced understanding."
            }
        },
        "Status Updates": {
            "Ongoing Updates": {
                "criteria": "Assess if the agent provided ongoing updates about the status of the customer's request.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote providing updates.",
                "Explanation": "Explanation of how updates informed the customer."
            },
            "Process Transparency": {
                "criteria": "Evaluate how transparent the agent was about the process involved in addressing the customer’s issue.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote reflecting process transparency.",
                "Explanation": "Explanation of how transparency affected trust."
            },
            "Expected Resolution Time": {
                "criteria": "Assess if the agent provided an expected resolution time for resolving the customer's issue.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote giving expected resolution time.",
                "Explanation": "Explanation of how resolution timing was communicated."
            }
        },
        "Information Gathering": {
            "Verification Questions": {
                "criteria": "Evaluate how effectively the agent used verification questions to confirm the customer’s identity.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote with verification questions.",
                "Explanation": "Explanation of the role of verification."
            },
            "Detail Collection": {
                "criteria": "Assess how well the agent collected detailed information needed to address the customer’s issue.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote collecting details.",
                "Explanation": "Explanation of the importance of detail collection."
            },
            "Contextual Questions": {
                "criteria": "Evaluate how well the agent asked contextual questions to understand the background of the customer’s issue.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote with contextual questions.",
                "Explanation": "Explanation of how context improved understanding."
            }
        },
        "Process Steps": {
            "criteria": "Assess if the agent effectively communicated the steps involved in resolving the customer's issue.",
            "Score": 0,
            "MaxScore": 5,
            "Quote": "Unique quote explaining process steps.",
            "Explanation": "Explanation of the clarity of communication."
        },
        "Neutral Tone": {
            "Balanced Communication": {
                "criteria": "Evaluate if the agent maintained a balanced communication style without showing bias.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote reflecting balanced communication.",
                "Explanation": "Explanation of the importance of neutrality."
            },
            "Objective Responses": {
                "criteria": "Assess if the agent provided objective and fact-based responses.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote demonstrating objectivity.",
                "Explanation": "Explanation of how objectivity aided the conversation."
            }
        },
        "Fact-Finding": {
            "Detailed Inquiry": {
                "criteria": "Assess how thoroughly the agent inquired about the customer's issue to gather all relevant facts.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote showing detailed inquiry.",
                "Explanation": "Explanation of the thoroughness of the inquiry."
            },
            "Context Gathering": {
                "criteria": "Evaluate how effectively the agent gathered context about the customer's issue.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote gathering context.",
                "Explanation": "Explanation of the significance of context."
            },
            "Symptom Identification": {
                "criteria": "Assess how well the agent identified and addressed the symptoms of the customer’s issue.",
                "Score": 0,
                "MaxScore": 5,
                "Quote": "Unique quote reflecting symptom identification.",
                "Explanation": "Explanation of the identification process."
            }
        },
        "Acknowledgement of Request": {
            "Request Confirmation": {
                "criteria": "Evaluate if the agent confirmed the customer’s request accurately.",
                "Score": 0,
                "MaxScore": 5,
                "quote": {
                    "customer": "Unique quote from customer.",
                    "agent": "Unique quote from agent."
                },
                "Explanation": "Explanation of how confirmation was handled."
            },
            "Status Assurance": {
                "criteria": "Assess if the agent provided assurance about the status of the customer’s request.",
                "Score": 0,
                "MaxScore": 5,
                "quote": {
                    "customer": "Unique quote from customer.",
                    "agent": "Unique quote from agent."
                },
                "Explanation": "Explanation of how assurance was conveyed."
            },
            "Preliminary Acknowledgement": {
                "criteria": "Evaluate if the agent provided a preliminary acknowledgment of the customer’s request.",
                "Score": 0,
                "MaxScore": 5,
                "quote": {
                    "customer": "Unique quote from customer.",
                    "agent": "Unique quote from agent."
                },
                "Explanation": "Explanation of the acknowledgment process."
            }
        },
        "Customer Effort Reduction": {
            "Ease of Process": {
                "criteria": "Assess if the agent suggested ways to make the process easier for the customer.",
                "Score": 0,
                "MaxScore": 5,
                "quote": {
                    "customer": "Unique quote from customer.",
                    "agent": "Unique quote from agent."
                },
                "Explanation": "Explanation of how the process was simplified."
            },
            "Self-Service Options": {
                "criteria": "Evaluate if the agent recommended self-service options where applicable.",
                "Score": 0,
                "MaxScore": 5,
                "quote": {
                    "customer": "What customer said.",
                    "agent": "What agent said regarding self-service."
                },
                "Explanation": "Explanation of the self-service options discussed."
            },
            "Simplified Procedures": {
                "criteria": "Assess if the agent proposed simplified procedures to reduce customer effort.",
                "Score": 0,
                "MaxScore": 5,
                "quote": {
                    "customer": "What customer said.",
                    "agent": "What agent said to simplify the process."
                },
                "Explanation": "Explanation of how procedures were simplified."
            }
        }
    }
}


Instructions:
- Maintain the exact JSON structure and all key names unchanged.
- Address every criterion using the provided structure.
- Use only integer values for scores unless a decimal (float) is specifically required. Scores must be within the range 0 to MaxScore.
- For each explanation, clearly evaluate the agent’s actions or omissions, and explain the reasoning behind the assigned score.
- Do not include "criteria" and "MaxScore" in the output.
- The output should be a valid JSON object only, with no additional text or commentary.
"""
    statement_analysis = None

    messages = [
        ("system", PROMPT),
        ("human", f"Full Transcript:\n{transcript}"),
    ]

    for attempt in range(max_retries):
        print(f"\n\nSending *Statement Analysis* prompt to OpenAI model (Attempt {attempt + 1}/{max_retries})")
        try:
            ai_response = llm.invoke(messages)
            ai_response_content = ai_response.content

            ai_response_content = extract_json(ai_response_content)

            # Check if JSON extraction was successful
            if ai_response_content is None:
                raise ValueError("No JSON found in response")

            statement_analysis = json.loads(ai_response_content)
            # statement_analysis = statement_analysis.get("script_adherence", None)
            print("\n\nCall Parameter Analysis:")
            print(statement_analysis)

            # If successful, break out of retry loop
            return statement_analysis

        except Exception as e:
            print(f"**ERROR** (Attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}")

            # If this was the last attempt, return None
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts. Returning None.")
                return None

            # Otherwise, wait a bit before retrying (optional)
            print("Retrying...")
            # Uncomment the line below if you want to add a delay between retries
            # import time; time.sleep(1)

    return statement_analysis


def generate_issue_tree_analysis(transcript):
    global llm

    PROMPT = """
    Analyze the call transcription and categorize any issues found using the
    provided classifications.

    Categories and Descriptions:
    1. Category: *Offboarding Request*:
        Issues related to discontinuation of service, either due to unavoidable
        circumstances or potentially preventable reasons. This includes requests
        driven by financial, operational, or product/service-related concerns.

        Sub Category - 1: *Non-avoidable Churn*:
          Customer exits that cannot be influenced, often due to external or
          logistical reasons.

          Sections:
          - *Aggregator Issue* (Normal): Refers to issues that arise due to the
          policies, systems, or downtimes of aggregator platforms (like Swiggy,
          Zomato).

          - *Business Shutdown* (Normal): Occurs when a merchant permanently or
          temporarily shuts down its business operations due to reasons such as
          financial crises, lack of viability, or regulatory compliance, etc.

          - *Outlet Shutdown* (Normal): A temporary or permanent closure of a
          specific outlet belonging to the merchant. Reasons may include
          staffing issues, local regulations, or internal business strategies,
          etc.

          - *POS Change* (Normal): This occurs when a merchant decides to change
          their Point of Sale (POS) system.

          - *POS_Issue tech* (Normal): Technical problems related to the
          integration or operation of the POS system with aggregator platform's
          services. These issues can involve system errors, network connectivity
          problems, or software malfunctions.

          - *POS_pricing issue* (Normal): Discrepancies in pricing data between
          the merchant’s POS system and the UrbanPiper platform. This can result
          in incorrect prices being displayed to customers, but the core issue
          lies with the POS configuration or sync issues.

          - *Temporary disintegration* (Normal): A temporary loss of integration
          between the merchant’s systems and platform. This could lead to the
          interruption of services such as order relays or inventory syncs, etc.

          - *UP_Nonpayment of Invoice* (Normal): Issues related to delays or
          failures in paying Urban Piper for its services.

        Sub Category - 2: *Avoidable Churn*:
          Customer exits that might be preventable with product adjustments,
          better service, or alternative solutions.

          Sections:
          - *Merchant's Product understanding* (Normal): These issues stem from
          the merchant’s inability to understand UrbanPiper's product offerings
          or features, leading to misconfigurations or poor usage of the
          platform.

          - *Product Feedback* (Normal): Feedback from merchants regarding the
          features or performance. This can include requests for new features,
          suggestions for improvements, or complaints about specific
          functionalities.

          - *UP_Feature Parity* (Normal): Occurs when merchants request feature
          parity between UrbanPiper's platform and competitors' offerings. This
          can arise when merchants feel they are losing business advantages due
          to missing or underdeveloped features.

          - *UP_Feature Requirement* (Normal): Merchants request new features or
          enhancements to existing functionalities in the system to better meet
          their business needs.

          - *Up_Issue_Product* (Normal): Issues related to UrbanPiper's product
          functionality, such as bugs, glitches, or unexpected behaviors that
          hinder the merchant’s use of the platform.

          - *Up_Issue_services* (Normal): Service-related problems experienced
          by merchants, including delays in support response times, poor service
          quality, or miscommunication, etc between Urban Piper and the merchant.

          - *UP_Moved to competitor* (Normal): Occurs when merchants opt to
          leave UrbanPiper and use a competitors services due to dissatisfaction
          with the product, pricing, or service.

    2. Category: *Issue*
        Problems directly affecting service functionality or user experience.
        Typically time-sensitive and critical for daily operations, requiring
        immediate resolution.

        Sub Category - 1: *Store Operations*:
          Operational issues affecting store functioning, such as device
          connectivity, POS configurations, or billing.

          Sections:
          - *Bill Sync Issue* (Critical): Occurs when there are discrepancies
          between the billing and the merchant’s POS system, leading to errors
          in the billing process.

          - *Floor type / table configuration* (Normal): Refers to issues with
          the configuration of floor layouts or table setups in the POS system,
          affecting the smooth flow of operations and billing.

          - *Issue with Register* (Critical): Problems with the merchant's
          register (cash register or POS system), such as software glitches or
          configuration errors that affect transactions.

          - *Unable to punch bill* (Critical): Merchants are unable to process
          or create bills in their system.

          - *Batch Production* (Normal): Issues related to the production of
          items in batches, affecting stock levels or availability on the
          platform.

          - *Purchase order* (Normal): Problems with the generation or
          processing of purchase orders in the prime inventory system,
          leading to stock or supply chain issues.

          - *Raw Materials* (Normal): Issues tracking or managing raw materials,
          which can lead to production or fulfillment problems.

          - *Recipes* (Normal): Issues tracking or managing raw materials within
          the Prime platform, which can lead to production or fulfillment
          problems.

          - *Stock Transfer* (Normal): Issues with transferring stock between
          locations or outlets, leading to inventory discrepancies or
          unavailability at specific locations, etc.

          - *iMin Device Issue (UK only)* (High): Issues specific to iMin
          devices used in the UK, such as connectivity or configuration problems
          that prevent order receipts from printing.

          - *Print Template* (High): Problems with the print templates used for
          generating order receipts, kitchen tickets, etc ,leading to incomplete
          or incorrect information.

          - *Printer Connectivity* (Critical): Merchants face connectivity
          issues between their POS system and the printer or just printer
          connectivity issues.

        Sub Category - 2: *Menu*:
          Issues affecting menu visibility, item details, or configuration
          errors in the system.

          Sections:
          - *Bulk action issue* (Normal): Merchants experience problems when
          attempting to perform bulk actions, such as updating multiple items or
          categories at once, etc.

          - *Item tags are not getting added* (Normal): Tags (e.g., 'Vegan',
          'Gluten-Free') are not properly applied to items, which limits the
          categorization and visibility of these items on the platform.

          - *Nutritional info not correct on UI* (Normal): The nutritional
          information displayed on the aggregator UI is inaccurate.

          - *Recommended Items not correct* (Normal): The recommended items
          feature displays incorrect or irrelevant items.

          - *Serves <n> value is not correct* (Normal): The 'serves X people'
          attribute is incorrectly displayed for items, leading to customer
          confusion about portion sizes.

          - *Unable to apply a discount on catalogue* (Normal): Merchants are
          unable to set or apply discounts to specific items or categories in
          their catalogue.

          - *Wrong sequence (Sort order) of items* (Normal): Items within a
          category are displayed in the wrong order.

          - *Wrong sequence (Sort order) of options* (Normal): Customization
          options for items are not shown in the correct order, which could
          confuse customers or reduce the likelihood of choosing add-ons, etc.

          - *Test / Staging Portal* (Normal): Refers to issues experienced in
          the test or staging environments where merchants test their
          configurations or updates before deploying to live systems.

          - *Category not showing* (High): A specific category of items is not
          visible, causing difficulties for customers trying to view or order
          products from that category, etc.

          - *Category timing not working* (High): Scheduled timings for specific
          categories (e.g., breakfast/lunch menus) are not functioning as
          expected.

          - *Discount not getting applied* (High): Discounts set by the merchant
          are not correctly applied at checkout.

          - *Discount not getting removed* (High): The discount, once applied,
          remains active even after it should have been removed or expired.

          - *Item description not updating* (High): Any changes made to the
          description of menu items by the merchant do not reflect on the
          aggregator platform, causing confusion or miscommunication with
          customers.

          - *Item images are not getting removed* (High): Merchants are unable
          to remove outdated or incorrect images from their menu items.

          - *Item images are not showing up* (High): Images for certain items
          are not visible in the interface.

          - *Item not showing* (High): Specific menu items do not appear on the
          UI, preventing customers from ordering those items.

          - *Menu validation error - Prism* (High): An error occurs during the
          menu validation process on the Prism system, preventing updates or
          changes to the menu from being successfully applied.

          - *Option not showing* (High): Customization options (e.g., size,
          flavor) for items are not visible on the platform.

          - *Request Timeout Error on Prime platform* (High): The system times
          out while trying to load or modify a catalogue item.

          - *Target Server Error* (High): A server-related error that prevents
          the catalogue from updating or functioning properly on the aggregator
          platform.

          - *Veg / Non-veg issue (food type)* (High): Mislabeling of food items
          as vegetarian or non-vegetarian on the platform.

          - *Wrong charges* (High): Incorrect charges are applied to items,
          either inflating or undercharging for items, leading to pricing
          discrepancies and customer complaints.

          - *Wrong item price* (High): Prices displayed for certain items on the
          aggregator platform do not match the intended pricing set by the
          merchant.

          - *Wrong option price* (High): Prices for options (such as additional
          toppings or sides) are incorrect.

          - *Wrong sequence (sort order) of category* (High): Categories are
          displayed in the wrong order on the platform.

          - *Received order for disabled item* (High): Item Inventory -
          Merchants receive orders for items that have been marked as 'disabled'
          or unavailable in their inventory.

          - *Unable to turn item 'On' / 'Off'* (High): Item Inventory -
          Merchants experience difficulties toggling items between 'available'
          and 'unavailable' in their inventory, affecting the items’ visibility
          on the aggregator platform.

          - *Received order for disabled option* (High): Option Inventory -
          Orders are placed for product options (e.g., sizes, toppings) that
          have been marked as disabled in the inventory.

          - *Unable to turn modifier 'On' / 'Off'* (High): Option Inventory -
          Merchants cannot toggle specific product modifiers (e.g., size,
          flavor) between available and unavailable.

          - *Order received after turning off store* (High): Despite the store
          being marked as 'closed', orders continue to be received, causing
          fulfillment problems.

          - *Blank Menu on aggregator UI* (Critical): The menu on the aggregator
          platform appears blank or does not load properly, preventing customers
          from viewing available items.

          - *Wrong taxes on aggregator UI* (Critical): Tax rates applied to
          items are incorrect on the aggregator platform.

          - *Store not visible on Aggregator UI* (Critical): The store does not
          appear on the aggregator's platform, making it impossible to place
          orders.

        Sub Category - 3: *Order*:
          Issues affecting the processing, accuracy, or management of orders,
          including problems with billing, notifications, and status updates.

          Sections:
          - *Order cancellation Request* (Normal): Merchants face issues
          receiving or processing customer requests for order cancellations.

          - *Rider Assignment Notification* (Normal): Merchants are not
          receiving timely or accurate notifications regarding rider assignments
          for delivery orders.

          - *Incorrect / missing charges on Orders* (Normal): Orders relayed to
          merchants have either incorrect or missing charges.

          - *Incorrect customer details* (Normal): The order relays incorrect or
          incomplete customer information (e.g., address, phone number).

          - *Incorrect discount on orders* (Normal): Discounts applied to
          customer orders are incorrect.

          - *Auto 'Food Ready' not working* (Normal): The system does not
          automatically update the order status to 'Food Ready' when it is
          prepared.

          - *Bulk order (additional rider) not getting updated* (Normal): The
          status or information for bulk orders, which require additional
          riders, is not updated properly in the system.

          - *Completed / Dispatch status is not getting updated* (Normal): The
          order status does not change to 'Completed' or 'Dispatched'.

          - *Order not getting marked 'Food Ready'* (Normal): The system does
          not mark an order as ready for pickup or delivery, leading to
          confusion about the order status.

          - *Order state mismatch between Aggregator and UrbanPiper* (Normal):
          There is a mismatch between the order status on the aggregator’s
          platform and Urban Piper’s system, leading to order processing errors.

          - *New Order Notification Sound* (High): The sound alert for new
          orders is not working or is inconsistent.

          - *Incorrect / missing Tax* (High): Tax information in the order
          details is incorrect or missing.

          - *Missing customization* (High): Customization details (e.g., extra
          toppings, special instructions) are missing from the relayed order,
          affecting order accuracy.

          - *Missing item* (High): An item from the customer's order is missing
          in the relayed data.

          - *Auto 'Acknowledge' not working* (High): The automatic
          acknowledgment of orders by the system is not functioning.

          - *Rider assignment failure* (High): The system fails to assign a
          rider to the order, causing delays in order fulfillment and delivery.

          - *Excessive Order Cancellations* (Critical): The merchant is
          experiencing a high volume of canceled orders.

          - *Merchant not able to acknowledge order* (Critical): Merchants are
          unable to acknowledge incoming orders.

          - *Merchant not able to cancel order* (Critical): Merchants are unable
          to cancel orders even when necessary, resulting in service disruptions.

          - *Order value mismatch* (Critical): The total value of the order
          relayed does not match the amount paid by the customer.

          - *No orders being received* (Critical): Merchants are not receiving
          orders from the aggregator platform.

          - *Order not visible on POS* (Critical): The order does not appear on
          the merchant's POS system after being relayed.

        Sub Category - 4: *Direct Ordering*:
          Issues affecting the direct ordering channel, including website or
          app-related malfunctions, and integration failures.

          Sections:
          - *Issue with GA, GTM, Firebase* (Normal): Problems with the
          integration of Google Analytics (GA), Google Tag Manager (GTM), or
          Firebase in the merchant’s tracking setup, which can result in data
          collection issues.

          - *App push notifications are not working* (Normal): The push
          notifications meant to reach customers through the Meraki marketing
          app fail to send.

          - *Campaign did not work but credits got debited* (High): Marketing
          campaign failed to launch or did not reach its intended audience, but
          the merchant's account was still charged for the credits used.

          - *SMS not delivered to end user (Meraki)* (High): SMS messages sent
          as part of a marketing campaign are not received by the customers.

          - *App Down [Android]* (Critical): The merchant's app on Android is
          not functioning, leading to a complete inability to access orders or
          store data.

          - *App Down [IOS]* (Critical): Similar to Android, but specific to
          merchants using the iOS version of the app, causing disruptions in
          operations.

          - *Website Down* (Critical): The merchant's website is unavailable,
          preventing customers from viewing products or placing orders.

          - *Website publish not responsive* (High): Merchants are unable to
          publish updates or changes to their website through the UrbanPiper
          portal.

          - *Meraki's end customer login issue* (High): End customers are unable
          to log in to the merchant's platform powered by Meraki.

          - *No Store Found error* (Critical): Customers see a 'No Store Found'
          error message when trying to access the merchant’s page.

          - *Banner images not visible / blurred* (High): Banner images on the
          merchant's website appear either blurry or not visible at all.

          - *Vulnerability Issue* (Critical): Security vulnerabilities found on
          the website, potentially exposing the platform to hacking or data
          breaches, etc

        Sub Category - 5: *Data*:
          Issues with analytics, reporting, or data accuracy that may impact
          business decisions.

          Sections:
          - *Blank link in report email* (High): Occurs when merchants receive
          an email notification for reports but the link is either missing or
          leads to an empty page, preventing them from accessing the reports.

          - *Reports not getting downloaded* (High): A technical issue where
          merchants are unable to download their requested reports from the
          UrbanPiper portal.

          - *Wrong data in analytics* (Normal): When the data presented in
          analytics reports is incorrect or misaligned with actual business
          activities, leading to misinformed decisions, etc.

          - *Wrong data in report* (Normal): In reports, where the data does not
          reflect accurate performance metrics or details.

        Sub Category - 6: *Escalation*:
          Issues that require immediate attention from specific teams (such as
          Onboarding, Sales, or Support) due to high-impact situations.

          Sections:
          - *Onboarding* (High): Queries related to the onboarding process, such
          as transferring the responsibility or documentation during the
          onboarding phase, process to set up their store on UrbanPiper, which
          includes configuring their account and integrating services,updates on
          the status of their onboarding process including pending
          configurations or verifications.

          - *Account Manager* (High): Issues that require urgent assistance from
          the assigned account manager, such as account setup or changes,
          handling escalated client requests, or managing client expectations
          and communication during critical phases of their journey, etc.

          - *Sales* (High): Queries that need immediate intervention from the
          sales team, including assistance with pricing, contract terms, new
          product offerings, or urgent pre-sales consultations necessary for
          converting leads into clients, etc.

          - *Support* (High): Support issues that impact business operations,
          such as resolving technical issues, troubleshooting system errors, or
          addressing urgent customer service concerns that require immediate
          resolution to prevent disruption, etc.

    3. Category: *Service Request*
        Requests for configuration changes, additional services, or product
        feature enablement, often impacting how the platform or app is used.

        Sub Category - 1: *Menu*:
          Requests related to menu customization, pricing, or training needs.

          Sections:
          - *Catalogue Config Request* (Normal): Merchants request specific
          changes or configurations in their catalogue, such as item additions,
          updates, or structural changes.

          - *Menu Configuration* (Normal): Issues or requests related to
          configuring the menu structure on the platform, including category and
          item arrangements.

          - *Image Configuration* (Normal): Refers to problems or requests
          related to uploading, editing, or configuring images for the
          merchant's catalogue or promotional materials.

          - *Price Updates* (Normal): Refers to issues or requests related to
          updating the prices of items on the platform, including batch or
          individual item price adjustments.

          - *Product Training* (Normal): Requests for product training or issues
          related to the proper understanding of how to use UrbanPiper's systems
          and features.

          - *Inventory Setup* (Normal): Refers to problems or queries related to
          the initial setup or ongoing management of inventory within the
          UrbanPiper system.

        Sub Category - 2: *Foundation*:
          Requests related to account access, onboarding, or permissions.

          Sections:
          - *Enable new platform/feature* (High): Merchants request the
          activation of new features or platforms within the UrbanPiper system,
          such as payment gateways or integrations, etc.

          - *Account Reactivation* (High): Merchants request the reactivation of
          a previously deactivated or suspended account.

          - *Demo Request* (High): Merchants request a demo of new features or
          product offerings from Urban Piper to better understand the
          capabilities of the platform.

          - *Onboarding Handover* (Normal): Queries related to the onboarding
          process, such as transferring the responsibility or documentation
          during the onboarding phase.

          - *Onboarding Request* (High): Merchants request to begin the
          onboarding process to set up their store on Urban Piper, which
          includes configuring their account and integrating services.

          - *Onboarding status check* (High): Merchants request updates on the
          status of their onboarding process, including pending configurations
          or verifications.

          - *Gamma credentials* (High): Issues with managing or using Gamma
          credentials, affecting access to the platform or specific features.

          - *Order Taker login* (High): Merchants face issues with logging into
          the 'Order Taker' feature of the platform, which is responsible for
          managing incoming orders.

          - *Prime portal creation* (Normal): Issues related to creating or
          configuring a Prime portal for merchants, including access and setup
          problems.

          - *ULE - Adding location* (High): Problems with adding new physical
          locations.

          - *ULE - New user invite* (High): Merchants are unable to send invites
          to new users or team members to join the Urban Piper system.

        Sub Category - 3: *Direct Ordering*:
          Requests for changes or new feature configurations on the direct
          ordering platform.

          Sections:
          - *Enable Payment Gateway* (High): Requests to enable or troubleshoot
          payment gateway integrations, affecting the ability to process
          payments from customers.

          - *Google Maps API integration (Meraki)* (Normal): Issues or requests
          related to the integration of Google Maps API for use in store
          location services or delivery tracking.

          - *Offer configuration* (Normal): Problems with setting up or managing
          promotional offers and discounts.

          - *Third Party access (GA, GTM)* (Normal): Merchants require
          assistance with granting third-party access, such as to Google
          Analytics (GA) or Google Tag Manager (GTM), for tracking purposes.

          - *UI / UX change - page configuration* (Normal): Requests or issues
          related to changing the user interface (UI) or user experience (UX)
          configuration.

          - *Meraki SMS / Email / WhatsApp campaign* (Normal): Issues
          specifically related to sending SMS, Email, or WhatsApp campaigns
          through the Meraki platform.

          - *New SMS template registration (Meraki)* (Normal): Merchants face
          problems registering or creating new SMS templates within the Meraki
          system.

          - *Banner images configuration* (Normal): Issues with configuring or
          updating banner images.

        Sub Category - 4: *Data*:
          Requests for new data reports, data configurations, or analytics.

          Sections:
          - *Existing Report Requirement* (Normal): Merchants request the
          continuation or enhancement of existing reports to better understand
          their business performance.

          - *New trigger event required on GA* (Normal): Merchants may need new
          event triggers set up within Google Analytics to track specific
          actions or behaviors on their platform.

    4. Category: *Feature Request*
        Requests for new platform features or major enhancements.

        Sub Category - 1: *Squad Name*:
          Sections:
          - *New feature requirement* (Normal): Merchants request new features
          or enhancements to existing functionalities in the UrbanPiper system
          to better meet their business needs.

    5. Category: *Query*
        Requests for information or general inquiries not directly related to
        troubleshooting or platform operations.

        Sub Category - 1: *External Enquiries*:
          Sections:
          - *Finance Query* (Normal): Merchants have questions or requests
          regarding billing, payments, or other financial matters related to
          their account with UrbanPiper, etc.

          - *HR queries* (Normal): Human resource-related questions or requests
          from merchants, possibly about managing staff accounts or permissions
          in the system.

          - *Sales Query* (Normal): Merchants ask questions related to sales
          operations, such as pricing, contracts, or performance metrics on the
          Urban Piper platform, etc.

        Sub Category - 2: *Squad Name*:
          Sections:
          - *Merchant query* (Normal): General inquiries or requests from
          merchants about using the platform, product features, or
          service-related issues, etc.

    6. Category: *Others*
        Non-critical issues that do not fall into the primary categories, such
        as spam or general marketing queries.

        Sub Category - 1: *Junk/Spam*:
          Sections:
          - *Junk/Spam* (Normal): Refers to issues or reports related to
          unwanted or spam messages being received by the merchant or customer
          through the UrbanPiper platform, etc.

          - *Marketing Email* (Normal): Issues related to marketing emails, such
          as delivery failures, incorrect content, or legal compliance concerns.

    Instructions:
    - The classiffication is structured as follows:
    Categories -> Sub categories -> Sections (Priority) -> Reason
    - Output must be in valid JSON format.
    - Identify all issues and classify each under one category, sub-category,
    and section based on the provided details.
    - Assign the correct priority (Normal, High, Critical) as per the
    sub-category and section.
    - Provide a clear reason for each categorization.

    Formatting:
    - Multiple issues: List each issue separately in an array.
    - Single issue: Still enclosed in an array with one object.
    - No issues: Return an empty array.
    - Strictly follow the structure (no extra text, comments, or trailing
    commas).
    - JSON Structure for OUTPUT
    {
      "response_body": [
        {
          "Category": "<Category Name>",
          "Sub_category": "<Sub-category Name>",
          "Section": "<Section Name>",
          "Priority": "<Priority>",
          "Reason": "<Reason for Categorization>"
        }
      ]
    }
    """
    issue_tree_analysis = None

    messages = [
        ("system", PROMPT),
        ("human", f"Full Transcript:\n{transcript}"),
    ]
    print("\n\nSending *Issue Tree Analysis* prompt to OpenAI model")
    try:
        ai_response = llm.invoke(messages)
        ai_response_content = ai_response.content

        ai_response_content = extract_json(ai_response_content)
        issue_tree_analysis = json.loads(ai_response_content)
        issue_tree_analysis = issue_tree_analysis.get("response_body", [])
        print("\n\nIssue Tree Analysis:")
        print(issue_tree_analysis)
    except Exception as e:
        print(f"**ERROR**: {type(e)}: {str(e)}")

    return issue_tree_analysis


def generate_call_rating(transcript, max_retries=3):
    global llm

    PROMPT = """
        Role: You are a top Quality Auditor at a call center evaluating agent
    compliance. Analyze the call transcript (which may be in English, Hindi, or
    mixed) and provide accurate ratings based on the parameters below.
    The response should be in exact json that i have provided below their should be no change in format.

    Respond in this JSON format with explanations:
    {
      "OpeningStatement": {
        "score": "Floating point score out of 8.8 (e.g., 7.2)",
        "quote": "Quote from transcript",
        "explanation": "Brief explanation"
      },
      "Personalization": {
        "score": "Floating point score out of 8.8 (e.g., 6.5)",
        "quote": "Quote from transcript",
        "explanation": "Brief explanation"
      },
      "RephraseParaphrase": {
        "score": "Floating point out of 8.8",
        "quote": "",
        "explanation": ""
      },
      "ActiveListening": {
        "score": "Floating point out of 8.8",
        "quote": "",
        "explanation": ""
      },
      "UnderstandingEmpathyApology": {
        "score": "Floating point out of 8.8",
        "quote": "",
        "explanation": ""
      },
      "HoldProcedures": {
        "score": "Floating point out of 8.8",
        "quote": "",
        "explanation": ""
      },
      "Escalations": {
        "score": "Floating point out of 8.8",
        "quote": "",
        "explanation": ""
      },
      "ProbingProcedures": {
        "score": "Floating point out of 8.8",
        "quote": "",
        "explanation": ""
      },
      "OwnershipExplanation": {
        "score": "Floating point out of 8.8",
        "quote": "",
        "explanation": ""
      },
      "SentimentProblemSolving": {
        "score": "Floating point out of 20.0",
        "quote": "",
        "explanation": ""
      },
      "OverallCallRating": {
        "score": "Floating point total out of 100.0",
        "explanation": "Brief overall assessment"
      }
    }


    Scoring Guidelines:
    1. Opening Statement (8.8): Greeting (3), Introduction (3), Assistance
    Offer (3)
    2. Personalization (8.8): Using customer's name and pleasantries
    3. Rephrase/Paraphrase (8.8): Demonstrating understanding of customer
    concerns
    4. Active Listening (8.8): No rushing, avoiding jargon, not interrupting
    5. Understanding/Empathy/Apology (8.8): Acknowledging issues and showing
    empathy
    6. Hold Procedures (8.8): Following hold script, seeking approval, informing
    of duration
    7. Escalations (8.8): Appropriate escalation and documentation
    8. Probing Procedures (8.8): Effective questioning without redundancy
    9. Ownership/Explanation (8.8): Taking responsibility and clearly explaining
    solutions
    10. Sentiment/Problem Solving (20): Customer's emotional state (Positive:
    20-12, Neutral: 12-5, Negative: 5-0)

    Note: If Hold Procedures or Escalations don't apply, exclude them from the
    final score calculation. Ensure scores and explanations reflect call
    variations accurately.

    """    
    call_rating_analysis = None

    messages = [
        ("system", PROMPT),
        ("human", f"Full Transcript:\n{transcript}"),
    ]

    for attempt in range(max_retries):
        print(f"\n\nSending *Call Rating Analysis* prompt to OpenAI model (Attempt {attempt + 1}/{max_retries})")
        try:
            ai_response = llm.invoke(messages)
            ai_response_content = ai_response.content

            ai_response_content = extract_json(ai_response_content)

            # Check if JSON extraction was successful
            if ai_response_content is None:
                raise ValueError("No JSON found in response")

            call_rating_analysis = json.loads(ai_response_content)
            print("\n\nCall Rating Analysis:")
            print(call_rating_analysis)

            # If successful, break out of retry loop
            return call_rating_analysis

        except Exception as e:
            print(f"**ERROR** (Attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}")

            # If this was the last attempt, return None
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts. Returning None.")
                return None

            # Otherwise, wait a bit before retrying (optional)
            print("Retrying...")
            # Uncomment the line below if you want to add a delay between retries
            # import time; time.sleep(1)

    return call_rating_analysis


def generate_ai_analysis(transcript, max_retries=3):
    ai_analysis = {}

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            running_functions = {
                executor.submit(generate_key_analysis, transcript, max_retries): "key_analysis",
                executor.submit(generate_statement_analysis, transcript, max_retries): "statement_analysis",
                executor.submit(generate_call_rating, transcript, max_retries): "call_rating",
                executor.submit(generate_issue_tree_analysis, transcript, max_retries): "issue_tree",
            }

            # Wait for results
            results = {}
            for future in concurrent.futures.as_completed(running_functions):
                function_name = running_functions[future]
                try:
                    results[function_name] = future.result()
                except Exception as err:
                    print(f"{function_name} generated an exception: {err}")
                    results[function_name] = None

            # Process results - the individual functions already handle retries,
            # so we just need to check if they returned valid data
            if results.get("key_analysis"):
                ai_analysis.update(results["key_analysis"])

            if results.get("statement_analysis"):
                ai_analysis.update(results["statement_analysis"])

            if results.get("issue_tree"):
                ai_analysis["complaint_insights"] = results["issue_tree"]

            if results.get("call_rating"):
                ai_analysis["OverallCallRatingAnalysis"] = results["call_rating"]

            print("\n\nFinal AI Analysis:")
            print(ai_analysis)

    except Exception as e:
        print(f"**ERROR**: {type(e).__name__}: {str(e)}")

    return ai_analysis


def get_words_in_text(text: str) -> int:
    """
    Get the number of words in a text

    Args:
        text (str): Input text

    Returns:
        None
    """
    return text.count(" ") + 1


def get_keywords(text: str) -> list:
    """
    Extract the important keywords from text

    Args:
        text (str): Input text

    Returns:
         list: List of keywords with similarity score
    """
    # Extract keywords dynamically using KeyBERT
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words="english",
        top_n=get_words_in_text(text) // 2,
    )
    print(keywords)
    return keywords


def analyse_audio(audio_path: str, temp_dir: str, country: str = None):
    file_name = os.path.splitext(os.path.basename(audio_path))
    file_name_no_ext = file_name[0]

    # Create process directory
    process_dir = os.path.join(temp_dir, file_name_no_ext)
    os.makedirs(process_dir, exist_ok=True)
    wav_path = os.path.join(process_dir, file_name_no_ext) + ".wav"
    # print(wav_path)

    convert_mp3_to_wav(audio_path, wav_path)

    ssm, transcript, gpt_analysis, keywords = None, None, None, None

    print(f"\n\n============ TRANSCRIBE AUDIO \nCountry: {country}\n")
    if country == "India":
        print("Working with DEEPGRAM")
        word_list, transcript = deepgram_transcribed_aligned(wav_path)
        print("Deepgram Transcript:", transcript)
        print("Word List:", word_list)
    else:
        print("Working with WHISPERX")
        word_list, transcript = whisperx_transcribed_aligned(wav_path)
        print("Whisper Transcript:", transcript)
        print("Word List:", word_list)

    # Run NeMo diarization
    diarization_result = run_diarization(
        wav_path, process_dir, file_name_no_ext, country
    )
    print("Diarization Result:", diarization_result)

    # Map words with diarized segments
    mapped_words = map_diarization_with_segments(word_list, diarization_result)
    print("Mapped Words:", mapped_words)

    # Format SSM - previously ssm = format_ssm(ssm, country=country)
    ssm = realign_mapped_segments(mapped_words)
    print("SSM:", ssm)

    # Replace entity deviations
    ssm = replace_deviations(ssm, data_type="dict")
    transcript = replace_deviations(transcript)

    formatted_conv = format_conversation(ssm)
    print(formatted_conv)

    gpt_analysis = generate_ai_analysis(formatted_conv)
    # print(gpt_analysis)

    if country == "India":
        keywords = get_keywords(transcript)
    else:
        keywords = get_keywords(
            gpt_analysis["Summary"]["Summary of the whole conversation"]
        )

    return ssm, transcript, gpt_analysis, keywords


# #################### END: Process Audio


# #################### START: Format JSON OUTPUT
def format_end_result(
    filename: str | None,
    json_data: dict | list,
    diarized_transcript: dict | list | None,
    full_transcript: str | None,
    ai_analysis: dict | list | None,
    key_words: list | None,
) -> tuple[dict, dict]:
    cdr_id = str(uuid.uuid4())

    tz_IST = pytz.timezone("Asia/Kolkata")

    json_result = {
        "id": str(uuid.uuid4()),
        "filename": filename,
        "status": "completed",
        "result": {
            "input": json_data,
            "conversation": diarized_transcript,
            "Customer_mentions": key_words,
        },
        "transcript": full_transcript,
        "ai_analysis": ai_analysis if full_transcript else None,
        "skill": "zuqo1",
        "conversation": True,
        "language": "english",
        "source": None,
        "cdr_id": cdr_id,
        "created_at": {"$date": datetime.now(tz_IST).isoformat()},
        "updated_at": {"$date": datetime.now(tz_IST).isoformat()},
        "timestamp": int(datetime.now(tz_IST).timestamp()),
    }
    json_cdr = {
        "uuid": cdr_id,
        "answerstamp": json_data["startTime"],
        "answertime": json_data["startTime"],
        "billsec": json_data["duration"],
        "callee": (
            json_data["to"]
            if json_data["callType"] == "Outgoing"
            else json_data["from"]
        ),  # .replace("+", "")
        "caller": (
            json_data["from"]
            if json_data["callType"] == "Outgoing"
            else json_data["to"]
        ),
        "callerhost": "3.139.177.227",
        "calltype": (
            "VoiceCallChatOutbound"
            if json_data["callType"] == "Outgoing"
            else "VoiceCallChat"
        ),
        "duration": json_data["duration"],
        "endpoint_disposition": "ANSWER",
        "endpoint_disposition2": "ANSWER",
        "endstamp": json_data["endTime"],
        "groupID": "zuqo1",
        "hangup_cause_q850": "16",
        "ivrexittime": "nil",
        "last_app": "bridge",
        "remote_media_ip": "112.206.251.158",
        "remote_media_port": "36173",
        "sip_hangup_disposition": "recv_bye",
        "sip_term_cause": "16",
        "skill": "zuqo1",
        "startstamp": json_data["startTime"],
        "team": "zuqo1",
        "timestamp": int(
            datetime.strptime(
                json_data["startTime"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ).timestamp()
        ),
        "userId": (
            json_data["email"]
            if json_data["callType"] == "Outgoing"
            else json_data["to"]
        ),
        "asr_notify": "success",
        "actualtalktime": str(json_data["duration"]) + "sec",
        "agentname": (
            json_data["Caller Name"] if json_data["callType"] == "Outgoing" else ""
        ),
    }
    print("Output generated for input json")
    # json_result = json.dumps(json_result, indent=4)
    # print(json_result)
    return json_result, json_cdr


# #################### END: Format JSON OUTPUT


# #################### START: Get Overall Call Rating
def get_overall_call_rating(analysis_result):
    def extract_score(text):
        """
        Extracts a numeric score from a text like '6', or '10 out of 10'.
        Returns the score as an integer or 0 if no score is found.
        """
        # Regex to capture scores in formats
        pattern = r"(\d+)(?:\s*out\s*of\s*\d+)?"
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return int(match.group(1))  # Return the first numeric match as an integer
        return 0

    if not analysis_result:
        return None

    overall_sentiment = "neutral"
    if (
        "Overall_sentiment" in analysis_result
        and "call_sentiment" in analysis_result["Overall_sentiment"]
    ):
        overall_sentiment = analysis_result["Overall_sentiment"]["call_sentiment"]
        print("Analysis Call Sentiment:", overall_sentiment)

    tss = 0  # Total satisfaction score
    pss = 0  # Problem solving skill
    if "customer_metrics" in analysis_result:
        tss = next(
            (
                item["score"]
                for item in analysis_result["customer_metrics"]
                if item["name"] == "TSS"
            ),
            "0",
        )

        tss = extract_score(tss)
        pss = next(
            (
                item["score"]
                for item in analysis_result["customer_metrics"]
                if item["name"] == "Problem Solving"
            ),
            "0",
        )
        pss = extract_score(pss)

    ocr = (tss + pss) / 2  # Overall call rating

    return ocr, overall_sentiment


# #################### END: Get Overall Call Rating


# #################### START: Push to DB
def push_to_db(asr_data: dict, cdr_data: dict) -> bool:
    global mongo_db_uri, MONGO_DB_NAME, MONGO_DB_ASR_COLLECTION, MONGO_DB_CDR_COLLECTION
    client = MongoClient(mongo_db_uri)

    db = client[MONGO_DB_NAME]
    asr_collection = db[MONGO_DB_ASR_COLLECTION]
    cdr_collection = db[MONGO_DB_CDR_COLLECTION]

    # Insert ASR data into the collection
    asr_result = asr_collection.insert_one(asr_data)
    if asr_result:
        print("ASR pushed to DB: ", asr_result.inserted_id)

    # Insert CDR data into the collection
    cdr_result = cdr_collection.insert_one(cdr_data)
    if cdr_result:
        print("CDR pushed to DB: ", cdr_result.inserted_id)

    return True if asr_result and cdr_result else False


# #################### END: Push to DB


# #################### START: Send Mail
def send_mail(recipients, subject, body):
    global SENDER_MAIL, SENDER_PWD

    is_sent = False
    bcc_recipients = ["asutosh@zuqo.io"]  # To be removed later

    try:
        msg = EmailMessage()
        msg["From"] = SENDER_MAIL
        msg["To"] = ", ".join(recipients)
        msg["Bcc"] = ", ".join(bcc_recipients)
        msg["Subject"] = subject
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.login(SENDER_MAIL, SENDER_PWD)
            smtp.send_message(msg)
            print(f"Alert email sent")
            is_sent = True
    except Exception as e:
        post_error()
        print(f"Email send failed\nERROR: {e}")
    finally:
        return is_sent


# #################### END: Send Mail


# #################### START: Detect Keywords
def detect_keywords_and_alert(analytics, conversation, call_details):
    recipients = ["cs.alerts@urbanpiper.com"]

    try:
        if "Speaker" not in analytics:
            # If there are no speaker segmentation; skip
            return

        customer_speaker = None
        speaker_data = analytics.get("Speaker", {})
        speaker_mapping = {}

        if isinstance(speaker_data, dict):
            if all(isinstance(v, dict) for v in speaker_data.values()):
                # Case 1: Nested under description
                speaker_mapping = next(iter(speaker_data.values()))
            else:
                # Case 2: Already mapping
                speaker_mapping = speaker_data

        # if "speaker_0" in speaker_mapping and "speaker_1" in speaker_mapping:
        #     if speaker_mapping["speaker_0"].lower() == "customer":
        #         # customer_speaker = "Speaker 0"
        #         customer_speaker = 0
        #     else:
        #         # customer_speaker = "Speaker 1"
        #         customer_speaker = 1
        #
        # if not customer_speaker:
        #     # If there are no customer data; skip
        #     return
        #
        # # Get only customer statements
        # customer_statements = []
        # for speech in conversation:
        #     if speech.get("speaker") == customer_speaker:
        #         customer_statements.append(speech.get("text", "").strip())

        # Extract only customer statements
        customer_statements = [
            entry["text"]
            for entry in conversation
            if speaker_mapping.get(f"speaker_{entry['speaker']}", "unknown")
            == "customer"
        ]

        transcript = " \n".join(customer_statements)

        system_prompt = f"""You are an expert call auditor.

Your task is to analyze the entire call context provided
(not individual statements in isolation) and determine whether any of the
defined intents are explicitly expressed in a negative way by the customer.
Only match intents that are directly, clearly, and explicitly stated by the
customer.

Valid Intents (Match Only if Clearly Expressed):
- Legal Action: Customer clearly states they will take legal action
  (e.g., “I will file a legal complaint”).
- Breach of Contract: Explicit mention of a contract/agreement violation.
- Fraud/Scam: Customer uses words like “fraud”, “scam”, “cheated” specifically
  in reference to UrbanPiper or its staff.
- Compensation for Loss: Customer explicitly asks for money back or
  compensation.
- Abuse or Defamation: Customer uses genuinely abusive language toward staff or
  threatens public defamation (e.g., “I’ll ruin your company online”).
  -- Note: Casual profanity (e.g., “fuck”, “shit”, “man”) used colloquially or
     out of frustration should NOT be tagged as abuse.
- Escalation: Customer asks to speak to someone higher or requests escalation.
- Terrible Service: Customer clearly mentions unresolved issues,
  repeated problems, or delays in support.
- Switching to Competitor / Churn: Customer says they will leave UrbanPiper or
  move to another platform.

JSON Output Structure:
{{
    "found_intent": true or false,
    "intent_list": [
        {{
            "intent": "The matched intent from the list",
            "statement": "The exact spoken statement by customer most resembling to the matched intent",
            "evidence": "Combined evidence from the full call context"
        }}
    ]
}}

Important Instructions:
- Evaluate the full call context, not line-by-line statements.
- Only consider what is directly stated by the customer.
- Only detect intents when expressed negatively, with frustration,
  dissatisfaction, or anger.
- Do NOT match:
    - Vague or emotional statements (e.g., “I’m unhappy”, “This is misleading”).
    - Implied or indirect mentions (e.g., “I’ll talk to my lawyer” ≠ legal action).
    - General loss without an explicit request for compensation (e.g., “I lost money” ≠ compensation).
- Do NOT include intents where the agent has already resolved the issue.
- Do NOT duplicate intents.
    - If an intent appears multiple times, list it once and combine all relevant
      evidence for that intent.
"""

        llm_query = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Here is the Customer Statements to analyze:\n{transcript}\nPlease respond with the analysis as JSON only.",
            },
        ]

        # Invoke the LLM
        ai_response = llm.invoke(llm_query)

        # Extract content
        ai_response_content = ai_response.content.strip()

        # Extract JSON content
        json_match = re.search(
            r"\{.*\}", ai_response_content, flags=re.DOTALL
        )  # Matches a JSON object
        if json_match:
            cleaned_response_content = json_match.group(0)
        else:
            print("No JSON content available in Email Alert prompt")
            return None

        # Convert the cleaned string to JSON
        try:
            parsed_json = json.loads(cleaned_response_content)
            print("Email Alert:", parsed_json)

            if parsed_json.get("found_intent", None):
                intent_list = []
                evidence_list = []
                statement_list = []
                for intents in parsed_json["intent_list"]:
                    intent_list.append(intents["intent"])
                    statement_list.append(intents["statement"])
                    evidence_list.append(intents["evidence"])

                intent_list = ", ".join(intent_list)
                statement_list = "- " + " \n- ".join(statement_list)
                evidence_list = "- " + " \n- ".join(evidence_list)
                print("Intent list: ", intent_list)
                print("Statement list: ", statement_list)
                print("Evidence list: ", evidence_list)

                utc_time = datetime.strptime(
                    call_details["time"], "%Y-%m-%dT%H:%M:%S.%fZ"
                )
                utc_time = utc_time.replace(tzinfo=timezone.utc)

                # Convert to IST (UTC+5:30)
                ist_offset = timedelta(hours=5, minutes=30)
                ist_time = utc_time.astimezone(timezone(ist_offset))

                # Format as DD-MM-YYYY HH:MM
                formatted_time = ist_time.strftime("%d-%m-%Y %H:%M")

                # Compose subject and body
                subject = f"ALERT: Critical Keywords Detected in Interaction #{call_details['Call Sid'][:8]}"
                body = f"""
Dear Team,

Our monitoring system has identified potentially concerning keywords in a recent conversation that requires immediate attention:

Conversation: https://urbanpiper.zuqo.app/admin/dashboard/ai-analytics/{call_details['Call Sid']}
Flagged Intents: {intent_list}
Statements:
{statement_list}

Reasons:
{evidence_list}

Customer Number: {call_details['to'] if call_details['callType'] == 'Outgoing' else call_details['from']}
Call Handled By: {call_details['Caller Name']} ({call_details['email']})
Time: {formatted_time}


Please review this interaction promptly and take appropriate action according to protocol.

Thank you,
Urban Piper Alert System
Powered by Zuqo
                """
                if (
                    len(
                        set(
                            call_details["Caller Name"].lower().split(" ")
                        ).intersection(AGENT_NAMES)
                    )
                    > 0
                ):
                    send_mail(recipients, subject, body)
                return
            else:
                print("No alert keywords found in the transcript.")
                return
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            print(
                "Cleaned Response Content:", cleaned_response_content
            )  # Debug: Print cleaned content
            return
    except Exception as err:
        post_error()
        print(f"{type(err)}: {err}")
        return


# #################### END: Detect Keywords


# #################### START: Main Process Audio
if __name__ == "__main__":
    import time

    print("+" * 50)
    print("RUNNING HELPER.PY EXAMPLE")
    print("+" * 50)

    # Ensure the download folder exists
    DOWNLOAD_DIR = "downloaded_audio_files"
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Create a temporary directory for processing audio files
    TEMP_DIR = "temp_dir"
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Example JSON
    input_json = {
        "from": "+918069098166",
        "to": "+916364330840",
        "dialCode": "91",
        "callType": "Outgoing",
        "duration": "43",
        "status": "Completed",
        "time": "2024-09-25T12:01:40.000Z",
        "callCharge": "0.00",
        "email": "arbab.tt@urbanpiper.com",
        "Caller Name": "Arbab Ali",
        "Call Sid": "9a1e9517-e50a-4566-8837-e82627ed8963",
        "startTime": "2024-09-25T12:00:48.000Z",
        "endTime": "2024-09-25T12:01:40.000Z",
        "recordingUrl": "https://callhippo-media.s3.amazonaws.com/callrecordings_hippa/9a1e9517-e50a-4566-8837-e82627ed8963.mp3",
        "countryName": "India",
        "answeredDevice": "web",
        "ringAnswerDuration": "00:43",
        "billedMinutes": 1,
        "hangupBy": "Agent",
        "callQueue": False,
        "reason": "",
    }
    audio_url = input_json.get("recordingUrl") or input_json.get("recording_url")
    # audio_url = "https://callhippo-media.s3.amazonaws.com/callrecordings_hippa/6c3fede2-26b8-42b2-acf9-0355d15e80bf.mp3"
    # audio_url = "https://callhippo-media.s3.amazonaws.com/callrecordings_hippa/70adcb47-36ed-4dcc-a070-2d629322efec.mp3"

    start_time = time.time()

    audio_file_path = download_audio_file(audio_url, DOWNLOAD_DIR)
    print(audio_file_path)

    diarized_transcript, full_transcript, ai_analysis, key_words = analyse_audio(
        audio_file_path, TEMP_DIR, input_json.get("countryName")
    )
    print("==========Diarize Output")
    print(diarized_transcript)

    print("==========Full Transcript")
    print(full_transcript)

    print("==========AI Analysis")
    print(ai_analysis)

    print("==========Keywords")
    print(key_words)

    audio_filename = "".join(os.path.splitext(os.path.basename(audio_url)))
    result_out, result_cdr = format_end_result(
        audio_filename,
        input_json,
        diarized_transcript,
        full_transcript,
        ai_analysis,
        key_words,
    )
    print(result_out)

    call_rating = None
    call_sentiment = None
    call_performance = get_overall_call_rating(result_out["ai_analysis"])
    if call_performance:
        call_rating, call_sentiment = call_performance
    print("Call Rating: ", call_rating, "\nCall Sentiment: ", call_sentiment)

    client = MongoClient(mongo_db_uri)
    db = client[MONGO_DB_NAME]
    asr_collection = db[MONGO_DB_ASR_COLLECTION]
    cdr_collection = db[MONGO_DB_CDR_COLLECTION]

    # Insert ASR data into the collection
    asr_result = asr_collection.insert_one(result_out)
    if asr_result:
        print("ASR pushed to DB: ", asr_result.inserted_id)

    # Insert CDR data into the collection
    cdr_result = cdr_collection.insert_one(result_cdr)
    if cdr_result:
        print("CDR pushed to DB: ", cdr_result.inserted_id)

    print("Time taken for the whole process:", time.time() - start_time)
