import numpy
import torch
import torch.nn.functional as F
import torchaudio
from faster_whisper import WhisperModel
# from wespeaker.cli.speaker import Speaker # We are replacing this
from speechbrain.inference.speaker import EncoderClassifier


# class Speaker_emb(Speaker):
#     def __init__(self, model_dir: str):
#         super().__init__(model_dir)

#     def extract_embedding_from_pcm(self, pcm: torch.Tensor, sample_rate: int):
#         pcm = pcm.to(torch.float)
#         if sample_rate != self.resample_rate:
#             pcm = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.resample_rate)(pcm)
#         feats = self.compute_fbank(pcm, sample_rate=self.resample_rate, cmn=True)
#         feats = feats.unsqueeze(0)
#         feats = feats.to(self.device)

#         with torch.no_grad():
#             outputs = self.model(feats)
#             outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
#         return outputs


# model_spk_dir = 'src/rl/wespeaker/chinese' # Old model
# model_spk = Speaker_emb(model_spk_dir) # Old model initialization

# Initialize SpeechBrain ECAPA-TDNN model
# Using a generic name for the savedir, adjust if a specific project structure is preferred.
speechbrain_speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)
SPEECHBRAIN_TARGET_SR = 16000

# Ensure os is imported for path checking

# test_spk function removed as it was for manual verification.

def get_emb(wav, sr):
    # wav -> (b, t), torch.tensor
    # Ensure model is on the same device as the input wav data
    device = wav.device
    speechbrain_speaker_model.to(device)
    speechbrain_speaker_model.eval() # Set to eval mode

    # Resample if necessary
    if sr != SPEECHBRAIN_TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SPEECHBRAIN_TARGET_SR).to(device)
        # Resample each item in the batch if they have different lengths,
        # or resample the whole batch if possible.
        # encode_batch expects a batch of tensors. If wav is already a batch (e.g. BxT)
        # and resampler can handle it, great. Otherwise, loop.
        # Assuming wav is (B, T)
        wav_resampled = resampler(wav)
    else:
        wav_resampled = wav

    # The encode_batch expects a batch of signals.
    # Input wav_resampled should be shape (batch_size, num_samples)
    # Output embeddings are typically (batch_size, 1, embed_dim)
    with torch.no_grad():
        embeddings = speechbrain_speaker_model.encode_batch(wav_resampled)

    # Squeeze to (batch_size, embed_dim) to match expected output for cal_sim
    if embeddings.ndim == 3 and embeddings.shape[1] == 1:
        embeddings = embeddings.squeeze(1)

    return embeddings


def cal_sim(emb1, emb2):
    return F.cosine_similarity(emb1, emb2)

# Initialize Faster Whisper model
# Using "cuda" for device if available, else "cpu". Adjust compute_type as needed.
# "float16" can be used for faster inference on GPUs that support it.
# "int8_float16" for even faster inference with quantization.
# Default is "float32" on CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "auto" # auto will select float32 on CPU, float16 on CUDA
model_asr = WhisperModel("large-v3", device=DEVICE, compute_type=COMPUTE_TYPE)


def test_asr():
    # This function would need a sample audio file (e.g., 'xx.wav') to run.
    # For now, it's a placeholder for testing.
    # Example:
    # current_file = 'path/to/your/test_audio.wav'
    # audio_input, _ = torchaudio.load(current_file)
    # if audio_input.shape[0] > 1: # if stereo, convert to mono
    #    audio_input = torch.mean(audio_input, dim=0, keepdim=True)
    # texts = get_asr(audio_input.unsqueeze(0), _) # unsqueeze to add batch dim
    # print(texts[0])
    print("test_asr function needs a sample audio file to run.")


def get_asr(audios, sr):
    # audios -> (b, t), torch.Tensor
    # Ensure audio is on CPU for faster_whisper if it expects numpy array,
    # or handle device placement according to faster_whisper's requirements.
    # Faster Whisper typically expects NumPy arrays.

    texts = []

    # Resample if necessary, assuming audios is a batch of tensors (B, T)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(audios.device)
        audios_resampled = resampler(audios)
    else:
        audios_resampled = audios

    for i in range(audios_resampled.size(0)):
        audio_input_np = audios_resampled[i, :].float().cpu().numpy()

        # Transcribe audio
        # Adjust beam_size, language, etc. as needed.
        # vad_filter=True can help with silences if your audio has them.
        segments, info = model_asr.transcribe(audio_input_np, beam_size=5, language="auto", vad_filter=True)

        transcribed_text = "".join(segment.text for segment in segments)
        texts.append(transcribed_text)

    return texts


def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.

    Main algorithm used is dynamic programming.

    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def cal_wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    # build the matrix
    d = editDistance(r, h)

    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / max(1, len(r))  # * 100
    return result
