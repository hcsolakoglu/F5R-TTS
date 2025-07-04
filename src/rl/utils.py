import numpy
import torch
import torch.nn.functional as F
import torchaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
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
import os

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


model_asr_dir = "src/rl/SenseVoiceSmall"
model_asr = AutoModel(model=model_asr_dir, device="cpu", disable_update=True)


def test_asr():
    current_file = 'xx.wav'
    pcm, _ = torchaudio.load(current_file)
    resampled_audio = [pcm[0]]

    res = model_asr.inference(
        input=resampled_audio,
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        disable_pbar=True,
        batch_size=len(resampled_audio)
    )
    text = rich_transcription_postprocess(res[0]["text"])
    print(text)


def get_asr(audios, sr):
    # audios -> (b, t)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        list_audios = [resampler(audios[i, :].unsqueeze(0))[0] for i in range(audios.size(0))]
    else:
        list_audios = [audios[i, :] for i in range(audios.size(0))]

    results = model_asr.inference(
        input=list_audios,
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        disable_pbar=True,
        batch_size=len(list_audios)
    )
    text = [rich_transcription_postprocess(res["text"]) for res in results]
    return text


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
