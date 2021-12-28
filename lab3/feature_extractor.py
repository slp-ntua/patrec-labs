# This is the code used to extract spectrograms / chromagrams
# This code is already executed for you. It's included for completeness
#
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import gzip
import os
import time

import librosa
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


def text_progessbar(seq, total=None):
    step = 1
    tick = time.time()
    while True:
        time_diff = time.time() - tick
        avg_speed = time_diff / step
        total_str = "of %n" % total if total else ""
        print(
            "step",
            step,
            "%.2f" % time_diff,
            "avg: %.2f iter/sec" % avg_speed,
            total_str,
        )
        step += 1
        yield next(seq)


all_bar_funcs = {
    "tqdm": lambda args: lambda x: tqdm(x, **args),
    "txt": lambda args: lambda x: text_progessbar(x, **args),
    "False": lambda args: iter,
    "None": lambda args: iter,
}


def ParallelExecutor(use_bar="tqdm", **joblib_args):
    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type" % bar)
            return Parallel(**joblib_args)(bar_func(op_iter))

        return tmp

    return aprun


def safe_mkdirs(path):
    """! Makes recursively all the directory in input path"""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            raise IOError(("Failed to create recursive directories: {}".format(path)))


class MusicSpectrogramExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feats=("power", "mel", "chroma", "fused"),
        hop_length=512,
        n_fft=2048,
        n_mels=128,
        ref=np.max,
        saveto=None,
        beat=False,
        n_jobs=8,
        verbose=2,
    ):
        self.feats = feats
        self.timescale = "full"
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.ref = ref
        self.n_jobs = n_jobs
        self.saveto = saveto
        self.beat = beat
        if self.saveto is not None:
            safe_mkdirs(self.saveto)
        self.verbose = verbose

    def _sync(self, S, beat_frames=None):
        if beat_frames is None:
            return S
        return librosa.util.sync(S, beat_frames, aggregate=np.median)

    def _power_spectrogram(self, y, beat_frames=None):
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        P = librosa.power_to_db(S ** 2.0, ref=self.ref)
        P = self._sync(P, beat_frames=beat_frames)
        S = self._sync(S, beat_frames=beat_frames)
        return P, S

    def _mel_spectrogram(self, y, S=None, sr=22050, beat_frames=None):
        S = None
        if S is not None:
            M = librosa.feature.melspectrogram(S=S, power=2.0)
        else:
            M = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,
                n_mels=self.n_mels,
            )
        M = self._sync(M, beat_frames=beat_frames)
        return librosa.power_to_db(M, ref=self.ref)

    def _cqt_chromagram(self, y, sr=22050, beat_frames=None):
        C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        C = self._sync(C, beat_frames=beat_frames)
        return C

    def _try_save(self, spectrogram, wav_path, feat):
        if self.saveto:
            wav_id = wav_path.split("/")[-1].split(".")[0]
            genre = wav_path.split("/")[-2]
            out_path = os.path.join(self.saveto, genre)
            safe_mkdirs(out_path)
            out_file = os.path.join(
                out_path, "{}.{}.{}.npy.gz".format(wav_id, feat, self.timescale)
            )
            with gzip.GzipFile(out_file, "w") as fd:
                np.save(fd, spectrogram, allow_pickle=False)

    def _load(self, filename):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            y, sr = librosa.load(filename)
        elif filename.endswith(".p") or filename.endswith(".pkl"):
            with open(filename, "rb") as f:
                y, sr = pickle.load(f)
        else:
            raise ValueError("{} has unsupported format".format(filename))
        return y, sr

    def _load_extract(self, filename):
        try:
            y, sr = self._load(filename)
            P, M, C, S = None, None, None, None
            beat_frames = None
            if self.beat:
                _, y_percussive = librosa.effects.hpss(y)
                _, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
            if "power" in self.feats:
                P, S = self._power_spectrogram(y, beat_frames=beat_frames)
                self._try_save(P, filename, "power")
            if "mel" in self.feats:
                M = self._mel_spectrogram(y, S=S, sr=sr, beat_frames=beat_frames)
                self._try_save(M, filename, "mel")
            if "chroma" in self.feats:
                C = self._cqt_chromagram(y, sr=sr, beat_frames=beat_frames)
                self._try_save(C, filename, "chroma")
            if "fused" in self.feats:
                specs = tuple(s for s in [P, M, C] if s is not None)
                F = np.vstack(specs)
                self._try_save(F, filename, "fused")
            return (filename, P, M, C, F)
        except Exception as e:
            print("Warning: Failed to extract features for {}".format(filename))
            if self.verbose:
                print(e)
            pass

    def fit(self, X, labels=None):
        return self

    def transform(self, paths, labels=None):
        """
        X is a list of paths to the music files
        """
        # Serial execution version
        # self.extracted = [self._load_extract(wav) for wav in tqdm(paths)]
        self.extracted = ParallelExecutor(n_jobs=self.n_jobs)(total=len(paths))(
            [delayed(self._load_extract)(filename) for filename in paths]
        )
        return self.extracted


if __name__ == "__main__":
    import sys

    import glob2

    path_to_wavs = sys.argv[1]
    out_spectrograms = sys.argv[2] if len(sys.argv) > 2 else None
    wavs = glob2.glob(os.path.join(path_to_wavs, "**", "*.wav"))

    features = MusicSpectrogramExtractor(
        feats=("mel", "chroma", "fused"), beat=False, saveto=out_spectrograms
    ).transform(wavs)
    print(len(features))
    print(features[0][2].shape)
    print(features[0][3].shape)
    print(features[0][4].shape)
