"""DHVANI audio authenticity analyzer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


MODEL_ID = os.getenv("DHVANI_MODEL_ID", "garystafford/wav2vec2-deepfake-voice-detector")
SAMPLE_RATE = 16_000
MAX_DURATION_SEC = 30.0


@dataclass
class AnalysisResult:
    synthetic_probability: float
    authentic_probability: float
    verdict: str
    verdict_en: str
    hindi_message: str
    model_id: str
    duration_sec: float


class VoiceAnalyzer:
    def __init__(self) -> None:
        self._model = None
        self._extractor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        self._model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
        self._model.to(self._device)
        self._model.eval()

    def analyze_file(self, audio_path: str | Path) -> AnalysisResult:
        self._ensure_loaded()
        waveform, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        duration_sec = len(waveform) / SAMPLE_RATE
        max_samples = int(MAX_DURATION_SEC * SAMPLE_RATE)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        inputs = self._extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        id2label = self._model.config.id2label
        label_probs = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}

        fake_prob = self._pick_probability(
            label_probs,
            keys=("fake", "spoof", "synthetic", "deepfake", "bonafide_neg", "label_1", "1"),
        )
        real_prob = self._pick_probability(
            label_probs,
            keys=("real", "bonafide", "genuine", "authentic", "label_0", "0"),
        )

        if fake_prob is None and real_prob is None:
            fake_prob = float(np.max(probs))
            real_prob = 1.0 - fake_prob
        elif fake_prob is None:
            fake_prob = max(0.0, 1.0 - real_prob)
        elif real_prob is None:
            real_prob = max(0.0, 1.0 - fake_prob)

        verdict, verdict_en, hindi_message = self._verdict(fake_prob)

        return AnalysisResult(
            synthetic_probability=round(fake_prob * 100, 1),
            authentic_probability=round(real_prob * 100, 1),
            verdict=verdict,
            verdict_en=verdict_en,
            hindi_message=hindi_message,
            model_id=MODEL_ID,
            duration_sec=round(duration_sec, 2),
        )

    @staticmethod
    def _pick_probability(label_probs: dict[str, float], keys: tuple[str, ...]) -> float | None:
        for key in keys:
            if key in label_probs:
                return label_probs[key]
        for label, prob in label_probs.items():
            if any(key in label for key in keys):
                return prob
        return None

    @staticmethod
    def _verdict(fake_prob: float) -> tuple[str, str, str]:
        pct = int(round(fake_prob * 100))
        if fake_prob >= 0.75:
            return (
                "LIKELY_FAKE",
                f"Likely AI-generated voice ({pct}% synthetic probability).",
                (
                    f"⚠️ {pct}% synthetic probability. Yeh awaaz AI-cloned lag rahi hai. "
                    "Order verify karein. Turant action mat lein."
                ),
            )
        if fake_prob >= 0.45:
            return (
                "SUSPICIOUS",
                f"Suspicious voice pattern ({pct}% synthetic probability).",
                (
                    f"⚠️ {pct}% synthetic probability. Sandeh janak awaaz. "
                    "Doosre channel se confirm karein."
                ),
            )
        return (
            "LIKELY_AUTHENTIC",
            f"Likely authentic voice ({pct}% synthetic probability).",
            (
                f"✅ Kam synthetic signs ({pct}%). Phir bhi critical order ho to "
                "dusre channel se verify karein."
            ),
        )