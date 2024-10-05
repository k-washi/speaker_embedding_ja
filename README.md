# speaker_embedding_ja

日本語の音声データセットで学習した話者埋め込みのモデルを提供しています。

Youtubeなどから話者数8036名、1621時間の比較的クリーンな音声データを収集し学習しました。

テストには、ランダムに選択した200名の音声データを使用し、EERで評価しています。

||EER|
|---|---|
|Our EcapaTDNN|9.05%|
|SpeechBrain EcapaTDNN|12.88%|
|JTubeSpeech XVector|40.24%|

参考のため、以下の話者埋込みモデルによるテスト結果も記載しています。

- [SpeechBrainのEcapaTDNN(speechbrain/spkrec-ecapa-voxceleb)](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- [sarulab-speech/xvector_jtubespeech](https://github.com/sarulab-speech/xvector_jtubespeech)

モデルは、SpeechBrainさんの[EcapaTDNN](speechbrain/spkrec-ecapa-voxceleb)を参考にいたしました。
また、学習手法に関しては、[ECAPA2: A Hybrid Neural Network Architecture and Training Strategy for Robust Speaker Embeddings](https://arxiv.org/abs/2401.08342)に記載されている手法を参考にしました。

## Usage

```python
from speaker_embedding_ja import SpeakerEmbeddingJa
```

# Dev

```
pip install -e .
```