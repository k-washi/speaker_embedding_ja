# speaker_embedding_ja

日本語の音声データセットで学習した話者埋め込みのモデルを提供しています。

Youtubeなどから収集した比較的クリーンな音声データ(話者数8036名、1621時間)を用いて学習しました。
モデルは、SpeechBrainさんの[EcapaTDNN](speechbrain/spkrec-ecapa-voxceleb)を参考にいたしました。
また、学習手法に関しては、[ECAPA2](https://arxiv.org/abs/2401.08342)に記載されている手法を参考にしました。

テストには、ランダムに選択した200名の音声データを使用し、EERで評価しています。

||EER|
|---|---|
|Our EcapaTDNN|9.05%|
|SpeechBrain EcapaTDNN|12.88%|
|JTubeSpeech XVector|40.24%|

参考のため、以下の話者埋込みモデルによるテスト結果も記載しています。

- [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- [sarulab-speech/xvector_jtubespeech](https://github.com/sarulab-speech/xvector_jtubespeech)



## Usage

```python
import torch
import torch.nn.functional as F
import torchaudio

audio_path = "path/to/audio.wav"
model = torch.hub.load(
        "k-washi/speaker_embedding_ja", 
        "ecapatdnn_ja_l512", 
        trust_repo=True, pretrained=True
    )
wav, sr = torchaudio.load(audio_path)
wav = torchaudio.transforms.Resample(sr, model.sample_rate)(wav) # (batch:1, wave length)


emb = model(wav) # -> (batch:, hidden_size)
emb = F.normalize(torch.FloatTensor(emb), p=2, dim=1).detach().cpu()

# embedding similarity
score = torch.mean(torch.matmul(emb, emb.T)) # # tensor([[1.]]) (batch1, batch2)
```

# Dataset

Youtubeからすべて集めるのが大変なので、以下のデータも使用させていただきました。

- [Laboro-ASV](https://laboro.ai/activity/column/engineer/laboro-asv/)
- [CommonVoice](https://commonvoice.mozilla.org/ja)

# Dev

```
pip install -e .
```