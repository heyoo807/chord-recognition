import librosa
import numpy as np
import madmom
import librosa.display
import matplotlib.pyplot as plt
from madmom.features import chords
from madmom.audio.chroma import DeepChromaProcessor

# 로드,소스 분리 및 현악기 크로마그램 생성
audio_path = './Lover.wav'
y, sr = librosa.load(audio_path)
harmonic, percussive = librosa.effects.hpss(y)
chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
chroma_avg = np.mean(chroma, axis=1)

# 크로마그램 시각화
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', cmap='coolwarm')
# plt.colorbar()
# plt.title('Chromagram')
# plt.xlabel('Time')
# plt.ylabel('Pitch Class')
# plt.show()


# key 추정
estimated_key = np.argmax(chroma_avg)
print(f"Estimated key: {estimated_key}")
# 크로마그램, 시간축 생성
chroma_os = librosa.feature.chroma_cqt(y=y, sr=sr)
times = librosa.times_like(chroma_os)

#코드 추출
proc = chords.DeepChromaChordRecognitionProcessor()
chord_recognition = chords.DeepChromaChordRecognitionProcessor()(chroma_os.T)

# chord_recognition = chords.DeepChromaChordRecognitionProcessor(chroma)

# Plot the chromagram with estimated chords
plt.figure(figsize=(14, 6))
librosa.display.specshow(chroma_os, y_axis='chroma', x_axis='time', cmap='coolwarm')
plt.title('Chromagram with Estimated Chords')
plt.colorbar()
plt.show()
