import librosa
import matplotlib.pyplot as plt
import numpy as np

# 加载音频
y, sr = librosa.load('C:\PROJECT\StableAudio\music.wav')
# 计算CQT（每八度12个频带，默认）
C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1'), n_bins=84)
# 显示对数幅度谱
librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
plt.colorbar()
plt.show()
