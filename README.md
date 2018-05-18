# AudioProcessor
My library of audio processing
## Short Time Fourier Transform
```python
X, F, T = mystft(source, fs, framesz, overlap)
```
输入参数source为音频数据，fs为采样频率，framesz为帧的大小，overlap为移动的步长 <br>
返回向量F和T以及矩阵X，其中向量F为频率序列，T为时间序列，X为STFT的结果
## Inverse Short Time Fourier Transform
```python
x = myistft(X, fs, datasz, framesz, overlap)
```
输入参数X为STFT结果，fs为采样频率，datasz为音频数据长度，framesz为帧的大小，overlap为移动的步长 <br>
返回向量x，为逆变换得到音频数据
## Result
<p align="center">
  <img src="https://github.com/Accelerator66/AudioProcessor/raw/master/result/data1.png" width="500"/>
  <br>原始音频数据<br><br>
  <img src="https://github.com/Accelerator66/AudioProcessor/raw/master/result/STFT_Magnitude1.png" width="500"/>
  <br>STFT时频图<br><br>
  <img src="https://github.com/Accelerator66/AudioProcessor/raw/master/result/ISTFT1.png" width="500"/>
  <br>还原后的音频数据<br><br>
</p>
