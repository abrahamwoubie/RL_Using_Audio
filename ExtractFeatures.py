import numpy as np
import aubio
from scipy import signal # audio processing
from scipy.fftpack import dct
from pydub import AudioSegment
import librosa # library for audio processing

class Extract_Features:


    def Extract_Samples(row, col,nRow,nCol):
        #print("Extract Sample of Row {} Col {} nRow {} nCol {}".format(row,col,nRow,nCol))

        fs = 100  # sample rate
        f = 2  # the frequency of the signal

        x = np.arange(fs)  # the points on the x axis for plotting

        # compute the value (amplitude) of the sin wave at the for each sample
        # if letter in b'G':
        if(row==nRow and col==nCol):
            samples = [100 + row + col + np.sin(2 * np.pi * f * (i / fs)) for i in x]
        else:
            samples = [row + col + np.sin(2 * np.pi * f * (i / fs)) for i in x]
        return samples

    def Extract_Spectrogram(row, col,nRow,nCol):
        fs = 10e3
        N = 1e5
        amp = 2 * np.sqrt(2)
        noise_power = 0.01 * fs / 2
        time = np.arange(N) / float(fs)
        mod = 500 * np.cos(2 * np.pi * 0.25 * time)
        carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
        # noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        # noise *= np.exp(-time / 5)
        # x = carrier + noise  # x is the sample
        x = carrier
        frequencies, times, spectrogram = signal.spectrogram(x, fs)
        if (row == nRow and col == nCol):
            spectrogram = spectrogram *100
        return spectrogram

    def Extract_Pitch(row, col,nRow,nCol):

        pitch_List = []
        sample_rate = 44100
        x = np.zeros(44100)
        for i in range(44100):
            x[i] = np.sin(2. * np.pi * i * 225. / sample_rate)

        # create pitch object
        p = aubio.pitch("yin", samplerate=sample_rate)

        # pad end of input vector with zeros
        pad_length = p.hop_size - x.shape[0] % p.hop_size
        x_padded = np.pad(x, (0, pad_length), 'constant', constant_values=0)
        # to reshape it in blocks of hop_size
        x_padded = x_padded.reshape(-1, p.hop_size)

        # input array should be of type aubio.float_type (defaults to float32)
        x_padded = x_padded.astype(aubio.float_type)

        # if letter in b'G':


        for frame, i in zip(x_padded, range(len(x_padded))):
            time_str = "%.2f" % (i * p.hop_size / float(sample_rate))
            pitch_candidate = p(frame)[0] + row + col + 100
            # print(pitch_candidate)
            pitch_List.append(pitch_candidate)

        return pitch_List

    def Extract_Raw_Data (row, col,nRow,nCol):
        sound = AudioSegment.from_wav("test.wav")
        raw_data = sound._data
        if(row == nRow and col == nCol):
            raw_data=raw_data*5
        return raw_data