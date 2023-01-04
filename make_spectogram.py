import librosa
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from librosa import display


FRAME_LEN = 0.025
FRAME_STRIDE = 0.010


def make_melspecrogram(wav_file, show_mode=False):
    # load wav file
    y, sr = librosa.load(wav_file, sr=44100)

    input_nfft = int(round(sr * FRAME_LEN))
    input_stride = int(round(sr * FRAME_STRIDE))

    S = librosa.feature.melspectrogram(
        y=y,
        n_mels=40,
        n_fft=input_nfft,
        hop_length=input_stride,
    )
    if show_mode:
        ## Showing
        plt.figure(figsize=(10, 4))

        # display
        display.specshow(
            librosa.power_to_db(S, ref=np.max),
            sr=sr,
            hop_length=input_stride,
            
        )
        # plt.colorbar(format="%+2.0f dB")
        
        plt.tight_layout(pad=0)
        plt.savefig(f"{wav_file}_melspectogram.png", bbox_inches='tight', pad_inches=0)
        
        print(f"Figure is Saved at..: {wav_file}_melspectogram.png")
        # plt.show()

    return S


if __name__ == "__main__":
    print("🚀Making Melspectogram...🚀")

    wav_lst = glob("data/*.wav")
    S_lst = []
    for wav_file in wav_lst:
        print(f"👀 Now.. {wav_file}")
        S = make_melspecrogram(wav_file, show_mode=True)
        S_lst.append(S)
