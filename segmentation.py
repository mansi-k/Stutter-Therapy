from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa
import pandas as pd
import os

common_path = "/home/mansi/anaconda3/beproject/stutter_det/ache_2/"
print(common_path)
flag = 0
flag2 = 0
for subdir1, dirs1, files1 in os.walk(common_path):
    if(flag == 0):
        flag = 1        
        for subd1 in dirs1:
            print(subd1)
            flag2 = 0
            for subdir, dirs, files in os.walk(common_path + "/" + subd1):
                if(flag2 == 0):
                    flag2 = 1                    
                    for subd in dirs:
                        
                        for filename in os.listdir(common_path + "/" + subd1 + "/" + subd):
                            #print(filename)
                            if os.path.isdir(common_path + "/" + subd1 + "/" + subd + "/" +filename) : 
                                break
                            #break if is dir
                            sound_file = AudioSegment.from_wav(common_path + "/" + subd1 + "/" + subd + "/" + filename)
                            
                            
                            audio_chunks = sound_file[::1000]

                            print(common_path + "/" + subd1 + "/" +subd  + "/" +filename)
                            if not os.path.isdir(common_path + "/" +subd1 + "/" + subd + "/" +filename + "_segments"):
                                os.makedirs(common_path + "/" +subd1 + "/" + subd + "/" +filename + "_segments")

                            for i, chunk in enumerate(audio_chunks):
                                out_file = common_path + "/" + subd1 + "/" + subd + "/" + filename +"_segments"+ "/chunks{0}.wav".format(i)
                                print("exporting", out_file)
                                chunk.export(out_file, format="wav")
                                print(out_file)
                                y, sr = librosa.load(out_file)
                                #hop_length = 128

                                # Separate harmonics and percussives into two waveforms
                                #y_harmonic, y_percussive = librosa.effects.hpss(y)

                                # Beat track on the percussive signal
                                #tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)
                                # Compute MFCC features from the raw signal
                                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                                delta = librosa.feature.delta(mfcc, order=2, mode='constant')

                                # And the first-order differences (delta features)
                                #mfcc_delta = librosa.feature.delta(mfcc,  mode='nearest')
                                print(delta.shape)

                                #df = pd.DataFrame(mfcc.flatten('F'))
                                df = pd.DataFrame(delta)
                                #df.to_csv("file_path.csv")
                          
                                df.to_csv(common_path + "/" + subd1 + "/" + subd + "/" + filename +"_segments"+ "/chunks{0}.csv".format(i))

                        
