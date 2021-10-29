"""
dataset_path : path to the dataset
json_path: path of the json where we store informations
num_segment = num of segments in which we divide
"""
import os
import librosa
import math
import json

DATASET_PATH ="Musical_genres"
JSON_PATH ="data.json"
SAMPLE_RATE = 22050
DURATION = 30 #duration in second of each trac in the data set
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=20, hop_length=512, num_segment=5):
    # dictonary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segment)
    expected_num_fcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)  #e.g. 1.2 -> 2

    # loop through generes (folders inside the dataset)
    for i,(dirpath, dirnames,filenames) in enumerate(os.walk(dataset_path)):

        #ensure that we are not at the root level (dataset level), we want to be at the gerne level
        if dirpath is not dataset_path:

            #save semantic level
            dirpath_components = dirpath.split("/") #genre/blues => ["genre","blues"]
            semantic_label = dirpath_components[-1] #last index
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            #process files for a specific genre
            for f in filenames:
                file_path = os.path.join(dirpath,f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                #process segments extracting mfcc and storing data
                for s in range(num_segment):
                    start_sample = num_samples_per_segment * s
                    finish_sample= start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr= SAMPLE_RATE,
                                                n_fft = n_fft,
                                                n_mfcc= n_mfcc,
                                                hop_length = hop_length)
                    mfcc = mfcc.T   #transpose the matrix

                    # store mfcc for segment if it has expected legnth
                    if len(mfcc) == expected_num_fcc_vectors_per_segment :
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1) #i-1 to ignore the first iteration on the root folder
                        print("{},segment:{}".format(file_path,s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


#if __name__ == "main":
save_mfcc(DATASET_PATH,JSON_PATH, num_segment=10)
