
import os
import argparse
import json
import random
import csv
from pydub import AudioSegment

def main():
    data = []
    file_path = '/home/ca/Downloads/project/nlpai/speechrecognition/cv-corpus-14.0-delta-2023-06-23/en/validated.tsv'
    save_json_path = '/home/ca/Downloads/project/nlpai/speechrecognition/cv-corpus-14.0-delta-2023-06-23/en'
    percent = 10
    convert = True

    directory = file_path.rpartition('/')[0]

    with open(file_path) as f:
        length = sum(1 for line in f)

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        index = 1
        if convert:
            print(str(length) + " files found")
        for row in reader:
            file_name = row['path']
            filename = file_name.rpartition('.')[0] + ".wav"
            text = row['sentence']
            if convert:
                data.append({
                    "key": directory + "/clips/" + filename,
                    "text": text
                })
                print("converting file " + str(index) + "/" + str(length) + " to wav", end="\r")
                src = directory + "/clips/" + file_name
                dst = directory + "/clips/" + filename
                sound = AudioSegment.from_mp3(src)
                sound.export(dst, format="wav")
                index = index + 1
            else:
                data.append({
                    "key": directory + "/clips/" + file_name,
                    "text": text
                })

    random.shuffle(data)
    print("creating JSONs")

    with open(save_json_path + "/train.json", "w") as f:
        d = len(data)
        i = 0
        while i < int(d - d / percent):
            r = data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i + 1

    with open(save_json_path + "/test.json", "w") as f:
        d = len(data)
        i = int(d - d / percent)
        while i < d:
            r = data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i + 1

    print("Done!")

if __name__ == "__main__":
    main()