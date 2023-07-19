import os
import json
import random

def main():
    data = []
    directory = "/home/ca/Downloads/project/nlpai/speechrecognition/cv-corpus-14.0-delta-2023-06-23/en/clips"
    filetxtname = "/home/ca/Downloads/project/nlpai/speechrecognition/cv-corpus-14.0-delta-2023-06-23/en/times"
    percent = 10
    

    with open(os.path.join(directory, filetxtname + ".txt")) as f: 
        for line in f: 
            line = line.strip()
            file_name, text = line.split('=')
            file_name = file_name.strip()
            text = text.strip()
            data.append({
                "key": file_name,
                "text": text
            })

    random.shuffle(data)

    train_data = data[:int(len(data) - len(data) / percent)]
    test_data = data[int(len(data) - len(data) / percent):]

    with open("/home/ca/Downloads/project/nlpai/speechrecognition/cv-corpus-14.0-delta-2023-06-23/en/train.json", 'w') as f:
        for item in train_data:
            line = json.dumps(item)
            f.write(line + "\n")

    with open("/home/ca/Downloads/project/nlpai/speechrecognition/cv-corpus-14.0-delta-2023-06-23/en/test.json", 'w') as f:
        for item in test_data:
            line = json.dumps(item)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
