## 1. Download Source Videos
First we would like to thank the work from [Music-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA) and [AVQA](https://mn.cs.tsinghua.edu.cn/avqa/) for providing data.
In this step, download the original video based on the link provided in the work above.

## 2. Align with the AVinstruct
A sample of AVinstruct is shown below:
```bash
    {
        "video_id": "00005580",
        "multi_choice": "",
        "correct_answer": "",
        "video": "/mnt/sda/imagebind_v_feats/00005580.npy", # visual-features
        "audio": "/mnt/sda/imagebind_a_feats/00005580.mp3.npy", # audio-features
        "conversations": [
            {
                "from": "human",
                "value": "Summarize the video to help answer:\n<Q>How many instruments are sounding in the video?<Q>\n<video>"
            },
            {
                "from": "gpt",
                "value": "The video shows three people playing violin and cello in a park. Two instruments are sounding in the video, the violin and the cello."
            }
        ]
    },
```
where video_id is used to retrieve the name of the downloaded video. We recommend using a frozen encoder to obtain the corresponding audiovisual embedding and save it as an npy file.
