import os
import subprocess
import pandas as pd
import time
import csv

'''
실행 시 data 경로를 로컬 환경에 맞게 수정해주세요.
'''
raw_data_dir = r"C:/Users/swu/Desktop/원본비디오"
csv_base_dir = r"D:/RGB"
audio_output_dir = r"D:/Audio"
ffmpeg_path = r"C:/Users/swu/Desktop/ffmpeg-2024-12-16-git-d2096679d5-essentials_build/ffmpeg-2024-12-16-git-d2096679d5-essentials_build/bin/ffmpeg.exe"

def is_already_logged(video_id, action_class, csv_path):
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as file:
            return any(f"{video_id},{action_class}" in line for line in file)
    return False

def extract_audio(video_path, audio_output_path):
    try:
        result = subprocess.run([
            ffmpeg_path, "-i", video_path,
            "-ar", "16000", "-ac", "1",    
            "-acodec", "pcm_s16le",               
            audio_output_path, "-y"                
        ], check=True, stderr=subprocess.PIPE)
        
        print(f"Extracted audio: {audio_output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {video_path}:\n{e.stderr.decode()}")
        return False

def process_videos(phase, classes_to_process):
    csv_path = os.path.join(csv_base_dir, phase, "18frames_videos.csv")
    audio_phase_dir = os.path.join(audio_output_dir, phase)
    os.makedirs(audio_phase_dir, exist_ok=True)

    no_audio_csv_path = os.path.join(audio_phase_dir, "18frames_no_audios.csv")
    if not os.path.exists(no_audio_csv_path):
        with open(no_audio_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "actionclass"])  

    no_audio_videos = [] 

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        video_id = row["id"]
        action_class = row["actionclass"]

        if action_class not in classes_to_process:
            continue

        video_path = os.path.join(raw_data_dir, phase, action_class, f"{video_id}.mp4")
        audio_output_path = os.path.join(audio_phase_dir, action_class, "18frames", f"{video_id}.wav")

        if os.path.exists(audio_output_path):
            print(f"Skipping {video_id} (audio already extracted)")
            continue

        os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)

        if os.path.exists(video_path):
            success = extract_audio(video_path, audio_output_path)

            if not success and not is_already_logged(video_id, action_class, no_audio_csv_path):
                no_audio_videos.append({"id": video_id, "actionclass": action_class})

        else:
            print(f"Video not found: {video_path}")
            if not is_already_logged(video_id, action_class, no_audio_csv_path):
                no_audio_videos.append({"id": video_id, "actionclass": action_class})

    if no_audio_videos:
        with open(no_audio_csv_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["id", "actionclass"])
            writer.writerows(no_audio_videos)
        print(f"Saved no-audio video list to: {no_audio_csv_path}")
    else:
        print(f"All videos in {phase} processed successfully.")

def run_audio_extraction(classes_to_process):
    print(f"오디오 추출 시작: {classes_to_process}")
    start_time = time.time()  

    for phase in ["training", "validation"]:
        process_videos(phase, classes_to_process)

    end_time = time.time() 
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print(f"오디오 추출 완료 ({int(minutes)}분 {int(seconds)}초)")

if __name__ == "__main__":
    run_audio_extraction(["default_class"])
