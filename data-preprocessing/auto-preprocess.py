from rgb import run_rgb_extraction
from opticalflow import run_optical_flow_extraction
from audio import run_audio_extraction
from soundcheck import run_silence_detection
import time

# 처리할 클래스 리스트 관리
classes_to_process = [
    "adult+male+singing",
    "adult+male+speaking",
    "applauding",
    "ascending",
    "asking",
    "assembling",
    "autographing",
    "baking",
    "balancing",
    "barbecuing",
    "barking",
    "bending",
    "bicycling",
    "biting",
    "blowing",
    "boarding",
    "boating",
    "boiling",
    "bowing",
    "bowling",
    "breaking",
    "brushing",
    "bubbling",
    "building",
    "bulldozing",
    "burying",
    "buying",
    "calling",
    "camping",
    "carrying",
    "carving",
    "catching",
    "chasing",
    "cheering",
    "chewing",
    "child+singing",
    "child+speaking",
    "clapping",
    "clawing",
    "cleaning",
    "clearing",
    "climbing",
    "clinging",
    "clipping",
    "closing",
    "coaching",
    "combing",
    "combusting"
]

def run_preprocessing_pipeline(classes):
    total_start_time = time.time()

    steps = [
        ("Step 1: RGB 프레임 추출", run_rgb_extraction)
        ("Step 2: Optical Flow 추출", run_optical_flow_extraction),
        ("Step 3: 오디오 추출", run_audio_extraction),
        ("Step 4: 무음 감지 및 삭제", run_silence_detection)
    ]

    for step_name, func in steps:
        print(f"\n{step_name} 시작")
        start_time = time.time()
        func(classes)
        print(f"{step_name} 완료 ({round(time.time() - start_time, 2)}초)")

    total_elapsed_time = time.time() - total_start_time
    minutes, seconds = divmod(total_elapsed_time, 60)
    print(f"\n전체 전처리 파이프라인 완료 (총 {int(minutes)}분 {int(seconds)}초)")

if __name__ == "__main__":
    print("전체 전처리 파이프라인 실행 시작...")
    run_preprocessing_pipeline(classes_to_process)
