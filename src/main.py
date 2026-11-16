from pathlib import Path

from pipeline import PCBDetectionPipeline


base = "/home/blumm/Documents/dev/DeepPCB/PCBData/group00041/00041"
img_id = "00041003"
ref = Path(base, f"{img_id}_temp.jpg")
target = Path(base, f"{img_id}_test.jpg")

PCBDetectionPipeline().process(ref, target)
