from common.utils import extract_visual_feature
from moviepy.editor import VideoFileClip

from scripts.video_processing import subclip

start = 5
end = 10
input_video_path = "10.mp4"
output_video_path = f"10_{start}_{end}.mp4"

subclip(input_video_path, output_video_path, fps=30, start=start, end=end)
