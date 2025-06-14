# from moviepy.editor import VideoFileClip

# # Load your video file
# video = VideoFileClip("custom_video.mp4")

# # Define the duration to crop (e.g., first 3 minutes = 180 seconds)
# start_time = 0
# end_time = 30 # 3 minutes

# # Trim the video
# cropped_video = video.subclip(start_time, end_time)

# # Save the cropped video
# cropped_video.write_videofile("cropped_video23.mp4", codec="libx264", audio_codec="aac",fps = 30)

from moviepy.editor import VideoFileClip

# Load the original 30fps video
video = VideoFileClip("C:/Users/ADMIN/Desktop/ahmad/data_train/test_video/output/custom_video.avi")

# Export the video with 60fps
video.write_videofile("output_60fps_video.mp4", codec="libx264", audio_codec="aac", fps=30)
