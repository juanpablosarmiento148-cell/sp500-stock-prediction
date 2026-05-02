import cv2
import os
from PIL import Image
import numpy as np

def convert_gif_to_mp4_smooth(gif_file, fps=24, output_name=None):
    if not os.path.exists(gif_file):
        print(f"File not found: {gif_file}")
        return
    
    if output_name is None:
        output_name = gif_file.replace('.gif', f'_smooth.mp4')
    
    print(f"Converting {gif_file} to {output_name} at {fps} FPS...")
    
    gif = Image.open(gif_file)
    frames = []
    
    try:
        while True:
            frames.append(gif.copy())
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    
    if frames:
        first = np.array(frames[0])
        h, w = first.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_name, fourcc, fps, (w, h))
        
        for frame in frames:
            out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        
        out.release()
        size = os.path.getsize(output_name) / 1024
        print(f"Saved: {output_name} ({size:.1f} KB)")

def main():
    print("=" * 50)
    print("Converting GIFs to Smooth MP4 (24 FPS)")
    print("=" * 50)
    
    files = [
        'taylor_sin.gif',
        'taylor_cos.gif', 
        'taylor_ex.gif',
        'taylor_log(1+x).gif',
        'taylor_combined.gif'
    ]
    
    fps = 24
    
    for f in files:
        convert_gif_to_mp4_smooth(f, fps)
    
    print("\nDone! All videos converted at 24 FPS")

if __name__ == "__main__":
    main()