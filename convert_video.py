import os
import subprocess

def convert_gif_to_mp4(gif_file, output_name=None):
    if not os.path.exists(gif_file):
        print(f"File not found: {gif_file}")
        return
    
    if output_name is None:
        output_name = gif_file.replace('.gif', '.mp4')
    
    print(f"Converting {gif_file} to {output_name}...")
    
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-framerate', '10',
            '-i', gif_file,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'fast',
            output_name
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            size = os.path.getsize(output_name) / 1024
            print(f"Saved: {output_name} ({size:.1f} KB)")
        else:
            print(f"FFmpeg error: {result.stderr}")
            convert_with_cv2(gif_file, output_name)
    except FileNotFoundError:
        print("ffmpeg not found, trying with opencv...")
        convert_with_cv2(gif_file, output_name)

def convert_with_cv2(gif_file, output_name):
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
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
            out = cv2.VideoWriter(output_name, fourcc, 10, (w, h))
            
            for frame in frames:
                out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
            
            out.release()
            size = os.path.getsize(output_name) / 1024
            print(f"Saved: {output_name} ({size:.1f} KB)")
    except ImportError:
        print("opencv-python not installed")

def main():
    print("=" * 50)
    print("Converting GIFs to MP4")
    print("=" * 50)
    
    files = [
        'taylor_sin.gif',
        'taylor_cos.gif', 
        'taylor_ex.gif',
        'taylor_log(1+x).gif',
        'taylor_combined.gif'
    ]
    
    for f in files:
        convert_gif_to_mp4(f)
    
    print("\nDone!")

if __name__ == "__main__":
    main()