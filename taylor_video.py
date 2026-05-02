import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
from PIL import Image

plt.style.use('dark_background')

COLORS = {
    'background': '#0a0a0f',
    'grid': '#1a1a2e',
    'grid_major': '#2a2a4e',
    'axis': '#4a4a6e',
    'sin': '#00ffff',
    'cos': '#ff00ff',
    'exp': '#00ff88',
    'log': '#ff8800',
    'text': '#8888aa',
    'title': '#ffffff',
}

def taylor_sin(x, n_terms, center=0.0):
    result = 0.0
    for n in range(n_terms):
        coef = ((-1) ** n) / math.factorial(2 * n + 1)
        result += coef * ((x - center) ** (2 * n + 1))
    return result

def taylor_cos(x, n_terms, center=0.0):
    result = 0.0
    for n in range(n_terms):
        coef = ((-1) ** n) / math.factorial(2 * n)
        result += coef * ((x - center) ** (2 * n))
    return result

def taylor_exp(x, n_terms, center=0.0):
    result = 0.0
    for n in range(n_terms):
        coef = 1 / math.factorial(n)
        result += coef * ((x - center) ** n)
    return result

def setup_axes(ax, xlim=(-7, 7), ylim=(-4, 4)):
    ax.set_facecolor(COLORS['background'])
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['axis'])
        spine.set_linewidth(1.5)
    
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 1))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1, 1))
    
    ax.tick_params(colors=COLORS['text'], labelsize=8)
    
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-', linewidth=0.5)
    ax.grid(True, which='major', alpha=0.5, color=COLORS['grid_major'], linestyle='-', linewidth=0.8)
    
    ax.axhline(y=0, color=COLORS['axis'], linewidth=1.5, alpha=0.8)
    ax.axvline(x=0, color=COLORS['axis'], linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('x', color=COLORS['text'], fontsize=12, fontweight='bold')
    ax.set_ylabel('f(x)', color=COLORS['text'], fontsize=12, fontweight='bold')

def create_frame(x_range, y_actual, y_approx, title, color, term_count):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])
    
    setup_axes(ax)
    
    ax.plot(x_range, y_actual, color=color, linewidth=2.5, alpha=0.4, label='Actual')
    ax.plot(x_range, y_approx, color=color, linewidth=3, alpha=0.95, label='Taylor')
    
    ax.fill_between(x_range, y_actual, y_approx, color=color, alpha=0.15)
    
    ax.set_title(f'{title}\nTerm: n = {term_count}', 
               color=COLORS['title'], fontsize=18, fontweight='bold', pad=20)
    
    legend = ax.legend(loc='upper right', facecolor=COLORS['background'], 
                     edgecolor=COLORS['text'], fontsize=11)
    legend.get_frame().set_alpha(0.8)
    for text in legend.get_texts():
        text.set_color(COLORS['text'])
    
    plt.tight_layout()
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

def animate_sin():
    print("Animating sin(x) Taylor Series...")
    x_range = np.linspace(-2 * math.pi, 2 * math.pi, 500)
    y_actual = np.sin(x_range)
    
    frames = []
    max_terms = 15
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])
    
    for term in range(1, max_terms + 1):
        ax.clear()
        setup_axes(ax)
        
        y_approx = np.array([taylor_sin(x, term) for x in x_range])
        
        ax.plot(x_range, y_actual, color=COLORS['sin'], linewidth=2, alpha=0.3, label='sin(x)')
        ax.plot(x_range, y_approx, color=COLORS['sin'], linewidth=3, label=f'Taylor (n={term})')
        ax.fill_between(x_range, y_actual, y_approx, color=COLORS['sin'], alpha=0.15)
        
        ax.set_title(f'sin(x) Taylor Series Expansion\nTerm: n = {term}', 
                   color=COLORS['title'], fontsize=18, fontweight='bold', pad=20)
        
        legend = ax.legend(loc='upper right', facecolor=COLORS['background'], 
                       edgecolor=COLORS['text'], fontsize=11)
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_color(COLORS['text'])
        
        plt.tight_layout()
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        ax.clear()
        plt.close(fig)
    
    return frames

def animate_cos():
    print("Animating cos(x) Taylor Series...")
    x_range = np.linspace(-2 * math.pi, 2 * math.pi, 500)
    y_actual = np.cos(x_range)
    
    frames = []
    max_terms = 15
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])
    
    for term in range(1, max_terms + 1):
        ax.clear()
        setup_axes(ax)
        
        y_approx = np.array([taylor_cos(x, term) for x in x_range])
        
        ax.plot(x_range, y_actual, color=COLORS['cos'], linewidth=2, alpha=0.3, label='cos(x)')
        ax.plot(x_range, y_approx, color=COLORS['cos'], linewidth=3, label=f'Taylor (n={term})')
        ax.fill_between(x_range, y_actual, y_approx, color=COLORS['cos'], alpha=0.15)
        
        ax.set_title(f'cos(x) Taylor Series Expansion\nTerm: n = {term}', 
                   color=COLORS['title'], fontsize=18, fontweight='bold', pad=20)
        
        legend = ax.legend(loc='upper right', facecolor=COLORS['background'], 
                       edgecolor=COLORS['text'], fontsize=11)
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_color(COLORS['text'])
        
        plt.tight_layout()
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        ax.clear()
        plt.close(fig)
    
    return frames

def animate_exp():
    print("Animating e^x Taylor Series...")
    x_range = np.linspace(-3, 3, 500)
    y_actual = np.exp(x_range)
    
    frames = []
    max_terms = 15
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])
    
    for term in range(1, max_terms + 1):
        ax.clear()
        setup_axes(ax, xlim=(-3, 3), ylim=(-1, 20))
        
        y_approx = np.array([taylor_exp(x, term) for x in x_range])
        
        ax.plot(x_range, y_actual, color=COLORS['exp'], linewidth=2, alpha=0.3, label='e^x')
        ax.plot(x_range, y_approx, color=COLORS['exp'], linewidth=3, label=f'Taylor (n={term})')
        ax.fill_between(x_range, y_actual, y_approx, color=COLORS['exp'], alpha=0.15)
        
        ax.set_title(f'e^x Taylor Series Expansion\nTerm: n = {term}', 
                   color=COLORS['title'], fontsize=18, fontweight='bold', pad=20)
        
        legend = ax.legend(loc='upper left', facecolor=COLORS['background'], 
                       edgecolor=COLORS['text'], fontsize=11)
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_color(COLORS['text'])
        
        plt.tight_layout()
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        ax.clear()
        plt.close(fig)
    
    return frames

def create_intro_frame():
    print("Creating intro frame...")
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])
    
    setup_axes(ax)
    
    ax.set_title('TAYLOR SERIES\nVisualization', 
               color=COLORS['title'], fontsize=32, fontweight='bold', pad=40)
    
    functions = [
        ('sin(x)', COLORS['sin']),
        ('cos(x)', COLORS['cos']),
        ('e^x', COLORS['exp']),
    ]
    
    for i, (func, color) in enumerate(functions):
        x_sample = np.linspace(-5, 5, 200)
        if func == 'sin(x)':
            y_sample = np.sin(x_sample)
        elif func == 'cos(x)':
            y_sample = np.cos(x_sample)
        else:
            y_sample = np.exp(x_sample)
        
        ax.plot(x_sample, y_sample, color=color, linewidth=2, alpha=0.7, label=func)
    
    legend = ax.legend(loc='upper left', facecolor=COLORS['background'], 
                   edgecolor=COLORS['text'], fontsize=14)
    legend.get_frame().set_alpha(0.8)
    for text in legend.get_texts():
        text.set_color(COLORS['text'])
    
    plt.tight_layout()
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return image

def save_video(frames, filename, fps=10, duration=2):
    print(f"Saving video: {filename}...")
    
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available, creating GIF instead...")
        create_gif(frames, filename.replace('.mp4', '.gif'))
        return
    
    import io
    images = []
    
    for frame in frames:
        images.append(Image.fromarray(frame))
    
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=duration * 1000,
        fps=fps
    )
    print(f"Saved: {filename}")

def create_gif(frames, filename, duration=500):
    print(f"Creating GIF: {filename}...")
    
    from PIL import Image
    
    images = [Image.fromarray(frame) for frame in frames]
    
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved: {filename}")

def save_all_videos():
    print("=" * 60)
    print("Creating Taylor Series Animation Videos")
    print("=" * 60)
    
    intro_frames = [create_intro_frame()] * 5
    
    sin_frames = animate_sin()
    save_all_videos(sin_frames, 'taylor_sin.gif', 300)
    
    cos_frames = animate_cos()
    save_all_videos(cos_frames, 'taylor_cos.gif', 300)
    
    exp_frames = animate_exp()
    save_all_videos(exp_frames, 'taylor_exp.gif', 300)
    
    print("\nAll animations complete!")
    print("Generated files:")
    print("  - taylor_sin.gif")
    print("  - taylor_cos.gif")
    print("  - taylor_exp.gif")

def save_all_videos(frames, filename, duration=300):
    from PIL import Image
    
    images = [Image.fromarray(frame) for frame in frames]
    
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved: {filename}")

def create_combined_animation():
    print("\n" + "=" * 60)
    print("Creating Combined Animation")
    print("=" * 60)
    
    x_range_sin = np.linspace(-2 * math.pi, 2 * math.pi, 400)
    x_range_cos = np.linspace(-2 * math.pi, 2 * math.pi, 400)
    x_range_exp = np.linspace(-3, 3, 400)
    
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    fig.suptitle('TAYLOR SERIES EVOLUTION IN CALCULUS', 
              color=COLORS['title'], fontsize=24, fontweight='bold', y=0.95)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    axes = [ax1, ax2, ax3, ax4]
    
    frames = []
    max_terms = 12
    
    for term in range(1, max_terms + 1):
        for ax in axes:
            ax.clear()
            ax.set_facecolor(COLORS['background'])
        
        setup_axes(ax1, xlim=(-7, 7), ylim=(-2, 2))
        setup_axes(ax2, xlim=(-7, 7), ylim=(-2, 2))
        setup_axes(ax3, xlim=(-3, 3), ylim=(-1, 20))
        setup_axes(ax4, xlim=(-7, 7), ylim=(-2, 2))
        
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_color(COLORS['axis'])
        
        y_sin_actual = np.sin(x_range_sin)
        y_sin_approx = np.array([taylor_sin(x, term) for x in x_range_sin])
        ax1.plot(x_range_sin, y_sin_actual, color=COLORS['sin'], linewidth=1.5, alpha=0.3)
        ax1.plot(x_range_sin, y_sin_approx, color=COLORS['sin'], linewidth=2.5)
        ax1.set_title('sin(x)', color=COLORS['sin'], fontsize=14, fontweight='bold')
        
        y_cos_actual = np.cos(x_range_cos)
        y_cos_approx = np.array([taylor_cos(x, term) for x in x_range_cos])
        ax2.plot(x_range_cos, y_cos_actual, color=COLORS['cos'], linewidth=1.5, alpha=0.3)
        ax2.plot(x_range_cos, y_cos_approx, color=COLORS['cos'], linewidth=2.5)
        ax2.set_title('cos(x)', color=COLORS['cos'], fontsize=14, fontweight='bold')
        
        y_exp_actual = np.exp(x_range_exp)
        y_exp_approx = np.array([taylor_exp(x, term) for x in x_range_exp])
        ax3.plot(x_range_exp, y_exp_actual, color=COLORS['exp'], linewidth=1.5, alpha=0.3)
        ax3.plot(x_range_exp, y_exp_approx, color=COLORS['exp'], linewidth=2.5)
        ax3.set_title('e^x', color=COLORS['exp'], fontsize=14, fontweight='bold')
        
        x_log = np.linspace(-0.95, 0.95, 200)
        x_log = x_log[x_log > -1]
        y_log_actual = np.log(x_log + 1)
        y_log_actual = y_log_actual[np.isfinite(y_log_actual)]
        
        y_log_approx_valid = []
        x_log_valid = []
        for x_val in x_log:
            if x_val > -1 and abs(x_val) < 1:
                val = sum(((-1) ** (n + 1)) / n * x_val ** n for n in range(1, term + 1))
                if np.isfinite(val):
                    y_log_approx_valid.append(val)
                    x_log_valid.append(x_val)
        if x_log_valid:
            ax4.plot(x_log_valid, y_log_approx_valid, color=COLORS['log'], linewidth=2.5)
        
        ax4.plot(x_log[x_log > -1], np.log(x_log[x_log > -1] + 1), color=COLORS['log'], linewidth=1.5, alpha=0.3)
        ax4.set_title('log(1+x)', color=COLORS['log'], fontsize=14, fontweight='bold')
        
        for ax in axes:
            ax.set_xlabel('x', color=COLORS['text'], fontsize=10)
            ax.set_ylabel('f(x)', color=COLORS['text'], fontsize=10)
        
        fig.text(0.5, 0.02, f'Term: n = {term}', ha='center', 
                color=COLORS['title'], fontsize=16, fontweight='bold')
        
        buf = BytesIO()
        fig.savefig(buf, format='png', facecolor=COLORS['background'], dpi=100)
        buf.seek(0)
        image = np.array(Image.open(buf))
        frames.append(image)
        buf.close()
        
        print(f"Frame {term}/{max_terms}")
    
    plt.close(fig)
    
    print("Saving combined animation...")
    
    images = [Image.fromarray(frame) for frame in frames]
    
    images[0].save(
        'taylor_combined.gif',
        save_all=True,
        append_images=images[1:],
        duration=400,
        loop=0
    )
    print("Saved: taylor_combined.gif")

if __name__ == "__main__":
    create_combined_animation()