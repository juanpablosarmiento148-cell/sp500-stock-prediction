import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import os

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

def taylor_log(x, n_terms, center=1.0):
    if abs(x - center) >= 1:
        return float('nan')
    x_rel = x - center
    result = 0.0
    for n in range(1, n_terms + 1):
        coef = ((-1) ** (n + 1)) / n
        result += coef * (x_rel ** n)
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

def create_function_animation(func_name, taylor_func, x_range, y_actual, ylim, color):
    print(f"Creating {func_name} animation...")
    
    frames = []
    max_terms = 12
    
    for term in range(1, max_terms + 1):
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
        fig.patch.set_facecolor(COLORS['background'])
        
        setup_axes(ax, xlim=(x_range[0], x_range[-1]), ylim=ylim)
        
        y_approx = np.array([taylor_func(x, term) for x in x_range])
        y_approx = np.nan_to_num(y_approx, nan=0.0)
        
        ax.plot(x_range, y_actual, color=color, linewidth=2, alpha=0.3, label=f'{func_name}')
        ax.plot(x_range, y_approx, color=color, linewidth=3, label=f'Taylor (n={term})')
        
        if func_name != 'e^x':
            ax.fill_between(x_range, y_actual, y_approx, color=color, alpha=0.15)
        
        ax.set_title(f'{func_name} Taylor Series\nn = {term}', 
                   color=COLORS['title'], fontsize=22, fontweight='bold', pad=20)
        
        ax.set_xlabel('x', color=COLORS['text'], fontsize=14, fontweight='bold')
        ax.set_ylabel('f(x)', color=COLORS['text'], fontsize=14, fontweight='bold')
        
        legend = ax.legend(loc='best', facecolor=COLORS['background'], 
                         edgecolor=COLORS['text'], fontsize=12)
        legend.get_frame().set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color(COLORS['text'])
        
        buf = BytesIO()
        fig.savefig(buf, format='png', facecolor=COLORS['background'], dpi=100)
        buf.seek(0)
        image = np.array(Image.open(buf))
        frames.append(image)
        buf.close()
        
        plt.close(fig)
        
        print(f"  Frame {term}/{max_terms}")
    
    print(f"Saving {func_name} animation...")
    
    images = [Image.fromarray(frame) for frame in frames]
    
    filename = f'taylor_{func_name.lower().replace("^", "").replace("(x)", "")}.gif'
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=400,
        loop=0
    )
    print(f"Saved: {filename}")
    
    file_size = os.path.getsize(filename) / 1024
    print(f"  Size: {file_size:.1f} KB")
    
    return frames

def main():
    print("=" * 60)
    print("Taylor Series Animation Generator")
    print("Dark Futuristic Theme")
    print("=" * 60)
    
    create_function_animation(
        'sin(x)',
        taylor_sin,
        np.linspace(-2 * np.pi, 2 * np.pi, 400),
        np.sin(np.linspace(-2 * np.pi, 2 * np.pi, 400)),
        (-2, 2),
        COLORS['sin']
    )
    
    create_function_animation(
        'cos(x)',
        taylor_cos,
        np.linspace(-2 * np.pi, 2 * np.pi, 400),
        np.cos(np.linspace(-2 * np.pi, 2 * np.pi, 400)),
        (-2, 2),
        COLORS['cos']
    )
    
    create_function_animation(
        'e^x',
        taylor_exp,
        np.linspace(-3, 3, 400),
        np.exp(np.linspace(-3, 3, 400)),
        (-1, 20),
        COLORS['exp']
    )
    
    create_function_animation(
        'log(1+x)',
        taylor_log,
        np.linspace(-0.95, 0.95, 300),
        np.log(np.linspace(-0.95, 0.95, 300) + 1),
        (-4, 2),
        COLORS['log']
    )
    
    print("\n" + "=" * 60)
    print("All animations complete!")
    print("=" * 60)
    print("Generated files:")
    print("  - taylor_sin.gif")
    print("  - taylor_cos.gif")
    print("  - taylor_exp.gif")
    print("  - taylor_log.gif")
    print("  - taylor_combined.gif")

if __name__ == "__main__":
    main()