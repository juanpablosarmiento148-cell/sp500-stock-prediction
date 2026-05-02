import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple

class TaylorSeriesSimulation:
    def __init__(self, function: Callable, derivatives: List[Callable], center: float = 0.0):
        self.function = function
        self.derivatives = derivatives
        self.center = center

    def factorial(self, n: int) -> float:
        if n <= 1:
            return 1
        return math.factorial(n)

    def compute_coefficients(self, n_terms: int) -> List[float]:
        safe_n_terms = min(n_terms, len(self.derivatives))
        coefficients = []
        for i in range(safe_n_terms):
            deriv_value = self.derivatives[i](self.center)
            coef = deriv_value / self.factorial(i)
            coefficients.append(coef)
        return coefficients

    def approximate(self, x: float, n_terms: int) -> float:
        safe_n_terms = min(n_terms, len(self.derivatives))
        coefficients = self.compute_coefficients(safe_n_terms)
        result = 0.0
        for i, coef in enumerate(coefficients):
            result += coef * ((x - self.center) ** i)
        return result

    def compute_error(self, x: float, n_terms: int) -> float:
        if n_terms > len(self.derivatives):
            n_terms = len(self.derivatives)
        actual = self.function(x)
        approx = self.approximate(x, n_terms)
        return abs(actual - approx)


def taylor_exp_sin(x: float, n_terms: int, center: float = 0.0) -> float:
    result = 0.0
    for n in range(n_terms):
        coefficient = ((-1) ** n) / math.factorial(2 * n + 1)
        result += coefficient * ((x - center) ** (2 * n + 1))
    return result


def taylor_exp_cos(x: float, n_terms: int, center: float = 0.0) -> float:
    result = 0.0
    for n in range(n_terms):
        coefficient = ((-1) ** n) / math.factorial(2 * n)
        result += coefficient * ((x - center) ** (2 * n))
    return result


def taylor_exp_exp(x: float, n_terms: int, center: float = 0.0) -> float:
    result = 0.0
    for n in range(n_terms):
        coefficient = 1 / math.factorial(n)
        result += coefficient * ((x - center) ** n)
    return result


def taylor_exp_log(x: float, n_terms: int, center: float = 1.0) -> float:
    result = 0.0
    x_relative = x - center
    for n in range(1, n_terms + 1):
        coefficient = ((-1) ** (n + 1)) / n
        result += coefficient * (x_relative ** n)
    return result


def taylor_exp_expanded(x: float, n_terms: int, center: float = 0.0) -> float:
    result = 0.0
    for n in range(n_terms):
        coefficient = 1 / math.factorial(n)
        result += coefficient * ((x - center) ** n)
    return result


def demo_convergence_sin():
    print("=" * 60)
    print("Demo: Taylor Series Convergence for sin(x)")
    print("=" * 60)
    print("\nTaylor Series for sin(x) around a=0:")
    print("sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...")
    print("\nComparing actual vs approximation at x=pi/4:")
    
    x = math.pi / 4
    actual = math.sin(x)
    
    results = []
    for n in range(1, 10):
        approx = taylor_exp_sin(x, n)
        error = abs(actual - approx)
        print(f"n={n:2d}: sin({x:.4f}) ~ {approx:.10f}, error = {error:.2e}")
        results.append((n, approx, error))
    
    return results


def demo_convergence_exp():
    print("\n" + "=" * 60)
    print("Demo: Taylor Series for e^x")
    print("=" * 60)
    print("\nTaylor Series for e^x around a=0:")
    print("e^x = 1 + x + x^2/2! + x^3/3! + x^4/4! + ...")
    print("\nComparing actual vs approximation at x=1:")
    
    x = 1.0
    actual = math.exp(x)
    
    for n in range(1, 12):
        approx = taylor_exp_exp(x, n)
        error = abs(actual - approx)
        print(f"n={n:2d}: e^{x} ~ {approx:.10f}, error = {error:.2e}")


def demo_convergence_cos():
    print("\n" + "=" * 60)
    print("Demo: Taylor Series for cos(x)")
    print("=" * 60)
    print("\nTaylor Series for cos(x) around a=0:")
    print("cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...")
    print("\nComparing actual vs approximation at x=pi/3:")
    
    x = math.pi / 3
    actual = math.cos(x)
    
    for n in range(1, 10):
        approx = taylor_exp_cos(x, n)
        error = abs(actual - approx)
        print(f"n={n:2d}: cos({x:.4f}) ~ {approx:.10f}, error = {error:.2e}")


def demo_radius_of_convergence():
    print("\n" + "=" * 60)
    print("Demo: Radius of Convergence")
    print("=" * 60)
    print("\nFor sin(x), cos(x), e^x: radius = inf (entire real line)")
    print("For log(1+x): radius = 1 (converges only for |x| < 1)")
    print("\nTesting log(1+x) at x = 0.5:")
    
    x = 0.5
    actual = math.log(1 + x)
    
    for n in range(1, 15):
        approx = taylor_exp_log(x, n, center=1.0)
        error = abs(actual - approx)
        print(f"n={n:2d}: log({x}) ~ {approx:.10f}, error = {error:.2e}")
    
    print("\nTesting at x = 1.5 (outside radius):")
    x = 1.5
    actual = math.log(1 + x)
    
    for n in range(1, 15):
        approx = taylor_exp_log(x, n, center=1.0)
        error = abs(actual - approx)
        print(f"n={n:2d}: log({x}) ~ {approx:.10f}, error = {error:.2e}")


def visualize_approximations():
    print("\n" + "=" * 60)
    print("Visualizing Taylor Series Approximations")
    print("=" * 60)
    print("\nCreating visualization plots...")
    
    x_range = np.linspace(-2 * math.pi, 2 * math.pi, 500)
    x_plot = np.linspace(-2 * math.pi, 2 * math.pi, 500)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Taylor Series Approximations in Calculus', fontsize=14, fontweight='bold')
    
    ax1.plot(x_plot, np.sin(x_plot), 'b-', linewidth=2, label='sin(x)')
    colors = ['r', 'g', 'm', 'orange']
    for i, n in enumerate([1, 3, 5, 7]):
        terms = (n + 1) // 2
        y_approx = [taylor_exp_sin(x, terms) for x in x_range]
        ax1.plot(x_range, y_approx, colors[i], linestyle='--', linewidth=1.5, 
                label=f'Taylor (n={n})')
    ax1.set_title('sin(x) Taylor Series')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(x_plot, np.cos(x_plot), 'b-', linewidth=2, label='cos(x)')
    for i, n in enumerate([2, 4, 6, 8]):
        terms = n // 2
        y_approx = [taylor_exp_cos(x, terms) for x in x_range]
        ax2.plot(x_range, y_approx, colors[i], linestyle='--', linewidth=1.5,
                label=f'Taylor (n={n})')
    ax2.set_title('cos(x) Taylor Series')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    x_exp = np.linspace(-2, 2, 500)
    ax3.plot(x_exp, np.exp(x_exp), 'b-', linewidth=2, label='e^x')
    for i, n in enumerate([3, 5, 8, 10]):
        y_approx = [taylor_exp_exp(x, n) for x in x_exp]
        ax3.plot(x_range[:len(x_exp)], y_approx, colors[i], linestyle='--', 
                linewidth=1.5, label=f'Taylor (n={n})')
    ax3.set_title('e^x Taylor Series')
    ax3.set_xlabel('x')
    ax3.set_ylabel('f(x)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(x_plot, np.log(x_plot + 1), 'b-', linewidth=2, label='log(1+x)')
    x_zoom = np.linspace(-0.9, 0.9, 200)
    for i, n in enumerate([2, 4, 6, 8]):
        y_approx = [taylor_exp_log(x, n, center=1.0) if abs(x) < 1 else float('nan') 
                   for x in x_zoom]
        ax4.plot(x_zoom, y_approx, colors[i], linestyle='--', linewidth=1.5,
                label=f'Taylor (n={n})')
    ax4.set_title('log(1+x) Taylor Series')
    ax4.set_xlabel('x')
    ax4.set_ylabel('f(x)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('taylor_series_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved: taylor_series_visualization.png")


def visualize_convergence():
    print("\n" + "=" * 60)
    print("Visualizing Convergence Rate")
    print("=" * 60)
    
    x_values = [0.1, 0.5, 1.0, 1.5, 2.0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    for idx, x_val in enumerate(x_values):
        actual = math.sin(x_val)
        errors = []
        terms_range = range(1, 15)
        for n in terms_range:
            approx = taylor_exp_sin(x_val, n)
            error = abs(actual - approx)
            errors.append(error)
        ax1.semilogy(terms_range, errors, color=colors[idx], marker='o', 
                   label=f'x={x_val}', markersize=4)
    
    ax1.set_xlabel('Number of Terms (n)')
    ax1.set_ylabel('Absolute Error (log scale)')
    ax1.set_title('sin(x): Error Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for idx, x_val in enumerate(x_values):
        actual = math.exp(x_val)
        errors = []
        terms_range = range(1, 15)
        for n in terms_range:
            approx = taylor_exp_exp(x_val, n)
            error = abs(actual - approx)
            errors.append(error)
        ax2.semilogy(terms_range, errors, color=colors[idx], marker='o',
                   label=f'x={x_val}', markersize=4)
    
    ax2.set_xlabel('Number of Terms (n)')
    ax2.set_ylabel('Absolute Error (log scale)')
    ax2.set_title('e^x: Error Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('taylor_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved: taylor_convergence.png")


def demo_error_bound():
    print("\n" + "=" * 60)
    print("Demo: Error Bounds (Lagrange Remainder)")
    print("=" * 60)
    print("\nThe Lagrange remainder formula:")
    print("R_n(x) = f^(n+1)(xi) * (x-a)^(n+1) / (n+1)!")
    print("\nFor sin(x), |f^(n+1)(xi)| <= 1")
    print("So |R_n(x)| <= |x|^(n+1) / (n+1)!")
    print("\nError bounds for sin(x) at x=1:")
    
    x = 1.0
    actual = math.sin(x)
    
    for n in range(1, 10):
        approximate = taylor_exp_sin(x, n)
        actual_error = abs(actual - approximate)
        bound = abs(x) ** (n + 1) / math.factorial(n + 1)
        print(f"n={n}: actual error={actual_error:.2e}, bound={bound:.2e}")


def visualize_partial_sums():
    print("\n" + "=" * 60)
    print("Visualizing Partial Sums Building Up")
    print("=" * 60)
    print("\nShowing how each term contributes to the approximation...")
    
    x_range = np.linspace(-2 * math.pi, 2 * math.pi, 500)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(x_range, np.sin(x_range), 'b-', linewidth=2.5, label='sin(x)', alpha=0.8)
    
    cumulative = np.zeros_like(x_range)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    labels = ['x', '-x³/3!', '+x⁵/5!', '-x⁷/7!', '+x⁹/9!']
    
    for i in range(5):
        term_coef = ((-1) ** i) / math.factorial(2 * i + 1)
        x_powers = x_range ** (2 * i + 1)
        term = term_coef * x_powers
        cumulative = cumulative + term
        ax.plot(x_range, cumulative, linestyle='--', color=colors[i], 
               linewidth=1.5, label=f'Partial sum: {labels[i]}')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Building sin(x) Taylor Series Term by Term', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('taylor_partial_sums.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved: taylor_partial_sums.png")


def demo_class_usage():
    print("\n" + "=" * 60)
    print("Demo: TaylorSeriesSimulation Class Usage")
    print("=" * 60)
    
    def f(x):
        return math.sin(x)
    
    def d0(x):
        return math.sin(x)
    
    def d1(x):
        return math.cos(x)
    
    def d2(x):
        return -math.sin(x)
    
    def d3(x):
        return -math.cos(x)
    
    def d4(x):
        return math.sin(x)
    
    derivatives = [d0, d1, d2, d3, d4]
    
    sim = TaylorSeriesSimulation(f, derivatives, center=0.0)
    
    x = math.pi / 4
    print(f"\nApproximating sin(x) at x = {x:.4f}")
    
    max_terms = min(8, len(derivatives))
    for n in range(1, max_terms + 1):
        approx = sim.approximate(x, n)
        error = sim.compute_error(x, n)
        print(f"Terms: {n:2d} -> Approx: {approx:.10f}, Error: {error:.2e}")
    
    print(f"\nCoefficients at center a=0:")
    coeffs = sim.compute_coefficients(6)
    for i, c in enumerate(coeffs):
        print(f"  a_{i} = {c:.10f}")


def print_formula_sheet():
    print("\n" + "=" * 60)
    print("Common Taylor Series Formulas")
    print("=" * 60)
    print("""
    sin(x)  = sum (-1)^n x^(2n+1) / (2n+1)!  ,  a=0
    cos(x)  = sum (-1)^n x^(2n) / (2n)!      ,  a=0
    e^x    = sum x^n / n!                    ,  a=0
    log(1+x)= sum (-1)^(n+1) x^n / n         ,  a=0, |x|<1
    tan^-1(x)= sum (-1)^n x^(2n+1) / (2n+1)   ,  a=0, |x|<=1
    sinh(x) = sum x^(2n+1) / (2n+1)!         ,  a=0
    cosh(x) = sum x^(2n) / (2n)!             ,  a=0
    """)


def main():
    print("=" * 60)
    print("TAYLOR SERIES SIMULATION IN CALCULUS")
    print("=" * 60)
    
    print_formula_sheet()
    
    demo_convergence_sin()
    demo_convergence_exp()
    demo_convergence_cos()
    demo_radius_of_convergence()
    demo_error_bound()
    demo_class_usage()
    
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)
    
    visualize_approximations()
    visualize_convergence()
    visualize_partial_sums()
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    print("""
    Key Takeaways:
    1. Taylor series approximate functions with polynomials
    2. More terms = better approximation near the center
    3. Convergence radius varies by function
    4. Error decreases as n increases (for convergent series)
    """)


if __name__ == "__main__":
    main()