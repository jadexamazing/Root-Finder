import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# --- Page config ---
st.set_page_config(
    page_title="🐰 Root Finder 🐰",
    page_icon="🐰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2&display=swap');

    body, .stApp {
        background-color: #7A73D1;
        font-family: 'Baloo 2', cursive;
        text-align: center;
    }
    .big-title {
        font-size: 56px;
        font-weight: 700;
        text-align: center;
        color: rgb(127, 85, 177);
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: purple;
        margin-top: 0;
        margin-bottom: 20px;
        line-height: 1.4;
    }
    .stButton>button {
        background-color: #0E2148;
        color: white;
        font-size: 20px;
        font-weight: 600;
        padding: 10px 24px;
        border-radius: 25px;
        border: none;
        transition: background-color 0.3s ease;
        display: block;
        margin: 0 auto;
    }
    .stButton>button:hover {
        background-color: #9a66e9;
        cursor: pointer;
    }
    .stTable table {
        border-collapse: separate !important;
        border-spacing: 0 8px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .stTable thead tr th {
        background-color: #F49BAB !important;
        color: #fff !important;
        padding: 8px !important;
        border-radius: 8px 8px 0 0 !important;
        text-align: center !important;
    }
    .stTable tbody tr td {
        background-color: #9B7EBD !important;
        padding: 8px !important;
        text-align: center !important;
    }
    .tooltip {
        border-bottom: 1px dotted #0E2148;
        cursor: help;
    }
    .section {
        margin-top: 25px;
        margin-bottom: 25px;
        padding: 20px;
        background-color: #f5f0ff;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(123, 63, 191, 0.2);
    }
    </style>
""", unsafe_allow_html=True)



# --- Session state for intro ---

if "started" not in st.session_state:
    st.session_state.started = False

# --- Intro Screen ---

if not st.session_state.started:
    st.markdown('<h1 class="big-title">🐰 ROOT FINDER 🐰</h1>', unsafe_allow_html=True)
    st.markdown("""
        <p class="subtitle">
        Welcome to <b>Root Finder</b> — Have fun exploring numerical root-finding like never before! —<br>

        These are the methods that is used:<br>
        - 🪓 Bisection Method<br>
        - ⚖️ Regula Falsi Method<br>
        - 🚶‍♂️ Incremental Method<br>
        - 🔁 Newton-Raphson Method<br>
        - 🔀 Secant Method<br>
        - 📈 Graphical Method<br><br>
        Ready to start? Click the button below! 🐰
        </p>
    """, unsafe_allow_html=True)
    if st.button("🌟 Start Finding Roots"):
        st.session_state.started = True
    st.stop()

# --- Main App Header ---
st.markdown('<h1 class="big-title">🐰 Root Finder 🐰</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter your function and select a method to find roots beautifully.</p>', unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("🔧 Input Parameters")
func_input = st.sidebar.text_input("Enter function f(x):", value="x**3 - 4*x - 9", help="Use Python syntax, e.g. x**2 - 5*x + 6")

method = st.sidebar.selectbox("Choose method:", 

                              ["Bisection", "Incremental Method", "Regula Falsi", "Newton-Raphson", "Secant", "Graphical"])

if method in ["Bisection", "Regula Falsi"]:
    xl = st.sidebar.number_input("Interval start (xl):", value=-5.0, help="Start of interval")
    xu = st.sidebar.number_input("Interval end (xu):", value=5.0, help="End of interval")
elif method == "Incremental Method":
    xl = st.sidebar.number_input("Start (xl):", value=-5.0, help="Start of search interval")
    xu = st.sidebar.number_input("End (xu):", value=5.0, help="End of search interval")
    delta = st.sidebar.number_input("Step size (delta):", value=0.5, min_value=0.001, help="Incremental step size")
elif method == "Newton-Raphson":
    x0 = st.sidebar.number_input("Initial guess (x0):", value=2.0, help="Starting point for Newton-Raphson")
elif method == "Secant":
    x0 = st.sidebar.number_input("Initial guess x0:", value=2.0, help="First initial guess")
    x1 = st.sidebar.number_input("Second guess x1:", value=3.0, help="Second initial guess")
elif method == "Graphical":
    a = st.sidebar.number_input("Plot start (a):", value=-10.0)
    b = st.sidebar.number_input("Plot end (b):", value=10.0)
st.sidebar.markdown("---")

# --- Function to safely parse ---
def parse_function(expr):
    x = sp.symbols('x')
    try:
        f_sympy = sp.sympify(expr)
        f = sp.lambdify(x, f_sympy, 'numpy')
        # Test function at 0 to catch errors
        f(0)
        return f, f_sympy
    except Exception as e:
        st.sidebar.error(f"Error parsing function: {e}")
        return None, None

# --- Numerical Methods ---
def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    results = []
    if f(a) * f(b) >= 0:
        st.warning("⚠️ f(a) and f(b) have the same sign. Bisection may not converge.")
    iter_count = 0
    while abs(b - a) > tol and iter_count < max_iter:
        c = (a + b)/2
        fa = f(a)
        fb = f(b)
        fc = f(c)
        ea = abs(b - a)
        results.append((iter_count, a, c, b, fa, fc, ea))
        if fc == 0:
            break
        if fa * fc < 0:
            b = c
        else:
            a = c
        iter_count += 1
    return c, results
def regula_falsi_method(f, a, b, tol=1e-6, max_iter=100):
    results = []
    if f(a) * f(b) >= 0:
        st.warning("⚠️ f(a) and f(b) have the same sign. Regula Falsi may not converge.")
    iter_count = 0
    c = a
    while iter_count < max_iter:
        fa = f(a)
        fb = f(b)
        c_prev = c
        c = (a*fb - b*fa)/(fb - fa)
        fc = f(c)
        ea = abs(c - c_prev) if iter_count > 0 else None
        results.append((iter_count, a, c, b, fa, fc, ea))
        if abs(fc) < tol:
            break
        if fa * fc < 0:
            b = c
        else:
            a = c
        iter_count += 1
    return c, results
def incremental_method(f, xl, xu, delta=0.01, max_iter=1000):
    results = []
    x = xl
    iter_count = 0
    found_interval = False
    while x < xu and iter_count < max_iter:
        fx = f(x)
        fx_delta = f(x + delta)
        results.append((iter_count, x, x + delta, fx, fx_delta))
        if fx * fx_delta < 0 or fx == 0 or fx_delta == 0:
            found_interval = True
            break
        x += delta
        iter_count += 1
    if found_interval:
        return (x, x + delta), results
    else:
        st.warning("No sign change found in interval.")
        return None, results

def newton_raphson_method(f_sympy, x0, tol=1e-6, max_iter=100):
    x = sp.symbols('x')
    f = sp.lambdify(x, f_sympy, 'numpy')
    f_prime_sympy = sp.diff(f_sympy, x)
    f_prime = sp.lambdify(x, f_prime_sympy, 'numpy')
    results = []
    xi = x0
    for i in range(max_iter):
        fxi = f(xi)
        fpxi = f_prime(xi)
        if fpxi == 0:
            st.error("Zero derivative - no solution.")
            return None, results
        xi_new = xi - fxi / fpxi
        ea = abs(xi_new - xi)
        results.append((i, xi, fxi, fpxi, ea))
        if ea < tol:
            return xi_new, results
        xi = xi_new
    st.warning("Max iterations reached.")
    return xi, results

def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    results = []
    for i in range(max_iter):
        f_x0 = f(x0)
        f_x1 = f(x1)
        if (f_x1 - f_x0) == 0:
            st.error("Zero denominator in Secant method.")
            return None, results
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        ea = abs(x2 - x1)
        results.append((i, x0, x1, x2, ea))
        if ea < tol:
            return x2, results
        x0, x1 = x1, x2
    st.warning("Max iterations reached.")
    return x2, results

def plot_function(f, f_sympy, a, b, root=None):
    x_vals = np.linspace(a, b, 400)
    y_vals = f(x_vals)
    plt.figure(figsize=(8,5))
    plt.axhline(0, color='black', lw=1)
    plt.plot(x_vals, y_vals, label=f'f(x) = {f_sympy}')
    if root is not None:
        plt.plot(root, f(root), 'ro', label=f'Root ≈ {root:.6f}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    st.pyplot(plt.gcf())
    plt.close()

# --- Run Calculation ---
if st.button("Find Root"):
    f, f_sympy = parse_function(func_input)
    if f is None:
        st.stop()

    with st.spinner("Calculating..."):
        if method == "Bisection":
            root, results = bisection_method(f, xl, xu)
            if root is not None:
                st.success(f"Estimated root: {root:.6f}")
                st.markdown("#### Iterations")
                st.markdown("*Note: `xl`, `xr`, and `xu` represent interval bounds and midpoint per iteration.*")
                st.table([{
                    "Iteration": i,
                    "xl": a,
                    "xr": c,
                    "xu": b,
                    "f(xl)": fa,
                    "f(xr)": fc,
                    "Error": err,
                    "f(xl)·f(xr)": "< 0" if fa * fc < 0 else "> 0",
                    "Remark": "1st subinterval" if fa * fc < 0 else "2nd subinterval"
                } for i, a, c, b, fa, fc, err in results])
                plot_function(f, f_sympy, xl, xu, root)
            else:
                st.error("Bisection method failed to find a root.")

        elif method == "Regula Falsi":
            root, results = regula_falsi_method(f, xl, xu)
            if root is not None:
                st.success(f"Estimated root: {root:.6f}")
                st.markdown("#### Iterations")
                st.markdown("*Note: `xr` is the intersection point. Remark shows interval update direction.*")
                st.table([{
                    "Iteration": i,
                    "xl": a,
                    "xr": c,
                    "xu": b,
                    "f(xl)": fa,
                    "f(xr)": fc,
                    "Error": err,
                    "f(xl)·f(xr)": "< 0" if fa * fc < 0 else "> 0",
                    "Remark": "1st subinterval" if fa * fc < 0 else "2nd subinterval"
                } for i, a, c, b, fa, fc, err in results])
                plot_function(f, f_sympy, xl, xu, root)
            else:
                st.error("Regula Falsi method failed to find a root.")
        elif method == "Incremental Method":
            interval, results = incremental_method(f, xl, xu, delta)
            if interval is not None:
                st.success(f"Root lies in interval: [{interval[0]:.6f}, {interval[1]:.6f}]")
            st.markdown("#### Steps")
            st.table([{
                "Step": i,
                "x": x_start,
                "x+delta": x_end,
                "f(x)": fx,
                "f(x+delta)": fxdelta,
                "f(x)·f(x+δ)": "< 0" if fx * fxdelta < 0 else "> 0",
                "Remark": "1st subinterval" if fx * fxdelta < 0 else "2nd subinterval"
            } for i, x_start, x_end, fx, fxdelta in results])
            plot_function(f, f_sympy, xl, xu)
                
        elif method == "Newton-Raphson":
            root, results = newton_raphson_method(f_sympy, x0)
            if root is not None:
                st.success(f"Estimated root: {root:.6f}")
                st.markdown("#### Iterations")
                st.table([{
                    "Iteration": i,
                    "xi": xi,
                    "f(xi)": fxi,
                    "f'(xi)": fpxi,
                    "Error": err
                } for i, xi, fxi, fpxi, err in results])
                plot_function(f, f_sympy, x0 - 5, x0 + 5, root)
            else:
                st.error("Newton-Raphson method failed to find a root.")

        elif method == "Secant":
            root, results = secant_method(f, x0, x1)
            if root is not None:
                st.success(f"Estimated root: {root:.6f}")
                st.markdown("#### Iterations")
                st.table([{
                    "Iteration": i,
                    "x0": x_0,
                    "x1": x_1,
                    "x2": x_2,
                    "Error": err
                } for i, x_0, x_1, x_2, err in results])
                plot_function(f, f_sympy, min(x0, x1) - 5, max(x0, x1) + 5, root)
            else:
                st.error("Secant method failed to find a root.")

        elif method == "Graphical":
            a_val = a
            b_val = b
            plot_function(f, f_sympy, a_val, b_val)
