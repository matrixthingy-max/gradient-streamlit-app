import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Gradient & Steepest Ascent", layout="wide")

st.title("Gradient and Direction of Steepest Ascent")
st.write("Interactive visualization for functions of two variables")

# Sidebar
st.sidebar.header("Choose Function")
func_type = st.sidebar.selectbox(
    "Function",
    ("f(x,y) = x² + y² (Paraboloid)", "f(x,y) = x² − y² (Saddle)")
)

x0 = st.sidebar.slider("x value", -5.0, 5.0, 1.0)
y0 = st.sidebar.slider("y value", -5.0, 5.0, 1.0)

# Functions
def f_simple(x, y):
    return x**2 + y**2

def grad_simple(x, y):
    return np.array([2*x, 2*y])

def f_complex(x, y):
    return x**2 - y**2

def grad_complex(x, y):
    return np.array([2*x, -2*y])

# Mesh
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)

if "Paraboloid" in func_type:
    Z = f_simple(X, Y)
    grad = grad_simple(x0, y0)
    z0 = f_simple(x0, y0)
else:
    Z = f_complex(X, Y)
    grad = grad_complex(x0, y0)
    z0 = f_complex(x0, y0)

grad_unit = grad / np.linalg.norm(grad) if np.linalg.norm(grad) != 0 else grad

# Plot
fig = go.Figure()

fig.add_surface(x=X, y=Y, z=Z, opacity=0.85)

fig.add_trace(go.Scatter3d(
    x=[x0], y=[y0], z=[z0],
    mode="markers",
    marker=dict(size=6, color="red"),
    name="Point (x,y)"
))

fig.add_trace(go.Scatter3d(
    x=[x0, x0 + grad_unit[0]],
    y=[y0, y0 + grad_unit[1]],
    z=[z0, z0 + np.linalg.norm(grad_unit)],
    mode="lines",
    line=dict(width=6, color="black"),
    name="Gradient Direction"
))

fig.update_layout(
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="f(x,y)"
    ),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Explanation
st.markdown("""
### 📘 Explanation
- The **gradient vector** points in the direction of **steepest ascent**
- The arrow shows ∇f(x,y)
- Moving the point changes the magnitude and direction of the gradient
""")
