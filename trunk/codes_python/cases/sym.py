#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd

import sympy as sy

sy.init_printing()

def func_XY_to_X_Y(f):
    """
    Wrapper for f(X) -> f(X[0], X[1])
    """
    return lambda X: np.array(f(X[0], X[1]))

## Introduction
x   =   sy.Symbol('x')
#print x.is_real # None
y   =   sy.Symbol('y', real=True)    ## positive=True imaginary = True (�  la place de real)   
z   =   sy.Symbol("z", imaginary=True)
n   =   sy.Symbol('n', integer=True) ## odd= True ou even=True etc

## sy.Type
n1  =   sy.Integer(19)
# n1 + 5 = 24 mais type(n1) est sympy.core.numbers.Integer
#  i.is_Integer, i.is_real, i.is_odd = (True, True, True)

#Plus rqpide :
n2  =   sy.sympify(19)  ##n2 = 19 
f1  =   sy.sympify(2.3) ## f = 2.30000000000000

# Pour controler la précision :
f1 = sy.Float("2.3", 3)

# Fraction
frac = sy.Rational(11,13)
r1 = sy.Rational(2, 3)
r2 = sy.Rational(4, 5)
print r1*r2 # 8/15
print r1/r2 # 5/6

## Functions 
x, y, z =   sy.symbols("x, y, z")
#>>> symbols('x:10')
#    (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
#>>> symbols('x(:c)')
#    (xa, xb, xc)
#>>> symbols('f,g,h', cls=Function)
#    (f, g, h)

f   =   sy.Function('f')(x)
g   =   sy.Function("g")(x,y,z)
g.free_symbols #returns a set of unique symbols contained in a given expression
#l = sy.Function("l")(x)
#m = sy.function("m")(l)

h = sy.Lambda(x, x**2)
# h:            x   ↦   x²
# h(5) = 25
# h(1+x):       x   ↦   (x+1)²
# h(sy.sin(x))  x   ↦   sin²(x)

## Expressions
expr = 1 + 2*x + 2*x**2 
# expr = 2x² + 2x + 1
print expr.args # (1, 2*x, 2*x**2)
print expr.args[1].args[1]  # x
print expr.args[-1].args[1] # x**2 

#### Voir Numerical Python Robert Johansson books

polr    =   sy.apart(1/(x**2 + 3*x + 2), x) # Sort la décomposition en factions rationnelles 
den_com =   sy.together(1 / (y * x + y) + 1 / (1+x))
simplif =   sy.cancel(y / (y * x + y))

x, y, z =   sy.symbols("x, y, z", real=True)
expr    =   x * y + z**2 *x
values  =   {x: 1.25, y: 0.4, z: 3.2}
print "{:.2f}".format(expr.subs(values))

## Numerical evaluations
# Use of sy.N 
#sy.N(np.pi + 2)
print sy.N(np.pi + h(6)) #(36 + np.pi)

expr = sy.sin(np.pi * x * sy.exp(x)) #  N'est pas appelable
expr_func = sy.lambdify(x, expr)     #  On l'a rendu appelable
# expr_func(10)   0.87939399793

expr_func = sy.lambdify(x, expr)
xvalues = xrange(5)
expr_func(np.asarray(xvalues)) ## Ne marche pas avec les listes
#array([ 0.        ,  0.77394269,  0.64198244,  0.72163867,  0.94361635])

### Calculus
## Intro
f       =   sy.Function('f')(x)
der_f   =   sy.diff(f,x) # f.diff(x)

# D�rivées successives 
sy.diff(f, x, 2) # sy.diff(f, x, n) dérivées n-ième

# Déviées partielles
g       =   sy.Function('g')(x, y)
der_g   =   g.diff(x,y)
der_g_suc = g.diff(x,3, y,4)

## Utilisation
expr = x**4 + x**3 + x**2 + x + 1
# expr.diff(x)  ↦  4x**3 + 3*x**2 + 2*x + 1

### Matrix
# Jacobian
# To specify a Jacobian for optimize.fsolve to use we need to define a function that evaluates the Jacobian for a given input vector
x, y    =   sy.symbols("x, y")
f_mat   =   sy.Matrix([y - x**3 -2*x**2 + 1, y + x**2 - 1])
#f_mat.jacobian(sympy.Matrix([x, y]))

## Autre 
d   =   sy.Derivative(sy.exp(sy.cos(x)), x)
ddd =   d.doit()

### Faire une figure avec un zoom
figure = True
if figure == True :
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    fig = plt.figure(figsize=(8, 4))
    def f(x):
        return 1/(1 + x**2) + 0.1/(1 + ((3 - x)/0.1)**2)


    def plot_and_format_axes(ax, x, f, fontsize):
        ax.plot(x, f(x), linewidth=2)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.set_xlabel(r"$x$", fontsize=fontsize)
        ax.set_ylabel(r"$f(x)$", fontsize=fontsize)

    # main graph
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.8], axisbg="#f5f5f5")
    x = np.linspace(-4, 14, 1000)
    plot_and_format_axes(ax, x, f, 18)

    # inset
    x0, x1 = 2.5, 3.5
    ## Les deux lignes pointillées
    ax.axvline(x0, ymax=0.3, color="grey", linestyle=":")  
    ax.axvline(x1, ymax=0.3, color="grey", linestyle=":")

    #Ajout de la figure
    ax = fig.add_axes([0.5, 0.5, 0.38, 0.42], axisbg='none') 
    #[début_horiz, début_vertical, fin_horiz, fin_vertical]
    x = np.linspace(x0, x1, 1000)
    plot_and_format_axes(ax, x, f, 14)

    mpl.rcParams["text.usetex"]=True

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.05, 0.25)

    ax.axhline(0)
    # text label
    ax.text(0, 0.1, "Et donc voila ", fontsize=14, family="serif")
    # annotation
    ax.plot(1, 0, "o")
    ax.annotate("Interesting",
                fontsize=14, family="serif",
                xy=(1, 0), xycoords="data",
                xytext=(+20, +50), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5")) #guette la fleche
    ax.text(2, 0.1, r"Equation: $i\hbar\partial_t \Psi = \hat{H}\Psi$",
                    fontsize=14, family="serif")

    ### Plot 3D
    mpl.rcParams["text.usetex"]=False

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), subplot_kw={'projection': '3d'})
    def title_and_labels(ax, title):
        ax.set_title(title)
        ax.set_xlabel("$x$", fontsize=16)
        ax.set_ylabel("$y$", fontsize=16)
        ax.set_zlabel("$z$", fontsize=16)

    x = y = np.linspace(-3, 3, 74)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(4 * R) / R
    norm = mpl.colors.Normalize(-abs(Z).max(), abs(Z).max())

    p = axes[0].plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0,
                            antialiased=False, norm=norm, cmap=mpl.cm.Blues)
    cb = fig.colorbar(p, ax=axes[0], shrink=0.6)
    title_and_labels(axes[0], "plot_surface")
    axes[1].plot_wireframe(X, Y, Z, rstride=2, cstride=2, color="darkgrey")
    title_and_labels(axes[1], "plot_wireframe")
    axes[2].contour(X, Y, Z, zdir='z', offset=0, norm=norm, cmap=mpl.cm.Blues)
    axes[2].contour(X, Y, Z, zdir='y', offset=3, norm=norm, cmap=mpl.cm.Blues)
    title_and_labels(axes[2], "contour")


    #a,b,c = sy.symbols('a,b,c', char=True)
