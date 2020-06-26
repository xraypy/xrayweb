#!/usr/bin/env python
import base64
from io import BytesIO

from flask import Flask, redirect, url_for, render_template
from flask import request, session

from matplotlib.figure import Figure

import xraydb


app = Flask('xrayweb', static_folder='static')
app.config.from_object(__name__)

@app.route('/element/', methods=['GET', 'POST'])
@app.route('/element/<elem>',  methods=['GET', 'POST'])
def element(elem=None):
    edges = atomic = lines = {}
    if elem is not None:
        edges= xraydb.xray_edges(elem)
        atomic= {'n': xraydb.atomic_number(elem), 'mass': xraydb.atomic_mass(elem), 'density': xraydb.atomic_density(elem)}
        lines= xraydb.xray_lines(elem)
    return render_template('elements.html', edges=edges, elem=elem, atomic=atomic, lines=lines)

@app.route('/formula/', methods=['GET', 'POST'])
@app.route('/formula/<fmla>', methods=['GET', 'POST'])
def formula(fmla=None):
    if fmla is not None:
        pass
        #obtain info from the database
    import matplotlib.pyplot as plt, mpld3
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3]) 
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    pstr = f"<img src='data:image/png;base64,{data}'/>" 
    #print(pstr)
    with open("plt.html", "w") as file:
        file.write(pstr)

    return render_template('formulas.html')

@app.route('/')
def index():
    return redirect(url_for('element'))