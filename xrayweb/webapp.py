#!/usr/bin/env python
import base64
import jinja2
from io import BytesIO

from flask import Flask, redirect, url_for, render_template
from flask import request, session

from matplotlib.figure import Figure

from jinja2_base64_filters import jinja2_base64_filters

env = jinja2.Environment()

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
@app.route('/formula/<material>', methods=['GET', 'POST'])
def formula(material=None):
    #env.filters['b64decode'] = base64.b64decode
    formula = message = ''
    abslen = 0.0
    if request.method == 'POST':
        formula = request.form.get('formula')
        density = request.form.get('density')
        energy = request.form.get('energy')

        #TODO: formula validation
        #once energy ranges can be inputted, extra verification needed to make sure the max is higher than the min
        try:
            df = float(density)
            if df < 0:
                message = 'Density must be a positive number'
        except:
            message = 'Density must be a positive number'

        try:
            ef = float(energy)
            if ef < 0:
                message = 'Energy must be a positive number'
        except:
            message = 'Energy must be a positive number'

        if not message:
            message = 'Input is valid'
        
    if message == 'Input is valid':
        #TODO: once functionality for multiple values is implemented, add a loop here to store all the values to be plotted later
        abslen = xraydb.material_mu(formula, ef, df)

    mdata = ()
    if material is not None:
        materials_dict = xraydb.materials._read_materials_db()
        mdata = materials_dict[material]


    #for name, data in materials_dict.items(): 
        #print(name, data)
    #print('Formula: ' + formula + ' Density: ' + density + ' Energy: ' + energy)
    """
    import matplotlib.pyplot as plt, mpld3
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3]) 
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    pstr = f"<img src='data:image/png;base64,{data}'/>" 
    print(env.filters['b64decode'])
    with open("plt.html", "w") as file:
        file.write(pstr)
    """
    return render_template('formulas.html', message=message, abslen=abslen, mdata=mdata)

@app.route('/')
def index():
    return redirect(url_for('element'))