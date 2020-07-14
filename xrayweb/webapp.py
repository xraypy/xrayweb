#!/usr/bin/env python
import base64
import jinja2
from io import BytesIO
import numpy as np

from flask import Flask, redirect, url_for, render_template, json
from flask import request, session

import plotly.graph_objects as go

from matplotlib.figure import Figure

# from jinja2_base64_filters import jinja2_base64_filters

env = jinja2.Environment()

import xraydb


def make_plot(x, y, material_name, formula_name, ytitle='mu',
                xlog_scale=False, ylog_scale=False):
    """
    build a plotly-style JSON plot
    """
    data = [{'x': x.tolist(),
             'y': y.tolist(),
             'type': 'scatter',
             'name': 'data',
             'line': {'width': 3},
             'hoverinfo': 'skip'}]
    title = formula_name
    if material_name not in ('', 'None', None):
        title = material_name
    if title in ('', 'None', None):
        title = ''

    xtype = 'linear'
    if xlog_scale:    xtype = 'log'
    ytype = 'linear'
    if ylog_scale:   ytype = 'log'
    layout = {'title': title,
              'height': 500,
              'width': 700,
              'showlegend': len(data) > 1,
              'xaxis': {'title': {'text': 'Energy (eV)'},
                        'type': xtype,
                        'tickformat': '.0f'},
              'yaxis': {'title': {'text': ytitle},
                        'zeroline': False,
                        'type': ytype,
                        'tickformat': '.2g'}    }
    plot_config = {'displaylogo': False,
                   'modeBarButtonsToRemove': [ 'hoverClosestCartesian',
                                               'hoverCompareCartesian',
                                               'toggleSpikelines']}

    return json.dumps({'data': data, 'layout': layout, 'config':
                       plot_config})


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
    absq = 0.0
    energies = 0.0
    num = 0.0
    energy1 = energy2 = step = None
    if request.method == 'POST':
        formula = request.form.get('formula')
        density = request.form.get('density')
        energy1 = request.form.get('energy1')
        energy2 = request.form.get('energy2')
        step = request.form.get('step')

        #TODO: formula validation
        #once energy ranges can be inputted, extra verification needed to make sure the max is higher than the min
        if not formula:
            message = 'Formula is a required field'
        elif not material:
            #verify using function
            pass


        try:
            df = float(density)
            if df <= 0:
                message = 'Density must be a positive number'
        except:
            message = 'Density must be a positive number'

        if energy1:
            try:
                ef = float(energy1)
                if ef <= 0:
                    message = 'Energy must be a positive number'
            except:
                message = 'Energy must be a positive number'
        else:
            message = 'Energy1 is a required field'

        if energy2:
            try:
                ef2 = float(energy2)
                if ef2 <= 0:
                    message = 'Energy must be a positive number'
            except:
                message = 'Energy must be a positive number'
            if ef and (ef > ef2): #possibly edit later based on feedback
                message = 'Energy must range from low to high values'

        if step:
            try:
                sf = float(step)
                if sf <= 0:
                    message = 'Step must be a positive number'
            except:
                message = 'Step must be a positive number'
            if not energy2:
                message = 'Step requires multiple energy values'

        if not message:
            message = 'Input is valid'

    if message == 'Input is valid':
        #TODO: once functionality for multiple values is implemented, add a loop here to store all the values to be plotted later
        #print(formula, ef, df)
        energies = []
        absq = []
        abslen = []
        if not energy2:
            energies.append(ef)
            val = xraydb.material_mu(formula, ef, df)
            absq.append(val)
            abslen.append(1 / val)
        elif not step:
            energies.append(ef)
            val = xraydb.material_mu(formula, ef, df)
            absq.append(val)
            abslen.append(1 / val)

            energies.append(ef2)
            val = xraydb.material_mu(formula, ef2, df)
            absq.append(val)
            abslen.append(1 / val)
        else:
            i = ef
            while i < ef2:
                energies.append(i)
                val = xraydb.material_mu(formula, i, df)
                absq.append(val)
                abslen.append(1 / val)
                i += sf
            energies.append(ef2)
            val = xraydb.material_mu(formula, ef2, df)
            absq.append(val)
            abslen.append(1 / val)
        num = len(energies) #this represents the number of energies and also corresponds to the number of absorption quantities/lengths
        message = ''


    materials_dict = xraydb.materials._read_materials_db()
    #print(materials_dict)
    
    if material is not None:
        mdata = materials_dict[material]

    matlist = list(materials_dict.keys())
    matlist = sorted(matlist)
    materials_dict = json.dumps(materials_dict)



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

    mu_plot = {}
    if energy1 is not None:
        en_array = np.arange(float(energy1), float(energy2), float(step))
        mu_array = 0*en_array
        if formula not in ('', 'None', None):
            mu_array = xraydb.material_mu(formula, en_array, density=float(density))

        mu_plot = make_plot(en_array, mu_array, material, formula, ylog_scale=True)


    return render_template('formulas.html', message=message, abslen=abslen,
                           mu_plot=mu_plot,
                           absq=absq, energies=energies, num=num,
                           mdata=mdata, matlist=matlist, materials_dict=materials_dict)

@app.route('/')
def index():
    return redirect(url_for('element'))
