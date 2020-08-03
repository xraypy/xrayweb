#!/usr/bin/env python
import base64
import jinja2
from io import BytesIO
import numpy as np
import math
import xraydb

from flask import Flask, redirect, url_for, render_template, json
from flask import request, session

import plotly.graph_objects as go

from matplotlib.figure import Figure

# from jinja2_base64_filters import jinja2_base64_filters

env = jinja2.Environment()

materials_dict = xraydb.materials._read_materials_db()
matlist = list(materials_dict.keys())
matlist = sorted(matlist)
materials_dict = json.dumps(materials_dict)

mirror_list = ('None', 'silicon', 'quartz', 'zerodur', 'ule glass',
               'aluminum', 'chromium', 'nickel', 'rhodium', 'palladium',
               'iridium', 'platinum', 'gold')

def nformat(val, length=11):
    """Format a number with '%g'-like format.

    Except that:
        a) the length of the output string is fixed.
        b) positive numbers will have a leading blank.
        b) the precision will be as high as possible.
        c) trailing zeros will not be trimmed.

    The precision will typically be at least ``length-7``,
    with ``length-6`` significant digits shown.

    Parameters
    ----------
    val : float
        Value to be formatted.
    length : int, optional
        Length of output string (default is 11).

    Returns
    -------
    str
        String of specified length.

    Notes
    ------
    Positive values will have leading blank.

    """
    try:
        expon = int(math.log10(abs(val)))
    except (OverflowError, ValueError):
        expon = 0
    length = max(length, 7)
    form = 'e'
    prec = length - 7
    if abs(expon) > 99:
        prec -= 1
    elif ((expon >= 0 and expon < (prec+4)) or
          (expon <= 0 and -expon < (prec-1))):
        form = 'f'
        prec += 4
        if expon > 0:
            prec -= expon
    fmt = '{0: %i.%i%s}' % (length, prec, form)
    return fmt.format(val)[:length]

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
             'hoverinfo': 'x+y'}]
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
              'hovermode': 'closest',
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

#takes form input and verifies that each is present and of the correct format
#returns a dictionary containing all the inputs converted into float if necessary along with a list of error messages
#if the input is valid then the output will be {'message': ['Input is valid']}
#the recipient must then extract the data they need from the dictionary
def validate_input(formula, density, step, energy1='1000', energy2='50000', mode='Log', material=None, angle='0.001',
    roughness = '0.0', page='formula'):
    output = {}
    message = []
    message.append('Error(s): ')
    df = ef = ef2 = sf = af = rf = 0
    isLog = True

    if not formula:
        message.append('Formula is a required field.')
    elif not material:
        #verify formula using function
        if not xraydb.validate_formula(formula):
            message.append('Unable to compute formula.')

    if density:
        try:
            df = float(density)
            if df <= 0:
                message.append('Density must be a positive number.')
        except:
            message.append('Density must be a positive number.')
    else:
        message.append('Density is a required field.')

    if energy1:
        try:
            ef = float(energy1)
            if ef <= 0:
                message.append('Energy1 must be a positive number.')
        except:
            message.append('Energy1 must be a positive number.')
    else:
        ef = 1000.0

    if energy2:
        try:
            ef2 = float(energy2)
            if ef2 <= 0:
                message.append('Energy2 must be a positive number.')
        except:
            print('hi')
            message.append('Energy2 must be a positive number.')
    else:
        ef2 = 50000.0

    sf = float(step)

    if ef > ef2:
        message.append('Energy1 must be less than Energy2.')

    isLog = True if mode == 'Log' else False

    if page == 'reflectivity':
        if angle:
            try:
                af = float(angle)
                if af < 0:
                    message.append('Angle must be positive.')
            except:
                message.append('Angle must be a number.')
        else:
            angle = 0.001

        if roughness:
            try:
                rf = float(roughness)
                if rf < 0:
                    message.append('Roughness must be positive.')
            except:
                message.append('Roughness must be a number.')
        else:
            roughness = 0.0

    if len(message) == 1:
        message[0] = 'Input is valid'

    output['message'] = message
    if message[0] == 'Input is valid':
        output['df'] = df
        output['ef'] = ef
        output['ef2'] = ef2
        output['sf'] = sf
        output['isLog'] = isLog
        output['af'] = af
        output['rf'] = rf
    #print(message)
    return output


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
    return render_template('elements.html', edges=edges, elem=elem, atomic=atomic, lines=lines, materials_dict=materials_dict)

@app.route('/formula/', methods=['GET', 'POST'])
@app.route('/formula/<material>', methods=['GET', 'POST'])
def formula(material=None):
    message = ['']
    abslen = absq = energies = []
    mu_plot = output = {}
    num = errors = 0
    isLog = True

    if request.method == 'POST':
        formula = request.form.get('formula')
        density = request.form.get('density')
        energy1 = request.form.get('energy1')
        energy2 = request.form.get('energy2')
        step = request.form.get('step')
        mode = request.form.get('mode')
        data = request.form.get('data')

        #input validation
        output = validate_input(formula, density, step, energy1, energy2, mode, material, page='formula')
        message = output['message']

    if message[0] == 'Input is valid':
        #unpack floats
        df = output['df']
        ef = output['ef']
        ef2 = output['ef2']
        sf = output['sf']
        isLog = output['isLog']

        #make plot
        while ef2 < (ef + (sf * 2)): #this ensures that there are always at least 3 energy values for a meaningful plot
            ef2 += sf
        ef2 += sf #this includes energy2 in the plot, normally np.arrage excludes the upper bound
        en_array = np.arange(ef, ef2, sf)
        num = en_array.size
        mu_array = xraydb.material_mu(formula, en_array, density=df)

        if data == 'Abslen':
            len_array = np.array([])
            for i in mu_array:
                len_array = np.append(len_array, 10000 / i)
            mu_plot = make_plot(en_array, len_array, material, formula, ylog_scale=isLog)
        else:
            mu_plot = make_plot(en_array, mu_array, material, formula, ylog_scale=isLog)

        energies = [nformat(x) for x in en_array]
        absq = [nformat(x) for x in mu_array]
        abslen = [10000 / float(nformat(x)) for x in mu_array]

        message = []
    else:
        errors = len(message)

    return render_template('formulas.html', message=message, errors=errors, abslen=abslen,
                           mu_plot=mu_plot,
                           absq=absq, energies=energies, num=num,
                           matlist=matlist, materials_dict=materials_dict, input=input)

@app.route('/reflectivity/', methods=['GET', 'POST'])
def reflectivity(material=None):
    message = ['']
    ref_plot = output = {}
    energies = reflectivities = []
    num = errors = 0
    df = ef = ef2 = sf = af = rf = 0
    isLog = True

    if request.method == 'POST':
        #obtain form input and verify
        formula = request.form.get('formula')
        density = request.form.get('density')
        angle = request.form.get('angle')
        roughness = request.form.get('roughness')
        polarization = request.form.get('polarization')
        energy1 = request.form.get('energy1')
        energy2 = request.form.get('energy2')
        step = request.form.get('step')
        mode = request.form.get('mode')

        output = validate_input(formula, density, step, energy1, energy2, mode, material, angle, roughness, page='reflectivity')
        message = output['message']

    #if verification passes, calculate output and pass to render_template
    if message[0] == 'Input is valid':
        df = output['df']
        ef = output['ef']
        ef2 = output['ef2']
        sf = output['sf']
        isLog = output['isLog']
        af = output['af']
        rf = output['rf']

        en_array = np.arange(ef, ef2, sf)
        num = en_array.size
        ref_array = xraydb.mirror_reflectivity(formula, af, en_array, df, rf, polarization)


        if num > 2:
            ref_plot = make_plot(en_array, ref_array, material, formula, ytitle='Reflectivity', ylog_scale=isLog)

        energies = [nformat(x) for x in en_array]
        reflectivities = [nformat(x) for x in ref_array]

        message = []
    else:
        errors = len(message)

    return render_template('reflectivity.html', message=message, errors=errors, ref_plot=ref_plot,
                            energies=energies, reflectivities=reflectivities, num=num,
                            matlist=mirror_list, materials_dict=materials_dict)

@app.route('/crystal/', methods=['GET', 'POST'])
def crystal():
    error = ''
    if request.method == 'POST':
        pass
    return render_template('monochromic_crystals.html', error=error)

@app.route('/')
def index():
    return redirect(url_for('element'))
