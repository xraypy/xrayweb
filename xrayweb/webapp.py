#!/usr/bin/env python
import os
import time
import numpy as np
import math
import xraydb

from flask import Flask, redirect, url_for, render_template, json
from flask import Response, request, session, send_from_directory

import plotly.graph_objects as go

materials_ = xraydb.materials._read_materials_db()
matlist = sorted(list(materials_.keys()))

mirror_list = ('None', 'silicon', 'quartz', 'zerodur', 'ule glass',
               'aluminum', 'chromium', 'nickel', 'rhodium', 'palladium',
               'iridium', 'platinum', 'gold')

materials_dict = json.dumps(materials_)

from random import seed, randrange
from string import printable

def random_string(n):
    """  random_string(n)
    generates a random string of length n, that will match:
       [a-z][a-z0-9](n-1)
    """
    seed(time.time())
    s = [printable[randrange(0,36)] for i in range(n-1)]
    s.insert(0, printable[randrange(10,36)])
    return ''.join(s)

def make_asciifile(header, array_names, arrays):
    buff = ['# %s' % l for l in header]
    buff.append("#---------------------")
    buff.append("# %s" % ' '.join(array_names))
    for i in range(len(arrays[0])):
        row = [a[i] for a in arrays]
        l = [nformat(x, length=12) for x in row]
        buff.append('  '.join(l))
    buff.append('')
    return '\n'.join(buff)

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
              xlog_scale=False, ylog_scale=False, y2=None,
              y1label='data', y2label='data2', yrange=None):
    """
    build a plotly-style JSON plot
    """
    data = [{'x': x.tolist(),
             'y': y.tolist(),
             'type': 'scatter',
             'name': y1label,
             'line': {'width': 3},
             'hoverinfo': 'skip'}]
    if y2 is not None:
        data.append({'x': x.tolist(),
                     'y': y2.tolist(),
                     'type': 'scatter',
                     'name': y2label,
                     'line': {'width': 3},
                     'hoverinfo': 'skip'})


    title = formula_name
    if material_name not in ('', 'None', None):
        title = material_name
    if title in ('', 'None', None):
        title = ''

    xtype = 'linear'
    if xlog_scale:
        xtype = 'log'
    ytype = 'linear'

    if ylog_scale:
        ytype = 'log'
        ymax = np.log10(y).max()
        ymin = max(np.log10(y).min(), -9)
        if y2 is not None:
            ymin = min(np.log10(y2).min(), ymin)
            ymax = max(np.log10(y2).max(), ymax)
        if yrange is None:
            yrange = (ymin-0.5, ymax+0.5)

    layout = {'title': title,
              'height': 450,
              'width': 550,
              'showlegend': len(data) > 1,
              'xaxis': {'title': {'text': 'Energy (eV)'},
                        'type': xtype,
                        'tickformat': '.0f'},
              'yaxis': {'title': {'text': ytitle},
                        'zeroline': False,
                        'type': ytype,
                        'range': yrange,
                        'tickformat': '.2g'}}

    if yrange is not None:
        layout['yaxis']['range'] = yrange

    plot_config = {'displaylogo': False,
                   'modeBarButtonsToRemove': [ 'hoverClosestCartesian',
                                               'hoverCompareCartesian',
                                               'toggleSpikelines']}

    return json.dumps({'data': data, 'layout': layout, 'config':
                       plot_config})

# takes form input and verifies that each is present and of the correct format
# returns a dictionary containing all the inputs converted into float if necessary along with a list of error messages
# if the input is valid then the output will be {'message': ['Input is valid']}
# the recipient must then extract the data they need from the dictionary
def validate_input(formula, density, step,
                   energy1='1000', energy2='50000', mode='Log',
                   material=None, angle='0.001', page='formula'):
    output = {}
    message = []
    message.append('Error(s): ')
    df = ef = ef2 = sf = af = 0
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
    elif ef == ef2:
        ef2 += sf

    isLog = True if mode == 'Log' else False

    if page == 'reflectivity':
        if angle:
            try:
                af = float(angle)/1000.0
                if af < 0:
                    message.append('Angle must be positive.')
            except:
                message.append('Angle must be a number.')
        else:
            angle = 0.001

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
    #print(message)
    return output


app = Flask('xrayweb', static_folder='static')
app.config.from_object(__name__)

@app.route('/attendata/<formula>/<rho>/<t>/<e1>/<e2>/<estep>')
def attendata(formula, rho, t, e1, e2, estep):
    en_array = np.arange(float(e1), float(e2), float(estep))
    rho = float(rho)
    mu_array = xraydb.material_mu(formula, en_array, density=rho)
    t = float(t)
    trans = np.exp(-0.1*t*mu_array)
    atten = 1 - trans

    header = (' X-ray attenuation data from xrayweb  %s ' % time.ctime(),
              ' Material.formula   : %s ' % formula,
              ' Material.density   : %.3f gr/cm^3 ' % rho,
              ' Material.thickness : %.3f mm ' % t,
              ' Column.1: energy (eV)',
              ' Column.2: atten_length (mm)' ,
              ' Column.3: trans_fraction',
              ' Column.4: atten_fraction')

    arr_names = ('energy       ', 'atten_length ',
                 'trans_fract  ', 'atten_fract  ')

    txt = make_asciifile(header, arr_names,
                         (en_array, 10/mu_array, trans, atten))

    fname = 'xrayweb_atten_%s_%s.txt' % (formula,
                                         time.strftime('%Y%h%d_%H%M%S'))
    return Response(txt, mimetype='text/plain',
                    headers={"Content-Disposition":
                             "attachment;filename=%s" % fname})

@app.route('/mirrordata/<formula>/<rho>/<angle>/<e1>/<e2>/<estep>')
def mirrordata(formula, rho, angle, e1, e2, estep):
    en_array = np.arange(float(e1), float(e2), float(estep))
    angle = float(angle)/1000.0
    rho   = float(rho)
    reflectivity = xraydb.mirror_reflectivity(formula, angle, en_array, rho)

    header = (' X-ray reflectivity data from xrayweb  %s ' % time.ctime(),
              ' Material.formula   : %s ' % formula,
              ' Material.density   : %.3f gr/cm^3 ' % rho,
              ' Material.angle     : %.3f mrad ' % angle,
              ' Column.1: energy (eV)',
              ' Column.2: reflectivity')

    arr_names = ('energy       ', 'reflectivity ')

    txt = make_asciifile(header, arr_names,
                         (en_array, reflectivity))

    fname = 'xrayweb_mirror_%s_%s.txt' % (formula,
                                          time.strftime('%Y%h%d_%H%M%S'))
    return Response(txt, mimetype='text/plain',
                    headers={"Content-Disposition":
                             "attachment;filename=%s" % fname})

@app.route('/element/', methods=['GET', 'POST'])
@app.route('/element/<elem>',  methods=['GET', 'POST'])
def element(elem=None):
    edges = atomic = lines = {}
    if elem is not None:
        edges= xraydb.xray_edges(elem)
        atomic= {'n': xraydb.atomic_number(elem), 'mass': xraydb.atomic_mass(elem),
                 'density': xraydb.atomic_density(elem)}
        lines= xraydb.xray_lines(elem)
    return render_template('elements.html', edges=edges, elem=elem, atomic=atomic,
                           lines=lines, materials_dict=materials_dict)

@app.route('/formula/', methods=['GET', 'POST'])
@app.route('/formula/<material>', methods=['GET', 'POST'])
def formula(material=None):
    message = ['']
    abslen = absq = energies = []
    mu_plot = atten_plot = output = {}
    num = errors = 0
    isLog = True
    datalink = None
    if request.method == 'POST':
        formula = request.form.get('formula')
        matname = request.form.get('matname')
        density = request.form.get('density')
        energy1 = request.form.get('energy1')
        energy2 = request.form.get('energy2')
        step = request.form.get('step')
        mode = request.form.get('mode')
        thickness = request.form.get('thickness')

        #input validation
        output = validate_input(formula, density, step, energy1, energy2, mode, material, page='formula')
        message = output['message']
    else:
        request.form = {'mats': 'silicon',
                        'formula': materials_['silicon'].formula,
                        'density': materials_['silicon'].density,
                        'energy1':  1000,
                        'energy2': 51000,
                        'step': '100',
                        'thickness': 1.00,
                        'mode': 'Linear'}

    if message[0] == 'Input is valid':
        #unpack floats
        df = output['df']
        ef = output['ef']
        ef2 = output['ef2']
        sf = output['sf']
        isLog = output['isLog']

        #make plot
        en_array = np.arange(ef, ef2, sf)
        num = en_array.size
        mu_array = xraydb.material_mu(formula, en_array, density=df)
        t = float(thickness)
        trans = np.exp(-0.1*t*mu_array)
        atten = 1 - trans

        if num > 2:

            mu_plot = make_plot(en_array, 10/mu_array, material, formula,
                                ylog_scale=isLog, ytitle='1/e length (mm)')
            atten_plot = make_plot(en_array, trans,
                                   material, "%.3f mm %s" % (t, formula),
                                   ylog_scale=isLog,
                                   y2=atten,
                                   yrange=[1.e-8, 1.05],
                                   ytitle='transmitted/attenuated fraction',
                                   y1label='transmitted',
                                   y2label='attenuated')
        message = []
    else:
        errors = len(message)

    return render_template('formulas.html', message=message, errors=errors,
                           datalink=datalink, mu_plot=mu_plot,
                           atten_plot=atten_plot, matlist=matlist,
                           materials_dict=materials_dict, input=input)

@app.route('/reflectivity/', methods=['GET', 'POST'])
def reflectivity(material=None):
    message = ['']
    ref_plot = output = {}
    energies = reflectivities = []
    num = errors = 0
    df = ef = ef2 = sf = af = 0
    isLog = True
    if request.method == 'POST':
        #obtain form input and verify
        formula1 = request.form.get('formula1', 'None')
        density1 = request.form.get('density1', '')
        angle1 = request.form.get('angle1', '0')
        material1 = request.form.get('mats1', 'silicon')

#         formula2 = request.form.get('formula2', 'None')
#         density2 = request.form.get('density2', '')
#         angle2 = request.form.get('angle2', '0')
#         material2 = request.form.get('mats2', 'None')

        energy1 = request.form.get('energy1', '1000')
        energy2 = request.form.get('energy2', '50000')
        estep  = request.form.get('step', '100')
        mode = request.form.get('mode', 'Linear')

        output1 = validate_input(formula1, density1, estep, energy1,
                                 energy2, mode, material1, angle1,
                                 page='reflectivity')

        message = output1['message']
        # if material2 is not 'None':
        #     output2 = validate_input(formula2, density2, estep, energy1, energy2, mode,
        #                              material2, angle2, page='reflectivity')
    else:
        request.form = {'mats1': 'silicon',
                        'formula1': materials_['silicon'].formula,
                        'density1': materials_['silicon'].density,
                        'angle1': 2.5,
                        'mats2': 'None',
                        'formula2': '',
                        'density2': '',
                        'angle2': 2.5,
                        'energy1': 1000,
                        'energy2': 50000,
                        'step': '100',
                        'mode': 'Linear'}

    #if verification passes, calculate output and pass to render_template
    if message[0] == 'Input is valid':
        df = output1['df']
        ef = output1['ef']
        ef2 = output1['ef2']
        sf = output1['sf']
        isLog = output1['isLog']
        af = output1['af']

        en_array = np.arange(ef, ef2, sf)
        num = en_array.size
        """
        if not num:
            np.append(en_array, [ef])
            print(num, en_array)
        """
        ref_array = xraydb.mirror_reflectivity(formula1, af, en_array, df)

        if num > 2:
            title = "%s, %s mrad" % (formula1, angle1)
            ref_plot = make_plot(en_array, ref_array, title, formula1,
                                 ytitle='Reflectivity', ylog_scale=isLog)

        energies = [nformat(x) for x in en_array]
        reflectivities = [nformat(x) for x in ref_array]
        message = []
    else:
        errors = len(message)

    return render_template('reflectivity.html', message=message,
                           errors=errors, ref_plot=ref_plot,
                           energies=energies,
                           reflectivities=reflectivities, num=num,
                           matlist=mirror_list,
                           materials_dict=materials_dict)

@app.route('/crystal/', methods=['GET', 'POST'])
def crystal():
    error = ''
    if request.method == 'POST':
        pass
    return render_template('monochromic_crystals.html', error=error)

@app.route('/')
def index():
    return redirect(url_for('element'))
