#!/usr/bin/env python
import os
import time
import json
from collections import OrderedDict
import numpy as np

from flask import (Flask, redirect, url_for, render_template,
                   request, session, Response)

import xraydb

top, _ =  os.path.split(os.path.abspath(__file__))

app = Flask('xrayweb',
            static_folder=os.path.join(top, 'static'),
            template_folder=os.path.join(top, 'templates'))
app.config.from_object(__name__)

materials_ = xraydb.materials._read_materials_db()
matlist    = sorted(list(materials_.keys()))

mirror_mat = ('silicon', 'quartz', 'zerodur', 'ule glass',
              'aluminum', 'chromium', 'nickel', 'rhodium', 'palladium',
              'iridium', 'platinum', 'gold')

materials_dict = json.dumps(materials_)

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
        expon = int(np.log10(abs(val)))
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
             'hoverinfo': 'x+y'}]
    if y2 is not None:
        data.append({'x': x.tolist(),
                     'y': y2.tolist(),
                     'type': 'scatter',
                     'name': y2label,
                     'line': {'width': 3},
                     'hoverinfo': 'x+y'})

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
              'hovermode': 'closest',
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


@app.route('/')
@app.route('/<elem>', methods=['GET', 'POST'])
def index(elem=None):
    edges = atomic = lines = {}
    if elem is not None:
        edges= xraydb.xray_edges(elem)
        atomic= {'n': xraydb.atomic_number(elem),
                 'mass': xraydb.atomic_mass(elem),
                 'density': xraydb.atomic_density(elem)}
        lines= xraydb.xray_lines(elem)
    return render_template('elements.html', edges=edges, elem=elem,
                           atomic=atomic, lines=lines, materials_dict=materials_dict)


@app.route('/element/')
@app.route('/element/<elem>')
def element(elem=None):
    edges = atomic = lines = {}
    if elem is not None:
        atomic= {'n': xraydb.atomic_number(elem),
                 'mass': xraydb.atomic_mass(elem),
                 'density': xraydb.atomic_density(elem)}
        _edges= xraydb.xray_edges(elem)
        _lines= xraydb.xray_lines(elem)
        lines = OrderedDict()
        for k in sorted(_lines.keys()):
            lines[k] = _lines[k]

        edges = OrderedDict()
        for k in sorted(_edges.keys()):
            edges[k] = _edges[k]
    return render_template('elements.html', edges=edges, elem=elem,
                           atomic=atomic, lines=lines, materials_dict=materials_dict)

@app.route('/about/')
def about():
    return render_template('about.html',
                           materials_dict=materials_dict)

@app.route('/atten/', methods=['GET', 'POST'])
@app.route('/atten/<material>', methods=['GET', 'POST'])
def atten(material=None):
    message = []
    energies = []
    mu_plot = atten_plot = {}
    num = errors = 0
    mode = 'Linear'
    datalink = None
    if request.method == 'POST':
        formula = request.form.get('formula')
        matname = request.form.get('matname')
        density = request.form.get('density')
        energy1 = request.form.get('energy1')
        energy2 = request.form.get('energy2')
        estep = request.form.get('step')
        mode = request.form.get('mode')
        thickness = request.form.get('thickness')

        #input validation
        if not xraydb.validate_formula(formula):
            message.append("cannot interpret chemical formula")

        try:
            density = max(0, float(density))
        except:
            message.append('Density must be a positive number.')

        if len(message) == 0:
            use_log = mode.lower() == 'log'
            # make plot
            en_array = np.arange(float(energy1), float(energy2)+float(estep), float(estep))

            num = en_array.size
            mu_array = xraydb.material_mu(formula, en_array, density=float(density))
            t = float(thickness)
            trans = np.exp(-0.1*t*mu_array)
            atten = 1 - trans

            mu_plot = make_plot(en_array, 10/mu_array, material, formula,
                                ylog_scale=use_log, ytitle='1/e length (mm)')
            atten_plot = make_plot(en_array, trans,
                                   material, "%.3f mm %s" % (t, formula),
                                   ylog_scale=use_log,
                                   y2=atten,
                                   ytitle='transmitted/attenuated fraction',
                                   y1label='transmitted',
                                   y2label='attenuated')



    else:
        request.form = {'mats': 'silicon',
                        'formula': materials_['silicon'].formula,
                        'density': materials_['silicon'].density,
                        'energy1':  1000,
                        'energy2': 51000,
                        'step': "50",
                        'thickness': 1.00,
                        'mode': 'Linear'}

    return render_template('attenuation.html', message=message, errors=len(message),
                           datalink=datalink, mu_plot=mu_plot,
                           atten_plot=atten_plot, matlist=matlist,
                           materials_dict=materials_dict, input=input)



@app.route('/reflectivity/', methods=['GET', 'POST'])
def reflectivity(material=None):
    message = []
    ref_plot = angc_plot = {}
    has_data = False

    if request.method == 'POST':
        formula1 = request.form.get('formula1', 'None')
        density1 = request.form.get('density1', '')
        angle1 = request.form.get('angle1', '0')
        material1 = request.form.get('mats1', 'silicon')

        energy1 = request.form.get('energy1', '1000')
        energy2 = request.form.get('energy2', '50000')
        estep  = request.form.get('step', '100')
        mode = request.form.get('mode', 'Linear')
        roughness = request.form.get('roughness')
        polarization = request.form.get('polarization')

        if not xraydb.validate_formula(formula1):
            message.append("cannot interpret chemical formula")

        try:
            density = max(0, float(density1))
        except:
            message.append('Density must be a positive number.')

        if len(message) == 0:
            has_data = True
            en_array = np.arange(float(energy1), float(energy2)+float(estep),
                                 float(estep))
            use_log = mode.lower() == 'log'

            ref_array = xraydb.mirror_reflectivity(formula1, 0.001*float(angle1),
                                                   en_array, density)
            title = "%s, %s mrad" % (formula1, angle1)
            ref_plot = make_plot(en_array, ref_array, title, formula1,
                                 ytitle='Reflectivity', ylog_scale=use_log)

            title = "%s Reflectivity, %s mrad" % (formula1, angle1)
            ref_plot = make_plot(en_array, ref_array, title, formula1,
                                 ytitle='Reflectivity', ylog_scale=use_log)

            _del, _bet, _ = xraydb.xray_delta_beta(formula1, density, en_array)
            ang_crit = 1000*np.arccos(1 - _del - 1j*_bet).real

            title = "%s, Critical Angle" % (formula1)
            angc_plot = make_plot(en_array, ang_crit, title, formula1,
                                  ytitle='Critical Angle (mrad)', ylog_scale=use_log)

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
                        'step': '50',
                        'polarization': 's',
                        'roughness': '0',
                        'mode': 'Linear'}


    return render_template('reflectivity.html', message=message,
                           errors=len(message), ref_plot=ref_plot, angc_plot=angc_plot,
                           has_data=has_data,
                           matlist=mirror_mat,
                           materials_dict=materials_dict)


@app.route('/scattering/', methods=['GET', 'POST'])
@app.route('/scattering/<elem>',  methods=['GET', 'POST'])
def scattering(elem=None):
    edges = atomic = lines = {}
    if elem is not None:
        edges= xraydb.xray_edges(elem)
        atomic= {'n': xraydb.atomic_number(elem),
                 'mass': xraydb.atomic_mass(elem),
                 'density': xraydb.atomic_density(elem)}
        lines= xraydb.xray_lines(elem)
    return render_template('scattering.html', edges=edges, elem=elem,
                           atomic=atomic, lines=lines, materials_dict=materials_dict)


@app.route('/ionchamber/', methods=['GET', 'POST'])
def ionchamber(elem=None):
    edges = atomic = lines = {}
    if elem is not None:
        edges= xraydb.xray_edges(elem)
        atomic= {'n': xraydb.atomic_number(elem),
                 'mass': xraydb.atomic_mass(elem),
                 'density': xraydb.atomic_density(elem)}
        lines= xraydb.xray_lines(elem)
    return render_template('ionchamber.html', edges=edges, elem=elem,
                           atomic=atomic, lines=lines, materials_dict=materials_dict)



@app.route('/darwinwidth/', methods=['GET', 'POST'])
@app.route('/darwinwidth/<elem>',  methods=['GET', 'POST'])
def darwinwidth(elem=None):
    edges = atomic = lines = {}
    if elem is not None:
        edges= xraydb.xray_edges(elem)
        atomic= {'n': xraydb.atomic_number(elem),
                 'mass': xraydb.atomic_mass(elem),
                 'density': xraydb.atomic_density(elem)}
        lines= xraydb.xray_lines(elem)
    return render_template('darwinwidth.html', edges=edges, elem=elem,
                           atomic=atomic, lines=lines, materials_dict=materials_dict)



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

@app.route('/attendata/<formula>/<rho>/<t>/<e1>/<e2>/<estep>')
def attendata(formula, rho, t, e1, e2, estep):
    en_array = np.arange(float(e1), float(e2)+float(estep), float(estep))
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

@app.route('/attenscript/<formula>/<rho>/<t>/<e1>/<e2>/<estep>')
def attenscript(formula, rho, t, e1, e2, estep):
    """attenuation data as python code"""
    script = """#!/usr/bin/env python
#
# X-rau attenuation calculations
# this requires Python3, numpy, matplotlib, and xraydb modules. Use:
#        pip install xraydb

import numpy as np
import matplotlib.pyplot as plt
import xraydb

# inputs from web form
formula = '{formula:s}'  # material chemical formula
density = {density:.8g}  # material density in gr/cm^3
thickness = {thick:.6f}  # material thickness, in mm
energy = np.arange({e1:.0f}, {e2:.0f}+{estep:.0f}, {estep:.0f})

mu_array = xraydb.material_mu(formula, energy, density=density)
atten_length = 10.0 / mu_array

trans = np.exp(-0.1*thickness*mu_array)
atten = 1 - trans

plt.plot(energy, atten_length, label='1/e length (mm)')
plt.xlabel('Energy (eV)')
plt.ylabel('1/e length (mm)')
plt.title('1/e length for %s' % formula)
plt.show()

plt.plot(energy, trans, label='transmitted')
plt.plot(energy, atten, label='attenuated')
plt.xlabel('Energy (eV)')
plt.ylabel('tranmitted/attenuated fraction')
plt.title('attenuation for %s' % formula)
plt.show()
""".format(formula=formula, density=float(rho), thick=float(t),
           e1=float(e1), e2=float(e2), estep=float(estep))

    fname = 'xrayweb_atten_%s_%s.py' % (formula,
                                          time.strftime('%Y%h%d_%H%M%S'))
    return Response(script, mimetype='text/plain',
                    headers={"Content-Disposition":
                             "attachment;filename=%s" % fname})


@app.route('/reflectdata/<formula>/<rho>/<angle>/<rough>/<polar>/<e1>/<e2>/<estep>')
def reflectdata(formula, rho, angle, rough, polar, e1, e2, estep):
    """mirror reflectivity data as file"""
    en_array = np.arange(float(e1), float(e2)+float(estep), float(estep))
    angle = float(angle)
    rho   = float(rho)
    rough = float(rough)
    reflectivity = xraydb.mirror_reflectivity(formula, 0.001*angle, en_array, rho,
                                              roughness=rough, polarization=polar)

    _del, _bet, _ = xraydb.xray_delta_beta(formula, rho, en_array)
    ang_crit = 1000*(np.pi/2 - np.arcsin(1 - _del - 1j*_bet)).real

    header = (' X-ray reflectivity data from xrayweb  %s ' % time.ctime(),
              ' Material.formula   : %s ' % formula,
              ' Material.density   : %.4f gr/cm^3 ' % rho,
              ' Material.angle     : %.4f mrad ' % angle,
              ' Material.roughness : %.4f Ang ' % rough,
              ' Material.polarization: %s ' % polar,
              ' Column.1: energy (eV)',
              ' Column.2: reflectivity',
              ' Column.3: crit_angle (mrad)')

    arr_names = ('energy       ', 'reflectivity ', 'crit_angle    ')

    txt = make_asciifile(header, arr_names, (en_array, reflectivity, ang_crit))

    fname = 'xrayweb_reflect_%s_%s.txt' % (formula,
                                          time.strftime('%Y%h%d_%H%M%S'))
    return Response(txt, mimetype='text/plain',
                    headers={"Content-Disposition":
                             "attachment;filename=%s" % fname})


@app.route('/reflectscript/<formula>/<rho>/<angle>/<rough>/<polar>/<e1>/<e2>/<estep>')
def reflectscript(formula, rho, angle, rough, polar, e1, e2, estep):
    """mirror reflectivity data as python code"""

    script = """#!/usr/bin/env python
#
# mirror reflectivity calculations
# this requires Python3, numpy, matplotlib, and xraydb modules. Use:
#        pip install xraydb
#
import numpy as np
import matplotlib.pyplot as plt
import xraydb

# inputs from web form
formula = '{formula:s}'  # mirror chemical formula
density = {density:.4f}  # mirror density in gr/cm^3
angle   = {angle:.4f}  # mirror angle in mrad
rough   = {rough:.4f}  # mirror roughness in Angstroms
polar   = '{polar:s}'  # mirror polarization ('s' for vert deflecting with horiz-polarized source)
energy = np.arange({e1:.0f}, {e2:.0f}+{estep:.0f}, {estep:.0f})


reflectivity = xraydb.mirror_reflectivity(formula, 0.001*angle, energy, density,
                                          roughness=rough, polarization=polar)

delta, beta, _ = xraydb.xray_delta_beta(formula, density, energy)
ang_crit = 1000*np.arccos(1 - delta - 1j*beta).real

plt.plot(energy, reflectivity, label='reflectivity')
plt.xlabel('Energy (eV)')
plt.ylabel('Reflectivity')
plt.title('X-ray Reflectivity for %s' % formula)
plt.show()

plt.plot(energy, ang_crit, label='Critical Angle')
plt.xlabel('Energy (eV)')
plt.ylabel('Critical Angle (mrad)')
plt.title('Critical Angle for %s' % formula)
plt.show()
""".format(formula=formula, density=float(rho),  angle=float(angle),
           rough=float(rough), polar=polar, e1=float(e1),
           e2=float(e2), estep=float(estep))

    fname = 'xrayweb_reflect_%s_%s.py' % (formula,
                                          time.strftime('%Y%h%d_%H%M%S'))
    return Response(script, mimetype='text/plain',
                    headers={"Content-Disposition":
                             "attachment;filename=%s" % fname})
