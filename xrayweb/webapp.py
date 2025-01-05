#!/usr/bin/env python
"""
web app for xraydb
"""
import os
import time
import json
from collections import OrderedDict, namedtuple
import numpy as np
import scipy.constants as consts

from flask import (Flask, render_template, request, Response,
                   send_from_directory)

import xraydb
from xraydb.xraydb import XrayLine
from xraydb import chemparse
from pyshortcuts import gformat

XrayEdge = namedtuple('XrayEdge', ('energy', 'fyield', 'jump_ratio', 'width'))
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

mirror2_mat = ('None', 'silicon', 'quartz', 'zerodur', 'ule glass',
              'aluminum', 'chromium', 'nickel', 'rhodium', 'palladium',
              'iridium', 'platinum', 'gold')


materials_dict = json.dumps(materials_)

lattice_constants = {'Si': 5.4309, 'Ge': 5.6578, 'C': 3.567}
PLANCK_HC = 1.e10 * consts.Planck * consts.c / consts.e

# list of allowed reflections for diamond structure: duplicates removed
hkl_list = ('1 1 1', '2 2 0', '3 1 1', '3 3 1', '3 3 3', '4 0 0', '4 2 2',
            '4 4 0', '4 4 4', '5 1 1', '5 3 1', '5 3 3', '5 5 1', '5 5 3',
            '5 5 5', '6 2 0', '6 4 2', '6 6 0', '6 6 4', '7 1 1', '7 3 1',
            '7 3 3', '7 5 1', '7 5 3', '7 5 5', '7 7 1', '7 7 3', '7 7 5',
            '7 7 7', '8 0 0', '8 2 2', '8 4 0', '8 4 4', '8 6 2', '8 6 6',
            '8 8 0', '8 8 4', '8 8 8', '9 1 1', '9 3 1', '9 5 1', '9 5 3',
            '9 5 5', '9 7 1', '9 7 3', '9 7 5', '9 7 7', '9 9 1', '9 9 3',
            '9 9 5', '9 9 7', '9 9 9', '10 2 0', '10 4 2', '10 6 0',
            '10 6 4', '10 8 2', '10 10 0', '10 10 4', '10 10 8',
            '11 1 1', '11 3 1', '11 3 3', '11 5 1', '11 5 3', '11 5 5',
            '11 7 1', '11 7 3', '11 7 5', '11 7 7',
            '11 9 1', '11 9 3', '11 9 5', '11 9 7', '11 9 9',
            '11 11 1', '11 11 3', '11 11 5', '11 11 7', '11 11 9', '11 11 11')

emission_energies = {}
analyzer_lines = ('Ka1', 'Ka2', 'Ka3', 'Kb1', 'Kb3', 'Kb5', 'La1', 'La2', 'Lb1', 'Lb3')
for z in range(1, 96):
    atsym = xraydb.atomic_symbol(z)
    xlines = xraydb.xray_lines(z)
    for line in analyzer_lines:
        key = f'{atsym}_{line}'
        en = xlines.get(line, None)
        if en is None:
            emission_energies[key] = '0'
        else:
            emission_energies[key] = f'{en.energy:.0f}'

emission_energies_json = json.dumps(emission_energies)

PY_TOP = """#!/usr/bin/env python
# this script requires Python3, numpy, scipy, matplotlib, and xraydb modules. Use:
#        pip install xraydb matplotlib
import numpy as np
import matplotlib.pyplot as plt
import xraydb
"""

EN_MIN = 50
EN_MAX = 725000
def energy_array(e1, e2, de):
    e1x = min(EN_MAX, max(EN_MIN, float(e1)))
    e2x = min(EN_MAX, max(EN_MIN, float(e2)))
    if e2x < e1x:
        e2x = e2x + e1x
    dex = max(1, float(de))
    return np.arange(e1x, e2x+dex, dex)

def th_diffracted(energy, hkl, a):
    hkl = np.array(hkl)
    omega = np.sqrt((hkl*hkl).sum()) * PLANCK_HC / (2 * a * energy)
    if abs(omega) > 1:
        omega = -0
    return (180/np.pi)*np.arcsin(omega)

def tick_format(x):
    try:
        xmin, xmax = x.min(), x.max()
    except:
        return '.6g'
    if abs(xmax-xmin) > 1000:
        xformat = '.0f'
    elif abs(xmax-xmin) > 100:
        xformat = '.1f'
    elif abs(xmax-xmin) > 10:
        xformat = '.2f'
    elif abs(xmax-xmin) > 1:
        xformat = '.3f'
    elif abs(xmax-xmin) > 0.1:
        xformat = '.4f'
    elif abs(xmax-xmin) > 0.01:
        xformat = '.5f'
    else:
        xformat = '.6g'
    return xformat


def make_plot(x, y, material_name, formula_name, ytitle='mu',
              xtitle='Energy (eV)', y1label='data',
              xlog_scale=False, ylog_scale=False, yrange=None,
              xformat=None, yformat=None,
              y2=None, y2label='data2',
              y3=None, y3label='data3',
              y4=None, y4label='data4',
              y5=None, y5label='data5',
              y6=None, y6label='data6'):
    """
    build a plotly-style JSON plot
    """
    data = [{'x': x.tolist(),
             'y': y.tolist(),
             'type': 'scatter',
             'name': y1label,
             'line': {'width': 3},
             'hoverinfo': 'x+y'}]

    for yn, ynlab in ((y2, y2label), (y3, y3label), (y4, y4label),
                      (y5, y5label), (y6, y6label)):
        if yn is not None:
            data.append({'x': x.tolist(),
                         'y': yn.tolist(),
                         'type': 'scatter',
                         'name': ynlab,
                         'line': {'width': 3},
                         'hoverinfo': 'x+y'})

    title = formula_name
    if material_name not in ('', 'None', None):
        title = material_name
    if title in ('', 'None', None):
        title = ''


    if xformat is None:
        xformat = tick_format(x)
    if yformat is None:
        yformat = tick_format(y)

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
              'xaxis': {'title': {'text': xtitle},
                        'type': xtype,
                        'tickformat': xformat},
              'yaxis': {'title': {'text': ytitle},
                        'zeroline': False,
                        'type': ytype,
                        'range': yrange,
                        'tickformat': yformat}}

    if yrange is not None:
        layout['yaxis']['range'] = yrange

    plot_config = {'displaylogo': False,
                   'modeBarButtonsToRemove': [ 'hoverClosestCartesian',
                                               'hoverCompareCartesian',
                                               'toggleSpikelines']}

    return json.dumps({'data': data, 'layout': layout, 'config':
                       plot_config})

def make_asciifile(header, array_names, arrays):
    buff = ['#XDI/1.1']
    buff.extend([f"# {l.strip()}" for l in header])
    buff.append("#---------------------")
    buff.append("# {' '.join(array_names)}")
    for i in range(len(arrays[0])):
        row = [a[i] for a in arrays]
        l = [gformat(x, length=12) for x in row]
        buff.append('  '.join(l))
    buff.append('')
    return '\n'.join(buff)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.static_folder, 'ixas_logo.ico',
                               mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    return render_template('elements.html', elem=None,
                           materials_dict=materials_dict)


@app.route('/element/')
@app.route('/element/<elem>')
def element(elem=None):
    edges = atomic = lines = widths = {}
    if elem is not None:
        atomic= {'n': xraydb.atomic_number(elem),
                 'mass': xraydb.atomic_mass(elem),
                 'density': xraydb.atomic_density(elem)}
        _edges= xraydb.xray_edges(elem)
        _lines= xraydb.xray_lines(elem)
        widths= dict(xraydb.core_width(elem))
        lines = OrderedDict()
        for k in sorted(_lines.keys()):
            en, inten, init, final = _lines[k] # XrayLine
            lines[k] = XrayLine(energy=f'{en:.1f}', intensity=f'{inten:.5f}',
                                initial_level=init, final_level=final)
        edges = OrderedDict()
        for k in sorted(_edges.keys()):
            en, fy, jump = _edges[k] # XrayEdge
            wid = widths[k]
            edges[k] = XrayEdge(energy=f'{en:.1f}', fyield=f'{fy:.5f}',
                                jump_ratio=f'{jump:.3f}', width=f'{wid:.3f}')

    return render_template('elements.html', edges=edges, elem=elem,
                           atomic=atomic, lines=lines, materials_dict=materials_dict)

@app.route('/about/')
def about():
    return render_template('about.html',
                           materials_dict=materials_dict)

@app.route('/atten/', methods=['GET', 'POST'])
@app.route('/atten/<material>/<density>', methods=['GET', 'POST'])
@app.route('/atten/<material>/<density>/<t>/<e1>/<e2>/<de>', methods=['GET', 'POST'])
def atten(material=None, density=None, t='1.0', e1='1000', e2='51000', de='50'):
    message = []
    mu_plot = atten_plot = {}
    num = errors = 0
    mode = 'Linear'
    datalink = None
    do_plot = True
    if request.method == 'POST':
        formula = request.form.get('formula')
        matname = request.form.get('matname')
        mode = request.form.get('mode')
        density = float(request.form.get('density'))
        e1 = float(request.form.get('e1'))
        e2 = float(request.form.get('e2'))
        de = float(request.form.get('de'))
        t = float(request.form.get('thickness'))
        #input validation
        if not xraydb.validate_formula(formula):
            message.append(
      f"cannot interpret chemical formula '{formula}' : check case and validity")

        try:
            density = max(0, float(density))
        except:
            message.append('Density must be a positive number.')

    else:
        e1 = float(e1)
        e2 = float(e2)
        de = float(de)
        t  = float(t)
        if material in materials_:
            mats = material
            formula = materials_['material'].formula
            density = materials_['material'].density

        elif material is not None and density is not None:
            formula = material
            density = float(density)
            mats = None
        else:
            do_plot = False
            mats = 'silicon'
            formula = materials_['silicon'].formula
            density = materials_['silicon'].density

        request.form = {'mats': mats, 'formula': formula,
                        'density': density,
                        'e1': e1, 'e2': e2, 'de': de,
                        'thickness': t, 'mode': 'Linear'}

    if do_plot and formula is not None:
        # make plot
        en_array = energy_array(e1, e2, de)
        num = en_array.size
        try:
            chemparse(formula)
            parsed = True
        except:
            parsed = False
        if parsed:
            mu_array = xraydb.material_mu(formula, en_array, density=float(density))

            trans = np.exp(-0.1*t*mu_array)
            atten = 1 - trans
            use_log = mode.lower() == 'log'
            mu_plot = make_plot(en_array, 10/mu_array, material, formula,
                                ylog_scale=use_log, ytitle='1/e length (mm)')
            atten_plot = make_plot(en_array, trans,
                                   material, f"{t:.3f} mm {formula}",
                                   ylog_scale=use_log,
                                   y2=atten,
                                   ytitle='transmitted/attenuated fraction',
                                   y1label='transmitted',
                                   y2label='attenuated')

    return render_template('attenuation.html', message=message, errors=len(message),
                           datalink=datalink, mu_plot=mu_plot, de=int(de),
                           atten_plot=atten_plot, matlist=matlist,
                           materials_dict=materials_dict) # , input=input)



@app.route('/reflectivity/', methods=['GET', 'POST'])
@app.route('/reflectivity/<formula>/<density>/<angle>/<rough>/<polar>/<e1>/<e2>/<de>/<mats>/<plotmode>', methods=['GET', 'POST'])
@app.route('/reflectivity/<formula>/<density>/<angle>/<rough>/<polar>/<mats>/<formula2>/<density2>/<angle2>/<rough2>/<polar2>/<mats2>/<e1>/<e2>/<de>/<plotmode>', methods=['GET', 'POST'])
def reflectivity(formula='Rh', density='12.41', angle='2', rough='10',
                 polar='s', mats='rhodium',  formula2='', density2='0', angle2='2',
                 rough2='10', polar2='s', mats2='None',
                 e1='1000', e2='51000', de='50', plotmode='linear'):
    message = []
    ref_plot = angc_plot = {}
    has_data = False
    if request.method == 'POST':
        formula = request.form.get('formula', 'None')
        density = request.form.get('density', '12.41')
        angle = request.form.get('angle', '2')
        mats = request.form.get('mats', 'rhodium')
        rough = request.form.get('rough', '10')
        polar = request.form.get('polar', 's')

        formula2 = request.form.get('formula2', 'None')
        density2 = request.form.get('density2', '0')
        angle2 = request.form.get('angle2', '2')
        mats2 = request.form.get('mats2', '')
        rough2 = request.form.get('rough2', '10')
        polar2 = request.form.get('polar2', 's')

        e1 = request.form.get('e1', '1000')
        e2 = request.form.get('e2', '51000')
        de  = request.form.get('de', '50')

        plotmode = request.form.get('plotmode', 'linear')
        do_calc = True
    else:
        do_calc = formula != ''
        if not do_calc:
            formula = materials_['rhodiumn'].formula
            density = materials_['rhodium'].density

        request.form = {'mats': mats, 'formula': formula,
                        'density': density, 'angle': angle,
                        'polar': polar,  'rough': rough,
                        'mats2': mats2, 'formula2': formula2,
                        'density2': density2, 'angle2': angle2,
                        'polar2': polar2,  'rough2': rough2,
                        'e1': e1, 'e2': e2, 'de': de,
                        'plotmode': plotmode}

    if do_calc:
        if not xraydb.validate_formula(formula):
            message.append("cannot interpret chemical formula")

        try:
            density = max(0.01, float(density))
        except:
            message.append('Density must be a positive number.')

        if len(message) == 0:
            has_data = True
            en_array = energy_array(e1, e2, de)
            use_log = plotmode.lower() == 'log'

            ref_array = xraydb.mirror_reflectivity(formula, 0.001*float(angle),
                                                   en_array, density,
                                                   roughness=float(rough),
                                                   polarization=polar)
            ref_title = f"Reflectivity: {formula}, {angle} mrad"

            _del, _bet, _ = xraydb.xray_delta_beta(formula, density, en_array)
            ang_crit = 1000*np.arccos(1 - _del - 1j*_bet).real

            ang_title = f"Critical Angle: {formula}"

            has_2mirrors = False
            if len(formula2) > 0:
                try:
                    density2 = max(0.01, float(density2))
                except:
                    density2 = 0.01
                if density2 > 0.01:
                    has_2mirrors = True

            if has_2mirrors:
                ref_title = f"Reflectivity: {formula}, {angle} mrad \n {formula2}, {angle2} mrad"
                ang_title = f"Critical Angle: {formula} and {formula2}"

                ref2_array = xraydb.mirror_reflectivity(formula2, 0.001*float(angle2),
                                                   en_array, density2,
                                                   roughness=float(rough2),
                                                   polarization=polar2)

                _del, _bet, _ = xraydb.xray_delta_beta(formula2, density2, en_array)
                ang_crit2 = 1000*np.arccos(1 - _del - 1j*_bet).real


                ref_plot = make_plot(en_array, ref_array, ref_title, formula,
                                     yformat='.3f', y1label=formula,
                                     y2=ref2_array, y2label=formula2,
                                     ytitle='Reflectivity', ylog_scale=use_log)
                angc_plot = make_plot(en_array, ang_crit, ang_title, formula,
                                      y1label=formula, y2=ang_crit2, y2label=formula2,
                                      ytitle='Critical Angle (mrad)', ylog_scale=use_log)


            else:
                ref_plot = make_plot(en_array, ref_array, ref_title, formula,
                                     yformat='.3f',
                                     ytitle='Reflectivity', ylog_scale=use_log)
                angc_plot = make_plot(en_array, ang_crit, ang_title, formula,
                                      ytitle='Critical Angle (mrad)', ylog_scale=use_log)

    return render_template('reflectivity.html', message=message,
                           errors=len(message), ref_plot=ref_plot, angc_plot=angc_plot,
                           has_data=has_data,
                           matlist2=mirror2_mat,
                           matlist=mirror_mat, de=int(de),
                           materials_dict=materials_dict)


@app.route('/scattering/', methods=['GET', 'POST'])
@app.route('/scattering/<elem>',  methods=['GET', 'POST'])
@app.route('/scattering/<elem>/<e1>/<e2>/<de>',  methods=['GET', 'POST'])
def scattering(elem=None, e1='1000', e2='50000', de='50'):
    f1f2_plot = mu_plot = {}
    if len(request.form) != 0:
        elem = request.form.get('elem', 'None')
        e1 = request.form.get('e1', e1)
        e2 = request.form.get('e2', e2)
        de = request.form.get('de', de)

    if elem not in (None, 'None', ''):
        atz = xraydb.atomic_number(elem)
        en_array = energy_array(e1, e2, de)
        mu_total = xraydb.mu_elam(elem, en_array, kind='total')
        mu_photo = xraydb.mu_elam(elem, en_array, kind='photo')
        mu_incoh = xraydb.mu_elam(elem, en_array, kind='incoh')
        mu_coher = xraydb.mu_elam(elem, en_array, kind='coh')
        yrange = [-0.25+min(-1.8,
                            np.log10(mu_photo.min()+1.e-5),
                            np.log10(mu_incoh.min()+1.e-5),
                            np.log10(mu_coher.min()+1.e-5)),
                  0.75+np.log10(mu_total.max()+1.e-5)]

        mu_plot = make_plot(en_array, mu_total, f'Mass Attenuation for {elem}',
                            elem, ytitle='mu/rho (cm^2/gr)',
                            xtitle='Energy (eV)', xlog_scale=False,
                            ylog_scale=True, yrange=yrange,
                            yformat='.2f',
                            y1label='Total',
                            y2=mu_photo, y2label='Photo-electric',
                            y3=mu_incoh, y3label='Incoherent',
                            y4=mu_coher, y4label='Coherent')
        if atz < 93:
            try:
                f1 = xraydb.f1_chantler(elem, en_array)
            except:
                f1 = xraydb.f1_chantler(elem, en_array, smoothing=1)

            f2 = xraydb.f2_chantler(elem, en_array)
            f1f2_plot = make_plot(en_array, f1,
                                  f'Resonant Scattering factors for {elem}',
                                  elem, ytitle='f1, f2 (electrons/atom)',
                                  xtitle='Energy (eV)',
                                  xlog_scale=False, ylog_scale=False, y2=f2,
                                  y1label='f1', y2label='f2')
    return render_template('scattering.html', elem=elem, e1=e1, e2=e2, de=int(de),
                           f1f2_plot=f1f2_plot, mu_plot=mu_plot,
                           materials_dict=materials_dict)


@app.route('/ionchamber/', methods=['GET', 'POST'])
def ionchamber(elem=None):
    message = []

    incident_flux = transmitted_flux = photo_flux = compton_flux = rayleigh_flux = ''
    transmitted_percent = photo_percent =  compton_percent = rayleigh_percent = ''

    mat1list = ('He', 'N2', 'Ne', 'Ar', 'Kr', 'Xe', 'C (diode)', 'Si (diode)', 'Ge (diode)')
    mat2list = ('None', 'He', 'N2', 'Ne', 'Ar', 'Kr', 'Xe')

    if request.method == 'POST':
        mat1 = request.form.get('mat1', 'None')
        mat2 = request.form.get('mat2', 'None')
        frac1 = request.form.get('frac1', '1')
        thick = request.form.get('thick', '10')
        pressure = request.form.get('pressure', '1')
        energy = request.form.get('energy', '10000')
        voltage = request.form.get('voltage', '1')
        amp_val = request.form.get('amp_val', '1')
        amp_units = request.form.get('amp_units', 'uA/V')

        amp_val = float(amp_val)
        try:
            pressure  = float(pressure)
        except ValueError:
            pressure = 1
        energy  = float(energy)
        voltage = float(voltage)
        thick   = float(thick)

        if 'diode' in mat1:
            mat1 = mat1.replace('(diode)', '').strip()
            mat2 = 'None'
        if mat2 in (None, 'None', ''):
            mat = {mat1: 1.0}
        else:
            mat = {mat1: float(frac1), mat2: 1-float(frac1)}

        flux = xraydb.ionchamber_fluxes(mat, volts=voltage, energy=energy,
                                        length=thick*pressure,
                                        sensitivity=amp_val,
                                        sensitivity_units=amp_units)
        flux_rayleigh = flux.incident-(flux.transmitted+flux.photo+flux.incoherent)

        incident_flux = f"{flux.incident:.7g}"
        transmitted_flux =f"{flux.transmitted:.7g}"
        photo_flux = f"{flux.photo:.7g}"
        compton_flux = f"{flux.incoherent:.7g}"
        rayleigh_flux = f"{flux_rayleigh:.7g}"

        incident_flux = gformat(flux.incident)
        transmitted_flux =gformat(flux.transmitted)
        photo_flux = gformat(flux.photo)
        compton_flux = gformat(flux.incoherent)
        rayleigh_flux = gformat(flux_rayleigh)

        transmitted_percent =f"{100*flux.transmitted/flux.incident:8.4f}"
        photo_percent = f"{100*flux.photo/flux.incident:8.4f}"
        compton_percent= f"{100*flux.incoherent/flux.incident:8.4f}"
        rayleigh_percent= f"{100*flux_rayleigh/flux.incident:8.4f}"

    else:
        request.form = {'mat1': 'N2',
                        'mat2': 'None',
                        'frac1': 1.0,
                        'thick': 10.0,
                        'pressure': 1,
                        'energy':  10000,
                        'voltage': 1.000,
                        'amp_val': '1',
                        'amp_units': 'uA/V'}

    return render_template('ionchamber.html',
                           incident_flux=incident_flux,
                           transmitted_flux=transmitted_flux,
                           photo_flux=photo_flux,
                           compton_flux=compton_flux,
                           rayleigh_flux=rayleigh_flux,
                           transmitted_percent=transmitted_percent,
                           photo_percent=photo_percent,
                           compton_percent=compton_percent,
                           rayleigh_percent=rayleigh_percent,
                           mat1list=mat1list,
                           mat2list=mat2list,
                           materials_dict=materials_dict)


@app.route('/darwinwidth/', methods=['GET', 'POST'])
@app.route('/darwinwidth/<xtal>/<hkl>/<energy>/<polar>/')
def darwinwidth(xtal='Si', hkl='1 1 1', energy='10000', polar='s'):
    xtal_list = ('Si', 'Ge', 'C')

    dtheta_plot = denergy_plot = None

    if request.method == 'POST':
        xtal = request.form.get('xtal', 'Si')
        hkl = request.form.get('hkl', '1 1 1')
        polar = request.form.get('polarization', 's')
        energy = request.form.get('energy', '10000')
        do_calc = True
    else:
        do_calc = xtal in xtal_list
        if not do_calc:
            xtal = 'Si'
        request.form = {'xtal': xtal, 'hkl': hkl.replace('_', ' '),
                        'polarization':polar, 'energy':energy}

    hkl = hkl.replace('_', ' ')
    hkl_tuple = tuple([int(a) for a in hkl.split()])
    if do_calc:
        energy = float(energy)
        h_, k_, l_ = hkl_tuple
        lambd = lattice_constants[xtal] / np.sqrt(h_*h_ + k_*k_ + l_*l_)
        energy_min = PLANCK_HC /(2*lambd)
        if energy < energy_min:
            theta_deg = "not allowed"
            return render_template('darwinwidth.html',
                           energy_min=f"{energy_min:.3f}",
                           theta_deg='not allowed',
                           energy_ev='-',
                           dtheta_plot='',
                           denergy_plot='',
                           theta_urad='',
                           theta_fwhm_deg='',
                           theta_fwhm_urad='',
                           theta_width_deg='',
                           theta_width_urad='',
                           rocking_fwhm_deg='',
                           rocking_fwhm_urad='',
                           rocking_fwhm_ev='',
                           refrac_off_deg='',
                           refrac_off_urad='',
                           refrac_off_ev='',
                           xtal_list=xtal_list,
                           hkl_list=hkl_list,
                           hkl=hkl,
                           materials_dict=materials_dict)


        else:
            out = xraydb.darwin_width(energy, xtal, hkl_tuple,
                                      polarization=polar, m=1)
            title=f"{xtal}({hkl}), '{polar}' polar, E={energy:.1f} eV"
            dtheta_plot = make_plot(out.dtheta*1.e6, out.intensity, title,
                                    xtal, y1label='1 bounce',
                                    yformat='.2f',
                                    y2=out.intensity**2,
                                    y2label='2 bounces',
                                    y3=out.rocking_curve,
                                    y3label='rocking curve',
                                    ytitle='reflectivity',
                                    xtitle='Angle(microrad)')

            denergy_plot = make_plot(out.denergy, out.intensity, title,
                                     xtal, y1label='1 bounce',
                                     yformat='.2f', y2=out.intensity**2,
                                     y2label='2 bounces',
                                     y3=out.rocking_curve,
                                     y3label='rocking curve',
                                     ytitle='reflectivity',
                                     xtitle='Energy (eV)')

            def format_angle(x):
                deg = f"{(x*180/np.pi):.6f}"
                urad = f"{(x*1e6):.3f}"
                idot = deg.find('.')
                deg = deg[:idot+4] + ' ' + deg[idot+4:]
                return deg, urad

            bragg_deg, bragg_urad = format_angle(out.theta)

            bragg_ev  = f"{(energy):.3f}"

            offset_deg, offset_urad = format_angle(out.theta_offset)

            offset_ev = -out.theta_offset*out.energy_width/out.theta_width
            offset_ev = f"{(offset_ev):.3f}"

            darwin_deg, darwin_urad = format_angle(out.theta_width)
            darwin_ev = f"{(out.energy_width):.3f}"

            rocking_deg,  rocking_urad = format_angle(out.rocking_theta_fwhm)
            rocking_ev = f"{(out.rocking_energy_fwhm):.3f}"

            # energy_fwhm = f"{(out.energy_fwhm):.3f}"

        return render_template('darwinwidth.html',
                           energy_min=f"{energy_min:.3f}",
                           dtheta_plot=dtheta_plot,
                           denergy_plot=denergy_plot,
                           bragg_ev=bragg_ev,
                           bragg_deg=bragg_deg,
                           bragg_urad=bragg_urad,

                           darwin_deg=darwin_deg,
                           darwin_urad=darwin_urad,
                           darwin_ev=darwin_ev,

                           rocking_deg=rocking_deg,
                           rocking_urad=rocking_urad,
                           rocking_ev=rocking_ev,
                           offset_deg=offset_deg,
                           offset_urad=offset_urad,
                           offset_ev=offset_ev,
                           xtal_list=xtal_list,
                           hkl_list=hkl_list,
                           hkl=hkl,
                           materials_dict=materials_dict)

@app.route('/analyzers/', methods=['GET', 'POST'])
@app.route('/analyzers/<elem>',  methods=['GET', 'POST'])
@app.route('/analyzers/<elem>/<energy>/<theta1>/<theta2>',  methods=['GET', 'POST'])
def analyzers(elem='', energy='10000', theta1='60', theta2='90'):
    line = 'Ka1'
    analyzer_results = None

    if len(request.form) != 0:
        elem = request.form.get('elem', '')
        line = request.form.get('line', 'Ka1')
        energy = request.form.get('energy', '10000')
        theta1 = request.form.get('theta1', '60')
        theta2 = request.form.get('theta2', '90')
    else:
        request.form = {'theta1': theta1, 'theta2':theta2,
                        'energy': energy, 'elem': elem, 'line':'Ka1'}

    if elem != '':
        analyzer_results = []
        for xtal in ('Si', 'Ge'):
            a = lattice_constants[xtal]
            for hkl in hkl_list:
                hkl_tuple = tuple([int(ref) for ref in hkl.split()])
                hkl_link = '_'.join([ref for ref in hkl.split()])
                thbragg = th_diffracted(float(energy), hkl_tuple, a)
                if thbragg < float(theta2) and thbragg > float(theta1):
                    dw = xraydb.darwin_width(float(energy), crystal=xtal,
                                             hkl=hkl_tuple, polarization='u')
                    analyzer_results.append((xtal, hkl, hkl_link,
                                             f"{thbragg:8.4f}",
                                             f"{(dw.theta_width*1e6):8.4f}",
                                             f"{(dw.energy_width):8.4f}"))
    return render_template('analyzers.html',
                           analyzer_results=analyzer_results,
                           emission_energies=emission_energies_json,
                           analyzer_lines=analyzer_lines,
                           materials_dict=materials_dict)



@app.route('/scatteringdata/<elem>/<e1>/<e2>/<de>/<fname>')
def scatteringdata(elem, e1, e2, de, fname):
    en_array = energy_array(e1, e2, de)
    mu_total = xraydb.mu_elam(elem, en_array, kind='total')
    mu_photo = xraydb.mu_elam(elem, en_array, kind='photo')
    mu_incoh = xraydb.mu_elam(elem, en_array, kind='incoh')
    mu_coher = xraydb.mu_elam(elem, en_array, kind='coh')

    header = [f' X-ray Atomic Scattering Cross-Sections from xrayweb  {time.ctime()}',
              f' Element : {elem}',
              f' Filename : {fname}',
              ' Column.1: Energy (eV)',
              ' Column.2: mu_total (cm^2/gr)',
              ' Column.3: mu_photo (cm^2/gr)  # Photo-electric',
              ' Column.4: mu_coher (cm^2/gr)  # Rayleigh',
              ' Column.5: mu_incoh (cm^2/gr)  # Compton']

    arr_names = ['energy       ', 'mu_total     ',
                 'mu_photo     ', 'mu_coher     ',
                 'mu_incoh     ']

    arrays = [en_array, mu_total, mu_photo, mu_coher, mu_incoh]

    atz = xraydb.atomic_number(elem)
    if atz < 93:
        header.extend([' Column.6: f1 (electrons/atom) # real resonant',
                       ' Column.7: f2 (electrons/atom) # imag resonant'])

        arr_names.extend([ 'f1           ', 'f2           '])
        arrays.extend([xraydb.f1_chantler(elem, en_array), xraydb.f2_chantler(elem, en_array)])

    txt = make_asciifile(header, arr_names, arrays)
    return Response(txt, mimetype='text/plain')

@app.route('/scatteringscript/<elem>/<e1>/<e2>/<de>/<fname>')
def scatteringscript(elem, e1, e2, de, fname):
    e1 = min(EN_MAX, max(EN_MIN, float(e1)))
    e2 = min(EN_MAX, max(EN_MIN, float(e2)))
    de = max(1, float(de))
    script = """{header:s}
# X-ray atomic scattering factors
# inputs from web form
elem   = '{elem:s}'
energy = np.arange({e1:.0f}, {e2:.0f}+{de:.0f}, {de:.0f})

mu_total = xraydb.mu_elam(elem, energy, kind='total')
mu_photo = xraydb.mu_elam(elem, energy, kind='photo')
mu_incoh = xraydb.mu_elam(elem, energy, kind='incoh')
mu_coher = xraydb.mu_elam(elem, energy, kind='coh')

plt.plot(energy, mu_total, label='Total')
plt.plot(energy, mu_photo, label='Photo-electric')
plt.plot(energy, mu_incoh, label='Incoherent')
plt.plot(energy, mu_coher, label='Coherent')
plt.xlabel('Energy (eV)')
plt.ylabel(r'$\\mu/\\rho \\rm\\,(cm^2/gr)$')  # ;
plt.legend()
plt.yscale('log')
plt.title('Mass Attenuation for {elem:s}')
plt.show()

atz = xraydb.atomic_number(elem)
if atz < 93:
    f1 = xraydb.f1_chantler(elem, energy)
    f2 = xraydb.f2_chantler(elem, energy)
    plt.plot(energy, f1, label='f1')
    plt.plot(energy, f2, label='f2')
    plt.xlabel('Energy (eV)')
    plt.ylabel('f1, f2 (electrons/atom)')
    plt.title('Resonant Scattering factors for {elem:s}')
    plt.legend(True)
    plt.show()
else:
    print('f1 and f2 are only available for Z<93')


""".format(header=PY_TOP, elem=elem, e1=e1, e2=e2, de=de)
    return Response(script, mimetype='text/plain')


@app.route('/darwindata/<xtal>/<hkl>/<energy>/<polar>/<fname>')
def darwindata(xtal, hkl, energy, polar, fname):
    hkl = hkl.replace('_', ' ')
    hkl_tuple = tuple([int(a) for a in hkl.split()])
    out = xraydb.darwin_width(float(energy), xtal, hkl_tuple,
                              polarization=polar)

    header = (' X-ray Monochromator Darwin Width from xrayweb  %s ' % time.ctime(),
              ' Monochromator.xtal        : %s ' % xtal,
              ' Monochromator.hkl         : %s ' % hkl,
              ' Monochromator.polarization: \'%s\' ' % polar,
              ' Monochromator.theta       : %.6f (deg) ' % (out.theta*180/np.pi),
              ' Monochromator.theta_width : %.6f (microrad) ' % (out.theta_width*1e6),
              ' Monochromator.energy_width: %.6f (eV) ' % out.energy_width,
              ' Monochromator.theta_fwhm  : %.6f (microrad) ' % (out.theta_fwhm*1e6),
              ' Monochromator.rocking_fwhm: %.6f (microrad) ' % (out.rocking_theta_fwhm*1e6),
              ' Monochromator.energy_fwhm : %.6f (eV) ' % out.energy_fwhm,
              ' Xray.Energy               : %s (eV)' % energy,
              ' Column.1: dtheta (microrad)' ,
              ' Column.2: denergy (eV)',
              ' Column.3: zeta (delta_lambda / lambda)',
              ' Column.4: intensity',
              ' Column.5: rocking_curve')
    arr_names = ('dtheta       ', 'denergy      ',
                 'zeta         ', 'intensity    ', 'rocking_curve')

    txt = make_asciifile(header, arr_names,
                        (out.dtheta*1e6, out.denergy, out.zeta,
                         out.intensity, out.rocking_curve))

    return Response(txt, mimetype='text/plain')

@app.route('/darwinscript/<xtal>/<hkl>/<energy>/<polar>/<fname>')
def darwinscript(xtal, hkl, energy, polar,  fname):
    "create script for darwin width"
    hkl = hkl.replace('_', ' ')
    hklval = list(hkl.split())
    script = """{header:s}
# X-ray monochromator Darwin Width calculations
# inputs from web form
# filename: {fname}
xtal    = '{xtal:s}'
h, k, l = ({h:s}, {k:s}, {l:s})
polarization = '{polar:s}'
energy = {energy:s}

dw = xraydb.darwin_width(energy, xtal, (h, k, l), polarization=polarization)

print('Mono Angle: %.5f deg' % (dw.theta*180/np.pi))
print('Angular width : %.5f microrad' % (dw.theta_width*1.e6))
print('Rocking Curve width : %.5f microrad' % (dw.rocking_theta_fwhm*1.e6))
print('Energy width  : %.5f eV' % (dw.energy_width))

plt.plot(dw.denergy, dw.intensity, label='1 bounce')
plt.plot(dw.denergy, dw.rocking_curve, label='rocking curve')
plt.xlabel('Energy (eV)')
plt.ylabel('reflectivity')
plt.legend()
plt.title(f'{{xtal}} {{(h, k, l)}}, "{{polarization}}" polar, E={{energy}} eV')
plt.show()


plt.plot(dw.dtheta*1e6, dw.intensity, label='1 bounce')
plt.plot(dw.dtheta*1e6, dw.rocking_curve, label='rocking curve')
plt.xlabel('Angle (microrad)')
plt.ylabel('reflectivity')
plt.legend()
plt.title(f'{{xtal}} {{(h, k, l)}}, "{{polarization}}" polar, E={{energy}} eV')
plt.show()
""".format(header=PY_TOP, fname=fname, xtal=xtal,
           h=hklval[0], k=hklval[1], l=hklval[2], polar=polar, energy=energy)
    return Response(script, mimetype='text/plain')


@app.route('/elementscript/<elem>/<fname>')
def elementscript(elem, fname):
    "output script for element properties"
    script = """{header:s}
# X-ray propertie
# filename: {fname}
elem  = '{elem:s}'
print('# Atomic Symbol: %s ' % elem)
print('# Atomic Number: %d ' % xraydb.atomic_number(elem))
print('# Atomic Moss:   %.4f ' % xraydb.atomic_mass(elem))

print('# X-ray Edges:')
print('#  Edge     Energy    Width  FlourYield  EdgeJump')
e_fmt = '  %5s  %9.1f  %7.4f    %8.5f  %8.5f'
l_fmt = '%7s  %9.1f   %8.5f  %11s'
widths = dict(xraydb.core_width(elem))
for key, val in xraydb.xray_edges(elem).items():
     wid = widths.get(key, 0)
     print(e_fmt % (key, val.energy, wid, val.fyield, val.jump_ratio))

print('# X-ray Lines:')
print('#  Line     Energy  Intensity       Levels')
for key, val in xraydb.xray_lines(elem).items():
     levels = '%s-%s' % (val.initial_level, val.final_level)
     print(l_fmt % (key, val.energy, val.intensity, levels))
""".format(header=PY_TOP, elem=elem, fname=fname)
    return Response(script, mimetype='text/plain')


@app.route('/attendata/<formula>/<rho>/<t>/<e1>/<e2>/<de>/<fname>')
def attendata(formula, rho, t, e1, e2, de, fname):
    "attenuation data"
    en_array = energy_array(e1, e2, de)
    rho = float(rho)

    try:
        _ = chemparse(formula)
        parsed = True
    except:
        txt = f"could not parse formula: '{formula}'"
        parsed = False
    if parsed:
        mu_array = xraydb.material_mu(formula, en_array, density=rho)
        t = float(t)
        trans = np.exp(-0.1*t*mu_array)
        xatten = 1 - trans

        header = (f' X-ray attenuation data from xrayweb  {time.ctime()}',
                  f' File.Name: {fname}',
                  f' Material.formula   : {formula}',
                  f' Material.density   : {rho:.4f} gr/cm^3 ',
                  f' Material.thickness : {t:.5f} mm ',
                  ' Column.1: energy (eV)',
                  ' Column.2: attenuation_length (mm)' ,
                  ' Column.3: transmitted_fraction',
                  ' Column.4: attenuated_fraction')

        arr_names = ('energy       ', 'atten_length ',
                     'trans_fract  ', 'atten_fract  ')

        txt = make_asciifile(header, arr_names,
                             (en_array, 10/mu_array, trans, xatten))

    return Response(txt, mimetype='text/plain')

@app.route('/attenscript/<formula>/<rho>/<t>/<e1>/<e2>/<de>/<fname>')
def attenscript(formula, rho, t, e1, e2, de, fname):
    """attenuation data as python code"""
    e1 = min(EN_MAX, max(EN_MIN, float(e1)))
    e2 = min(EN_MAX, max(EN_MIN, float(e2)))
    de = max(1, float(de))
    script = """{header:s}
# X-ray attenuation calculations
# inputs from web form
# Filename: {fname}
formula = '{formula:s}'  # material chemical formula
density = {density:.4f}  # material density in gr/cm^3
thickness = {thick:.6f}  # material thickness, in mm
energy = np.arange({e1:.0f}, {e2:.0f}+{de:.0f}, {de:.0f})

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
""".format(header=PY_TOP, formula=formula, fname=fname,
           density=float(rho), thick=float(t),
           e1=e1, e2=e2, de=de)
    return Response(script, mimetype='text/plain')



@app.route('/reflectdata/<formula>/<density>/<angle>/<rough>/<polar>/<e1>/<e2>/<de>/<fname>')
def reflectdata(formula, density, angle, rough, polar, e1, e2, de, fname):
    """mirror reflectivity data as file"""
    en_array = energy_array(e1, e2, de)
    angle = float(angle)
    density  = float(density)
    rough = float(rough)
    refl = xraydb.mirror_reflectivity(formula, 0.001*angle, en_array, density,
                                              roughness=rough, polarization=polar)

    _del, _bet, _ = xraydb.xray_delta_beta(formula, density, en_array)
    ang_crit = 1000*(np.pi/2 - np.arcsin(1 - _del - 1j*_bet)).real

    header = (f' X-ray reflectivity data from xrayweb  {time.ctime()}',
              f' Material.formula   : {formula}',
              f' Material.density   : {density:.4f} gr/cm^3',
              f' Material.angle     : {angle:.4f} mrad',
              f' Material.roughness : {rough:.4f} Ang',
              f' Material.polarization: {polar}',
              ' Column.1: energy (eV)',
              ' Column.2: reflectivity',
              ' Column.3: critical_angle (mrad)')

    arr_names = ('energy       ', 'reflectivity ', 'crit_angle    ')
    txt = make_asciifile(header, arr_names, (en_array, refl, ang_crit))
    return Response(txt, mimetype='text/plain')


@app.route('/reflectscript/<formula>/<density>/<angle>/<rough>/<polar>/<e1>/<e2>/<de>/<fname>')
def reflectscript(formula, density, angle, rough, polar, e1, e2, de, fname):
    """mirror reflectivity data as python code"""
    e1 = min(EN_MAX, max(EN_MIN, float(e1)))
    e2 = min(EN_MAX, max(EN_MIN, float(e2)))
    de = max(1, float(de))
    script = """{header:s}
# mirror reflectivity calculations
# inputs from web form
formula = '{formula:s}'  # mirror chemical formula
density = {density:.4f}  # mirror density in gr/cm^3
angle   = {angle:.4f}  # mirror angle in mrad
rough   = {rough:.4f}  # mirror roughness in Angstroms
polar   = '{polar:s}'  # mirror polarization ('s' for vert deflecting with horiz-polarized source)
energy = np.arange({e1:.0f}, {e2:.0f}+{de:.0f}, {de:.0f})


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
""".format(header=PY_TOP, formula=formula, density=float(density),
           angle=float(angle), rough=float(rough), polar=polar,
           e1=e1, e2=e2, de=de)
    return Response(script, mimetype='text/plain')

@app.route('/fluxscript/<mat1>/<mat2>/<frac1>/<thick>/<pressure>/<energy>/<voltage>/<amp_val>/<amp_units>/<fname>')
def fluxscript(mat1, mat2='', frac1='1', thick='10', pressure='1', energy='10000',
               voltage='1', amp_val='1', amp_units='nA_V', fname='xrayweb_flux.py'):
    """ion chamber flux script"""
    if 'diode' in mat1:
        mat1 = mat1.replace('(diode)', '').replace('%20', '').strip()
        mat2 = 'None'
    script = """{header:s}
# X-ray ion chamber flux calculation
# inputs from web form
mat1 = '{mat1:s}'
mat2 = '{mat2:s}'
frac1 = {frac1:s}
thick = {thick:s}
pressure = {pressure:s}
voltage = {voltage:s}
energy = {energy:s}
amp_val = {amp_val:s}
amp_units = '{amp_units:s}'

if mat2 in (None, 'None', ''):
    mat = {{mat1: 1.0}}
else:
    mat = {{mat1: float(frac1), mat2: 1-float(frac1)}}

flux = xraydb.ionchamber_fluxes(mat, volts=voltage, energy=energy,
                                length=thick*pressure,
                                sensitivity=amp_val,
                                sensitivity_units=amp_units)

print(f'Incident to Detector: {{flux.incident:.7g}}')
print(f'Transmitted out of Detector: {{flux.transmitted:.7g}}')
print(f'Absorbed for Photo Current: {{flux.photo:.7g}}')
print(f'Scattered by Compton Effect: {{flux.incoherent:.7g}}')

""".format(header=PY_TOP, mat1=mat1, mat2=mat2, frac1=frac1, thick=thick,
           pressure=pressure, voltage=voltage, energy=energy, amp_val=amp_val,
           amp_units=amp_units.replace('_', '/'))
    return Response(script, mimetype='text/plain')

@app.route('/analyzerscript/<energy>/<theta1>/<theta2>/<fname>')
def analyzerscript(energy, theta1, theta2, fname):
    """analyzer script"""
    script = """{header:s}

# PLANCK_HC = 12398.419843320  # eV*Angstroms
import scipy.constants as consts
PLANCK_HC = 1.e10 * consts.Planck * consts.c / consts.e

# list of allowed reflections for diamond structure: duplicates removed
hkl_list = ('1 1 1', '2 2 0', '3 1 1', '3 3 1', '3 3 3', '4 0 0',
            '4 2 2', '4 4 0', '4 4 4', '5 3 1', '5 3 3', '5 5 1',
            '5 5 3', '5 5 5', '6 2 0', '6 4 2', '6 6 0', '6 6 4',
            '7 3 3', '7 5 3', '7 5 5', '7 7 3', '7 7 5', '7 7 7',
            '8 0 0', '8 4 0', '8 4 4', '8 6 2', '8 6 6', '8 8 0',
            '8 8 4', '8 8 8', '9 3 1', '9 5 3', '9 5 5', '9 7 3',
            '9 7 5', '9 7 7', '9 9 1', '9 9 3', '9 9 5', '9 9 7',
            '9 9 9', '10 4 2', '10 6 4', '10 8 2',  '10 10 0',
            '10 10 4', '10 10 8', '11 7 5', '11 7 7', '11 9 1',
            '11 9 5', '11 9 7', '11 9 9', '11 11 5', '11 11 7',
            '11 11 9', '11 11 11')

lattice_constants = {{'Si': 5.4309, 'Ge': 5.6578}}


def theta_bragg(energy, hkl, a):
    '''Bragg diffraction angle for energy in eV, lattice constant in Ang
    returns 0 if invalid refletion
    '''
    hkl = np.array(hkl)
    omega = np.sqrt((hkl*hkl).sum()) * PLANCK_HC / (2 * a * energy)
    if abs(omega) > 1:
        omega = -0
    return (180/np.pi)*np.arcsin(omega)


fmt = '    %s (%s)          %7.3f        %7.3f            %7.3f '
legend = '#Cystal/Reflection  BraggAngle(deg)  DarwinWith (urad)  DarwinWidth (eV)'

def show_analyzers(energy, theta_min=60, theta_max=90):
    '''print table of candidate analyzers for energy and angle range'''
    print(legend)
    for xtal in ('Si', 'Ge'):
        a = lattice_constants[xtal]
        for hkl in hkl_list:
            hkl_tuple = tuple([int(ref) for ref in hkl.split()])
            thbragg = theta_bragg(energy, hkl_tuple, a)
            if thbragg < theta_max and thbragg > theta_min:
                dw = xraydb.darwin_width(energy, crystal=xtal,
                                         hkl=hkl_tuple, polarization='u')
                print(fmt % (xtal, hkl, thbragg,
                             dw.theta_width*1e6, dw.energy_width))

show_analyzers({energy:s}, theta_min={theta1:s}, theta_max= {theta2:s})

""".format(header=PY_TOP, energy=energy, theta1=theta1, theta2=theta2)
    return Response(script, mimetype='text/plain')


@app.route('/transmission_sample/', methods=['GET', 'POST'])
@app.route('/transmission_sample/<sample>/<energy>/<absorp_total>/<area>/<density>/', methods=['GET', 'POST'])
def transmission_sample(sample=None, energy=None, absorp_total=None, area=None, density=None):
    result = None

    if request.method == 'POST':
        sample = {}
        for i in range(1, 11):
            name = request.form.get(f'component{i}-name')
            if name:
                try:
                    val = float(request.form.get(f'component{i}-frac'))
                except ValueError:
                    val = -1
                sample[name] = val
        energy = float(request.form.get('energy'))
        absorp_total = float(request.form.get('absorp_total'))
        area = float(request.form.get('area'))
        density = request.form.get('density', None)
        if density:
            density = float(density)

        frac_type = request.form.get('frac_type')
        s = xraydb.transmission_sample(sample=sample, energy=energy, absorp_total=absorp_total,
                                       area=area, density=density, frac_type=frac_type)

        result = {}
        result['Energy (eV)'] = f'{s.energy_eV:.2f}'
        result['Density (gr/cm^3)'] = 'None' if not density else f'{s.density:.2f}'
        result['Area (cm^2)'] = f'{s.area_cm2:.2f}'
        result['Total Absorption'] = f'{s.absorp_total:.2f}'
        result['Thickness (\u03bCm)'] = 'Requires Density' if not s.thickness_mm else f'{s.thickness_mm*1000.0:.2f}'
        result['Absorption length (\u03bCm)'] = 'Requires Density' if not s.absorption_length_um else f'{s.absorption_length_um:.2f}'
        result['Total Mass (mg)'] = f'{s.mass_total_mg:.2f}'

        mass_fracs = [f'{el:s}:{mass:.3f}' for el, mass in s.mass_fractions.items()]
        result['Mass Fractions'] = ', '.join(mass_fracs)

        masses = [f'{el:s}:{mass:.3f}' for el, mass in s.mass_components_mg.items()]
        result['Element Masses (mg)'] = ', '.join(masses)

        steps = [f'{el:s}:{step:.3f}' for el, step in s.absorbance_steps.items()]
        result['Absorbance steps'] = ', '.join(steps)

        if not request.form.get('getpythonscript'):
            return render_template('transmission_sample.html', materials_dict=materials_dict,
                                result=result)
        else:
            if not density:
                density = 'None'
            script = """{header:s}

# XAFS transmission mode sample calculation
# inputs from web form
sample = {sample}
energy = {energy}
absorp_total = {absorp_total}
area = {area}
density = {density}
frac_type = '{frac_type:s}'

samp = xraydb.transmission_sample(sample=sample, energy=energy, absorp_total=absorp_total,
                                       area=area, density=density, frac_type=frac_type)

print(samp)
""".format(header=PY_TOP, sample=sample, energy=energy,
absorp_total=absorp_total, area=area, density=density, frac_type=frac_type)
            return Response(script, mimetype='text/plain')

    return render_template('transmission_sample.html', materials_dict=materials_dict,
                            result=result)
