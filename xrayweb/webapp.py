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
              xtitle='Energy (eV)', y1label='data',
              xlog_scale=False, ylog_scale=False, yrange=None,
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
@app.route('/scattering/<elem>/<e1>/<e2>/<de>/<mode>',  methods=['GET', 'POST'])
def scattering(elem=None, e1='1000', e2='50000', de='50', mode='Log'):
    f1f2_plot = mu_plot = None
    if len(request.form) != 0:
        elem = request.form.get('elem', 'None')
        e1 = request.form.get('e1', e1)
        e2 = request.form.get('e2', e2)
        de = request.form.get('de', de)
        mode = request.form.get('mode', mode)

    if elem not in (None, 'None', ''):
        energy = np.arange(float(e1), float(e2)+float(de), float(de))

        mu_total = xraydb.mu_elam(elem, energy, kind='total')
        mu_photo = xraydb.mu_elam(elem, energy, kind='photo')
        mu_incoh = xraydb.mu_elam(elem, energy, kind='incoh')
        mu_coher = xraydb.mu_elam(elem, energy, kind='coh')
        yrange = [-0.25+min(-1.8,
                            np.log10(mu_photo.min()+1.e-5),
                            np.log10(mu_incoh.min()+1.e-5),
                            np.log10(mu_coher.min()+1.e-5)),
                  0.75+np.log10(mu_total.max()+1.e-5)]

        mu_plot = make_plot(energy, mu_total, 'Mass Attenuation for %s' %
                            elem, elem, ytitle='mu/rho (cm^2/gr)',
                            xtitle='Energy (eV)', xlog_scale=False,
                            ylog_scale=True, yrange=yrange,
                            y1label='Total',
                            y2=mu_photo, y2label='Photo-electric',
                            y3=mu_incoh, y3label='Inchorent',
                            y4=mu_coher, y4label='Coherent')

        f1 = xraydb.f1_chantler(elem, energy)
        f2 = xraydb.f2_chantler(elem, energy)
        f1f2_plot = make_plot(energy, f1, 'Resonant Scattering factors for %s' % elem,
                              elem, ytitle='f1, f2 (electrons/atom)',
                              xtitle='Energy (eV)',
                              xlog_scale=False, ylog_scale=False, y2=f2,
                              y1label='f1', y2label='f2')

    return render_template('scattering.html', elem=elem, e1=e1, e2=e2,
                           f1f2_plot=f1f2_plot, mu_plot=mu_plot,
                           de=de, mode=mode, materials_dict=materials_dict)


@app.route('/ionchamber/', methods=['GET', 'POST'])
def ionchamber(elem=None):
    message = []

    incident_flux = transmitted_flux = photo_flux = ''
    mat1list = ('He', 'N2', 'Ne', 'Ar', 'Kr', 'Xe') # 'Si (diode)', 'Ge (diode)')
    mat2list = ('None', 'He', 'N2', 'Ne', 'Ar', 'Kr', 'Xe')

    if request.method == 'POST':
        mat1 = request.form.get('mat1', 'None')
        mat2 = request.form.get('mat2', 'None')
        frac1 = request.form.get('frac1', '1')
        thick = request.form.get('thick', '1')
        pressure = request.form.get('pressure', '1')
        energy = request.form.get('energy', '10000')
        voltage = request.form.get('voltage', '1')
        amp_val = request.form.get('amp_val', '1')
        amp_units = request.form.get('amp_units', 'uA/V')

        amp_val = float(amp_val)
        pressure  = float(pressure)
        energy  = float(energy)
        voltage = float(voltage)
        thick   = float(thick)

        if 'diode' in mat1:
            mat1 = mat1.replace(' (diode)', '')
            if mat1.startswith('Si'):
                wfun = 3.68 # Si effective ionization potential
            else:
                wfun = 2.97 # Ge effective ionization potential

        else:
            if mat2 in (None, 'None', ''):
                mat = {mat1: 1.0}
            else:
                mat = {mat1: float(frac1), mat2: 1-float(frac1)}

        flux = xraydb.ionchamber_fluxes(mat, volts=voltage, energy=energy,
                                        length=thick*pressure,
                                        sensitivity=amp_val,
                                        sensitivity_units=amp_units)

        incident_flux = "%.7g" % flux.incident
        transmitted_flux = "%.7g" % flux.transmitted
        photo_flux = "%.7g" % flux.photo
    else:
        request.form = {'mat1': 'N2',
                        'mat2': 'None',
                        'frac1': 1.0,
                        'thick': 100.0,
                        'pressure': 1,
                        'energy':  10000,
                        'voltage': 1.000,
                        'amp_val': '1',
                        'amp_units': 'uA/V'}

    return render_template('ionchamber.html',
                           incident_flux=incident_flux,
                           transmitted_flux=transmitted_flux,
                           photo_flux=photo_flux,
                           mat1list=mat1list,
                           mat2list=mat2list,
                           materials_dict=materials_dict)


@app.route('/darwinwidth/', methods=['GET', 'POST'])
def darwinwidth():
    xtal_list = ('Si', 'Ge', 'C')
    hkl_list = ('111', '220', '311', '331', '333', '400',
                '422', '440', '444', '511', '531', '533',
                '551', '553', '555', '620', '642', '660',
                '664', '711', '731', '733', '751', '753',
                '755', '771', '773', '775', '777', '800',
                '822', '840', '844', '862', '866', '880',
                '884', '888', '911', '931', '933', '951',
                '953', '955', '971', '973', '975', '977',
                '991', '993', '995', '997', '999')
    dtheta_plot = denergy_plot = None
    theta_deg = theta_fwhm = energy_fwhm = ''
    if request.method == 'POST':
        xtal = request.form.get('xtal', 'Si')
        hkl = request.form.get('hkl', '111')
        harmonic= request.form.get('harmonic', '1')
        energy = request.form.get('energy', '10000')

        hkl_tuple = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
        m = int(harmonic)
        energy = float(energy)
        out = xraydb.darwin_width(energy, xtal, hkl_tuple, m=m)

        title='%s(%s), order=%d, E=%.1f eV' % (xtal, hkl, m, energy)
        dtheta_plot = make_plot(out.dtheta*1.e6, out.intensity,  title, xtal,
                                ytitle='reflectivity', xtitle='Angle (microrad)')

        denergy_plot = make_plot(out.denergy, out.intensity,  title, xtal,
                                ytitle='reflectivity', xtitle='Energy (eV)')

        theta_deg = "%.5f" % (out.theta * 180 / np.pi)
        theta_fwhm = "%.5f" % (out.theta_fwhm * 1.e6)
        energy_fwhm = "%.5f" % out.energy_fwhm
    else:
        request.form = {'xtal': 'Si', 'hkl':'111',
                        'harmonic':'1', 'energy':'10000'}

    return render_template('darwinwidth.html',
                           dtheta_plot=dtheta_plot,
                           denergy_plot=denergy_plot,
                           theta_deg=theta_deg,
                           theta_fwhm=theta_fwhm,
                           energy_fwhm=energy_fwhm,
                           xtal_list=xtal_list,
                           hkl_list=hkl_list,
                           materials_dict=materials_dict)


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



@app.route('/scatteringdata/<elem>/<e1>/<e2>/<de>/<mode>/<fname>')
def scatteringdata(elem, e1, e2, de, mode, fname):

    energy = np.arange(float(e1), float(e2)+float(de), float(de))

    mu_total = xraydb.mu_elam(elem, energy, kind='total')
    mu_photo = xraydb.mu_elam(elem, energy, kind='photo')
    mu_incoh = xraydb.mu_elam(elem, energy, kind='incoh')
    mu_coher = xraydb.mu_elam(elem, energy, kind='coh')
    f1 = xraydb.f1_chantler(elem, energy)
    f2 = xraydb.f2_chantler(elem, energy)

    header = (' X-ray Atomic Scattering Cross-Sections from xrayweb  %s ' % time.ctime(),
              ' Element : %s ' % elem,
              ' Column.1: Energy (eV)',
              ' Column.2: mu_total (cm^2/gr)',
              ' Column.3: mu_photo (cm^2/gr)  # Photo-electric',
              ' Column.4: mu_coher (cm^2/gr)  # Rayleigh',
              ' Column.5: mu_incoh (cm^2/gr)  # Compton',
              ' Column.6: f1 (electrons/atom) # real resonant',
              ' Column.7: f2 (electrons/atom) # imag resonant')

    arr_names = ('energy       ', 'mu_total     ',
                 'mu_photo     ', 'mu_coher     ',
                 'mu_incoh     ', 'f1           ', 'f2           ')

    txt = make_asciifile(header, arr_names,
                         (energy, mu_total, mu_photo, mu_coher, mu_incoh, f1, f2))
    return Response(txt, mimetype='text/plain')

@app.route('/scatteringscript/<elem>/<e1>/<e2>/<de>/<mode>/<fname>')
def scatteringscript(elem, e1, e2, de, mode, fname):
    script = """#!/usr/bin/env python
#
# X-ray atomic scattering factors
# this requires Python3, numpy, matplotlib, and xraydb modules. Use:
#        pip install xraydb

import numpy as np
import matplotlib.pyplot as plt
import xraydb

# inputs from web form
elem    = '{elem:s}'
mode    = '{mode:s}'
energy = np.arange({e1:.0f}, {e2:.0f}+{de:.0f}, {de:.0f})

mu_total = xraydb.mu_elam(elem, energy, kind='total')
mu_photo = xraydb.mu_elam(elem, energy, kind='photo')
mu_incoh = xraydb.mu_elam(elem, energy, kind='incoh')
mu_coher = xraydb.mu_elam(elem, energy, kind='coh')

f1 = xraydb.f1_chantler(elem, energy)
f2 = xraydb.f2_chantler(elem, energy)

plt.plot(energy, f1, label='f1')
plt.plot(energy, f2, label='f2')
plt.xlabel('Energy (eV)')
plt.ylabel('f1, f2 (electrons/atom)')
plt.title('Resonant Scattering factors for {elem:s}')
plt.legend(True)
plt.show()

plt.plot(energy, mu_total, label='Total')
plt.plot(energy, mu_photo, label='Photo-electric')
plt.plot(energy, mu_incoh, label='Incoherent')
plt.plot(energy, mu_coher, label='Coherent')
plt.xlabel('Energy (eV)')
plt.ylabel(r'$\mu/\\rho \\rm\,(cm^2/gr)$')
plt.legend()
plt.yscale(mode.lower())
plt.title('Mass Attenuation for {elem:s}')
plt.show()

""".format(elem=elem, e1=float(e1), e2=float(e2), de=float(de), mode=mode)
    return Response(script, mimetype='text/plain')


@app.route('/darwindata/<xtal>/<hkl>/<m>/<energy>/<fname>')
def darwindata(xtal, hkl, m, energy, fname):
    hkl_tuple = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
    out = xraydb.darwin_width(float(energy), xtal, hkl_tuple, m=int(m))

    header = (' X-ray Monochromator Darwin Width from xrayweb  %s ' % time.ctime(),
              ' Monochromator.xtal       : %s ' % xtal,
              ' Monochromator.hkl        : %s ' % hkl,
              ' Monochromator.harmonic   : %s ' % m,
              ' Monochromator.theta      : %.5f (deg) ' % (out.theta*180/np.pi),
              ' Monochromator.theta_fwhm : %.5f (microrad) ' % (out.theta_fwhm*1e6),
              ' Monochromator.energy_fwhm: %.5f (eV) ' % out.energy_fwhm,
              ' Xray.Energy              : %s (eV)' % energy,
              ' Column.1: dtheta (microrad)' ,
              ' Column.2: denergy (eV)',
              ' Column.3: zeta (delta_lambda / lambda)',
              ' Column.4: intensity')
    arr_names = ('dtheta       ', 'denergy      ',
                 'zeta         ', 'intensity    ')

    txt = make_asciifile(header, arr_names,
                         (out.dtheta*1e6, out.denergy, out.zeta, out.intensity))

    return Response(txt, mimetype='text/plain')

@app.route('/darwinscript/<xtal>/<hkl>/<m>/<energy>/<fname>')
def darwinscript(xtal, hkl, m, energy, fname):
    script = """#!/usr/bin/env python
#
# X-ray monochromator Darwin Width calculations
# this requires Python3, numpy, matplotlib, and xraydb modules. Use:
#        pip install xraydb

import numpy as np
import matplotlib.pyplot as plt
import xraydb

# inputs from web form
xtal    = '{xtal:s}'
h, k, l = ({h:s}, {k:s}, {l:s})
harmonic = {m:s}
energy = {energy:s}

dw = xraydb.darwin_width(energy, xtal, (h, k, l), m=harmonic)

print('Mono Angle: %.5f deg' % (dw.theta*180/np.pi))
print('Angular width FWHM: %.5f microrad' % (dw.theta_fwhm*1.e6))
print('Energy width FWHM: %.5f eV' % (dw.energy_fwhm))

plt.plot(dw.denergy, dw.intensity)
plt.xlabel('Energy (eV)')
plt.ylabel('reflectivity')
plt.title('{xtal:s} ({hkl:s}), order={m:s}, E={energy:s} eV')
plt.show()

plt.plot(dw.dtheta*1e6, dw.intensity)
plt.xlabel('Angle (microrad)')
plt.ylabel('reflectivity')
plt.title('{xtal:s} ({hkl:s}), order={m:s}, E={energy:s} eV')
plt.show()

""".format(xtal=xtal, hkl=hkl, h=hkl[0], k=hkl[1], l=hkl[2], m=m, energy=energy)
    return Response(script, mimetype='text/plain')


@app.route('/attendata/<formula>/<rho>/<t>/<e1>/<e2>/<estep>/<fname>')
def attendata(formula, rho, t, e1, e2, estep, fname):
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
              ' Column.2: attenuation_length (mm)' ,
              ' Column.3: transmitted_fraction',
              ' Column.4: attenuated_fraction')

    arr_names = ('energy       ', 'atten_length ',
                 'trans_fract  ', 'atten_fract  ')

    txt = make_asciifile(header, arr_names,
                         (en_array, 10/mu_array, trans, atten))

    return Response(txt, mimetype='text/plain')

@app.route('/attenscript/<formula>/<rho>/<t>/<e1>/<e2>/<estep>/<fname>')
def attenscript(formula, rho, t, e1, e2, estep, fname):
    """attenuation data as python code"""
    script = """#!/usr/bin/env python
#
# X-ray attenuation calculations
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
    return Response(script, mimetype='text/plain')



@app.route('/reflectdata/<formula>/<rho>/<angle>/<rough>/<polar>/<e1>/<e2>/<estep>/<fname>')
def reflectdata(formula, rho, angle, rough, polar, e1, e2, estep, fname):
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
              ' Column.3: critical_angle (mrad)')

    arr_names = ('energy       ', 'reflectivity ', 'crit_angle    ')
    txt = make_asciifile(header, arr_names, (en_array, reflectivity, ang_crit))
    return Response(txt, mimetype='text/plain')


@app.route('/reflectscript/<formula>/<rho>/<angle>/<rough>/<polar>/<e1>/<e2>/<estep>/<fname>')
def reflectscript(formula, rho, angle, rough, polar, e1, e2, estep, fname):
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
    return Response(script, mimetype='text/plain')

@app.route('/fluxscript/<mat1>/<mat2>/<frac1>/<thick>/<pressure>/<energy>/<voltage>/<amp_val>/<amp_units>/<fname>')
def fluxscript(mat1, mat2, frac1, thick, pressure, energy,
               voltage, amp_val, amp_units, fname):
    """ion chamber flux script"""
    script = """#!/usr/bin/env python
#
# X-ray ion chamber flux calculation
# this requires Python3, numpy, matplotlib, and xraydb modules. Use:
#        pip install xraydb

import numpy as np
import matplotlib.pyplot as plt
import xraydb

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
    mat = dict(mat1=1.0)
else:
    mat = dict(mat1=float(frac1), mat2=1-float(frac1))

flux = xraydb.ionchamber_fluxes(mat, volts=voltage, energy=energy,
                                length=thick*pressure,
                                sensitivity=amp_val,
                                sensitivity_units=amp_units)

print('Incident to Detector: %.7g' % flux.incident)
print('Absorbed for Photo Current: %.7g ' % flux.photo)
print('Transmitted out of Detector: %.7g ' % flux.transmitted)



""".format(mat1=mat1, mat2=mat2, frac1=frac1, thick=thick, pressure=pressure,
           voltage=voltage, energy=energy, amp_val=amp_val,
           amp_units=amp_units.replace('_', '/'))

    return Response(script, mimetype='text/plain')
