#!/usr/bin/env python

from flask import Flask, redirect, url_for, render_template
from flask import request, session

import xraydb


app = Flask('xrayweb', static_folder='static')
app.config.from_object(__name__)

@app.route('/element/', methods=['GET', 'POST'])
@app.route('/element/<elem>',  methods=['GET', 'POST'])
def element(elem=None):
    edges = atomic = {}
    if elem is not None:
        edges= xraydb.xray_edges(elem)
        atomic= {'n': xraydb.atomic_number(elem), 'mass': xraydb.atomic_mass(elem)}
    return render_template('elements.html', edges=edges, elem=elem, atomic=atomic)

@app.route('/')
def index():
    return redirect(url_for('element'))
