#!/usr/bin/env python

from flask import Flask, redirect, url_for, render_template
from flask import request, session

import xraydb


app = Flask('xrayweb', static_folder='static')
app.config.from_object(__name__)

@app.route('/element/', methods=['GET', 'POST'])
@app.route('/element/<elem>',  methods=['GET', 'POST'])
def element(elem=None):
    edges = {}
    if elem is not None:
        edges= xraydb.xray_edges(elem)
    return render_template('elements.html', edges=edges, elem=elem)

@app.route('/')
def index():
    return redirect(url_for('element'))
