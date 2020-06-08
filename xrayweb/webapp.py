#!/usr/bin/env python

from flask import Flask, redirect, url_for, render_template
from flask import request, session

import xraydb

db = xraydb.get_xraydb()

app = Flask('xrayweb', static_folder='static')
app.config.from_object(__name__)


@app.route('/element/', methods=['GET', 'POST'])
@app.route('/element/<elem>',  methods=['GET', 'POST'])
def element(elem=None):

    global db 
    edges = {}
    if elem is not None:
        edges= db.xray_edges(elem)
    print("ELEMENT ", db, edges)
    return render_template('elements.html', edges=edges, elem=elem)

@app.route('/')
def index():
    return redirect(url_for('element'))


if __name__ == "__main__":
    app.jinja_env.cache = {}
    app.run(port=PORT)
