#!/usr/bin/env python
"""
test the xray Web App
"""

from werkzeug.middleware.profiler import ProfilerMiddleware
from xrayweb import app

app.jinja_env.cache = {}
app.secret_key = 'this is a secret key'
app.run(debug=True, port=4966)
