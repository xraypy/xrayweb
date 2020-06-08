#!/usr/bin/env python
"""
test the xray Web App
"""

from xrayweb import app

app.jinja_env.cache = {}
app.run(debug=True, port=4966)
