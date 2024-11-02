# xrayweb
Web interface for X-ray properties of the elements,
using https://github.com/xraypy/XrayDB,
Python, Flask, and Plotly.

Existing versions of this wep application  are running at

 https://seescience.org/xraydb and at
 https://xraydb.xrayabsorption.org


## install

XrayWeb requires Python 3.9 or higher

To install this code, use


    > pip install xrayweb

To run a local version of this web application, run the script "run.py" with

      > python run.py

will launch a local web server with the app running at http://127.0.0.1:4966/


## deploying with Apache and WSGI

To deploy this web application with Apache, you will need to install `mod_wsgi`
(for Python3) for your instance of Apache.  You will also have create a wsgi
python script that can be run by the web server.

For examplle, you may make a folder `/var/www/xraydb` and place in the file
`xraydb.wsgi`, containing

    #!/uar/bin/python
    # file /var/www/xraydb/xraydb.wsgi
    from xrayweb import app as application


Then,  you will need to load the wsgi module in your Apache configuration, with


    # make sure wsgi module is loaded
    <IfModule !wsgi_module>
        LoadModule wsgi_module modules/mod_wsgi_python3.so
    </IfModule>

    # define /xraydb URL
    WSGIDaemonProcess xraydb user=apache group=apache threads=5
    WSGIScriptAlias /xraydb /var/www/xraydb/xraydb.wsgi
    <Directory /var/www/xraydb>
       WSGIProcessGroup xraydb
       WSGIApplicationGroup %{GLOBAL}
       Options all
       Require all granted
    </Directory>

Restarting apache,  the script should run on your web server at
https://example.com/xraydb
