#!/usr/bin/env python
"""
Production server runner for Tapsi Food Map Dashboard
Works on both Windows and Linux
"""

import os
import sys
import platform
import multiprocessing

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, load_data

def get_worker_count():
    """Calculate optimal worker count based on CPU cores"""
    # Use (2 * CPU cores + 1) as recommended by gunicorn
    return (multiprocessing.cpu_count() * 2) + 1

def run_production_server():
    """Run the production server with appropriate configuration"""
    # Load data before starting server
    print("Loading data for production server...")
    load_data()
    
    # Determine the operating system
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        # On Windows, use waitress (pure Python WSGI server)
        try:
            from waitress import serve
            print("Starting Waitress server (Windows)...")
            print(f"Server running on http://0.0.0.0:5001")
            serve(app, host='0.0.0.0', port=5001, threads=8)
        except ImportError:
            print("Waitress not installed. Please install it with: pip install waitress")
            print("Falling back to Flask development server...")
            app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        # On Linux/Unix, use gunicorn
        try:
            import gunicorn.app.base
            
            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()
                
                def load_config(self):
                    for key, value in self.options.items():
                        if key in self.cfg.settings and value is not None:
                            self.cfg.set(key.lower(), value)
                
                def load(self):
                    return self.application
            
            options = {
                'bind': '0.0.0.0:5001',
                'workers': get_worker_count(),
                'worker_class': 'sync',
                'timeout': 120,  # Increase timeout for coverage grid calculations
                'keepalive': 2,
                'threads': 2,
                'accesslog': '-',  # Log to stdout
                'errorlog': '-',   # Log to stdout
                'preload_app': True,  # Load app before forking workers
            }
            
            print(f"Starting Gunicorn server (Linux/Unix)...")
            print(f"Workers: {options['workers']}")
            print(f"Server running on http://0.0.0.0:5001")
            
            StandaloneApplication(app, options).run()
            
        except ImportError:
            print("Gunicorn not installed. Please install it with: pip install gunicorn")
            print("Falling back to Flask development server...")
            app.run(host='0.0.0.0', port=5001, debug=False)

if __name__ == '__main__':
    run_production_server()