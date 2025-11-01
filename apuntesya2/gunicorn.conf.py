# Gunicorn configuration
import multiprocessing

bind = "0.0.0.0:10000"
timeout = 120
workers = max(2, multiprocessing.cpu_count() * 2 + 1)
worker_tmp_dir = "/dev/shm"
# accesslog = "-"
# errorlog = "-"
# loglevel = "info"
