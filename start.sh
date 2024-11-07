#!/bin/sh
gunicorn --bind 0.0.0.0:$PORT chat:app
