# gridlandia
A repo for demoing a small energy systems model with a user interface.

## run

    gunicorn --bind 0.0.0.0:5000 wsgi:app
    
    flask run --host=0.0.0.0
    
    
## Docker

    docker compose up