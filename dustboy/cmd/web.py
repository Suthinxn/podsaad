from dustboy import web


def main():
    app = web.create_app()
    options = web.get_program_options()
    app.run(host=options.host, port=int(options.port), debug=options.debug)
