import datetime

def timestamp():
    """ Obtains the current timestamp in a human-readable way """

    timestamp = (
        str(datetime.datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    )

    return timestamp
