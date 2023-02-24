import neptune.new as neptune


class NeptuneLogger:
    """
    Class that uses neptune library to log model behaviour
    """

    def __init__(self, neptune_params):
        """
        @param neptune_params: dict, neptune parameters to be used to initialize the run
        """
        self.neptune_params = neptune_params
        self.run = None

    def start(self):
        """
        Initialize the run
        """
        self.run = neptune.init_run(**self.neptune_params)

    def log_metric(self, name, val):
        """
        Log single value

        @param name: string, name of metric
        @param val: ambigious, value of metric
        """
        self.run[name] = val

    def log_stepmetric(self, name, val, step):
        """
        Log multiple values (with certain step)

        @param name: string, name of metric
        @param val: ambigious, value of metric
        @param step: float, step of logging
        """
        self.run[name].append(value=val, step=step)

    def end(self):
        """
        Terminate the run
        """
        self.run.stop()
