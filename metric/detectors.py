from detector import Detector


class XGBoostDetector(Detector):

    def __init__(self, flag_times='REAL'):
        self.flag_times = flag_times

    def detect(self, url):

        result = url['XGBoost']
        if self.flag_times == 'REAL':
            duration = url['XGBoostTime']
        elif self.flag_times == 'Wang':
            duration = 0.0377  # Wang Dataset
        elif self.flag_times == 'Bahnsen':
            duration = 0.0379  # Bahnsen Dataset
        # print("XGBoost - before: "+"pred:"+str(result)+"_time:"+str(duration))
        return result, duration

    def set_flag_times(self, flag_times='REAL'):
        self.flag_times = flag_times


class ClassificationUsingRNNDetector(Detector):

    def __init__(self, flag_times='REAL'):
        self.flag_times = flag_times

    def detect(self, url):

        result = url['ClassificationUsingRNN']

        if self.flag_times == 'REAL':
            duration = url['ClassificationUsingRNNTime']
        elif self.flag_times == 'Wang':
            duration = 0.0183  # Wang Dataset
        elif self.flag_times == 'Bahnsen':
            duration = 0.0398  # Bahnsen Dataset
        # print("ClassificationUsingRNN - before: "+"pred:"+str(result)+"_time:"+str(duration))
        return result, duration

    def set_flag_times(self, flag_times='REAL'):
        self.flag_times = flag_times


class FFNNDetector(Detector):

    def __init__(self, flag_times='REAL'):
        self.flag_times = flag_times

    def detect(self, url):

        result = url['FFNN']
        if self.flag_times == 'REAL':
            duration = url['FFNNTime']
        elif self.flag_times == 'Wang':
            duration = 30.9334  # Wang Dataset
        elif self.flag_times == 'Bahnsen':
            duration = 18.6656  # Bahnsen Dataset
        # print("FFNN - before: "+"pred:"+str(result)+"_time:"+str(duration))
        return result, duration

    def set_flag_times(self, flag_times='REAL'):
        self.flag_times = flag_times


class CNNDetector(Detector):

    def __init__(self, flag_times='REAL'):
        self.flag_times = flag_times

    def detect(self, url):

        result = url['CnnModel']

        if self.flag_times == 'REAL':
            duration = url['CnnModelTime']
        elif self.flag_times == 'Wang':
            duration = 0.0024  # Wang Dataset
        elif self.flag_times == 'Bahnsen':
            duration = 0.0025  # Bahnsen Dataset
        # print("CNN - before: "+"pred:"+str(result)+"_time:"+str(duration))
        return result, duration

    def set_flag_times(self, flag_times='REAL'):
        self.flag_times = flag_times


class PDRCNNDetector(Detector):

    def __init__(self, flag_times='REAL'):
        self.flag_times = flag_times

    def detect(self, url):

        result = url['PDRCNN']

        if self.flag_times == 'REAL':
            duration = url['PDRCNNTime']
        elif self.flag_times == 'Wang':
            duration = 0.0421  # Wang Dataset
        elif self.flag_times == 'Bahnsen':
            duration = 0.0706  # Bahnsen Dataset
        # print("PDRCNN - before: "+"pred:"+str(result)+"_time:"+str(duration))
        return result, duration

    def set_flag_times(self, flag_times='REAL'):
        self.flag_times = flag_times
