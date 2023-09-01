from abc import abstractmethod


class Detector:
    @abstractmethod
    def detect(self, fd):
        """
        Processes the file and return a result and costs - the probability
        for the file being malicious (ranged from 0 to 1) and costs of any
        type (time, CPU, money, etc.). If the detector does not return a
        probability, it can return boolean result: 0 or 1.

        The process can be eith a classic ML process (feature extraction and
        prediction), or scanning the file through an existing product and
        processing the result.
        """
        result = None
        costs = None
        return result, costs


class MockTimedDetector(Detector):
    def __init__(self):
        pass

    def detect(self, fd, flag=None):
        start = datetime.now()
        time.sleep(random.uniform(0.5, 2))
        result = random.uniform(0, 1)
        end = datetime.now()
        duration = (end - start).total_seconds()
        return result, duration
