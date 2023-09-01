class Config:
    """
    Configuration class for the project.

    Attributes:
        general (dict): General configuration settings.
        data (dict): Data-related configuration settings.
        model (dict): Model-related configuration settings.
        detectors (dict): Detectors-related configuration settings.
    """

    def __init__(self):
        """
        Initializes a new instance of the Config class with default configuration settings.
        """
        self._general = {
            'debug_level_logs': False,

        }
        self._data = {
            'default_dataset_path': '../data/',
            'default_dataset': 'Bahnsen', # 'Wang'
            'dataset_postfix': '.csv',
            'experiment_number': 2,
            'test_set_split': 0.15,
            'valid_set_split': 0.1,
            'benign_label': 5,
            'phishing_label': 6,            
            'batch_metrics_size': 100,
        }
        self._model = {

            'mode': 'test',
            'model_dir_path': "pretrained_weights",
            'load_model_flag': True,
            'act_deterministically_flag': False,
            'num_of_epochs': 1,
            'random_seed': 123,
            'learning_rate': 0.00001, 
            'eps': 0.00000001,
            'replay_buffer_size': 5000,
            'save_output_filename': "metric_cetra_statistics.csv",
            'metrics_bonus_function': lambda r: 1.2 * r,
            'user_defined_metrics_dict': {'TPR': {'min': 0.95, 'extra': 0.97}, 'FPR': {'min': -1, 'extra': 0},
                             'TNR': {'min': -1, 'extra': 0}, 'FNR': {'min': -1,'extra': 0}}  # , 'TNR': { 'min':0.95,'extra':0.97}, 'FNR': { 'min':0.05,'extra':0.02}
         
        }
        self._detectors = {
            'flag_times': "REAL", # real times - 'REAL', Average_Wang_Dataset - 'Wang', Average_Bahnsen_Dataset - 'Bahnsen'
            'illegal_reward': -10000
        }
        
    @property
    def general(self) -> dict:
        """
        Returns the general configuration settings.

        Returns:
            dict: The general configuration settings.
        """
        return self._general

    @general.setter
    def general(self, value: dict) -> None:
        """
        Sets the general configuration settings.

        Parameters:
            value (dict): The general configuration settings to set.
        """
        self._general = value

    @property
    def data(self) -> dict:
        """
        Returns the data configuration settings.

        Returns:
            dict: The data configuration settings.
        """
        return self._data

    @data.setter
    def data(self, value: dict) -> None:
        """
        Sets the data configuration settings.

        Parameters:
            value (dict): The data configuration settings to set.
        """
        self._data = value

    @property
    def model(self) -> dict:
        """
        Returns the model configuration settings.

        Returns:
            dict: The model configuration settings.
        """
        return self._model

    @model.setter
    def model(self, value: dict) -> None:
        """
        Sets the model configuration settings.

        Parameters:
            value (dict): The model configuration settings to set.
        """
        self._model = value

    @property
    def detectors(self) -> dict:
        """
        Returns the detectors configuration settings.

        Returns:
            dict: The detectors configuration settings.
        """
        return self._detectors

    @detectors.setter
    def detectors(self, value: dict) -> None:
        """
        Sets the detectors configuration settings.

        Parameters:
            value (dict): The detectors configuration settings to set.
        """
        self._detectors = value