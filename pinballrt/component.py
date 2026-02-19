
class Component():

    def __init__(self, name, params):
        """General class for model compoents, used for tracking parameters

        Args:
            name (str): Name of component
            params (dict[str, dict]): dictionary of parameters, typically in the format:
                param_name: value, range, fixed, units
        """
        
        self.name = name
        self.params = params

        for param_name in params:
             self._set_attribute(param_name)

    def _set_attribute(self, param_name):
        """Create a class attrbute based on the parameter value

        Args:
            param_name (str): name of parameter
        """
        
        atr_name = str(param_name)
        atr_value = self.params[param_name]["value"]
        units = self.params[param_name]["units"]

        if "Log" in param_name:
            atr_name = atr_name.replace("Log", "")
            atr_value = 10**atr_value

        if units:
            atr_value *= units
        
        setattr(self, atr_name, atr_value)

    def set_param(self, param_name=None, value=None, range=None, fixed=None, units=None):
        """Set the attributes of a single parameter

        Args:
            param_name (str, optional): Name of parameter. Defaults to None.
            value (float, optional): Value of the parameter. Defaults to None.
            range (tuple, optional): Range of possible values of the parameter. Defaults to None.
            fixed (bool, optional): Determines if the parameter is fixed or can vary. Defaults to None.
            units (astropy.units.Units, optional): Units of the parameter, if applicable. Defaults to None.
        """
        if value:
            self.params[param_name]["value"] = value
        if range:
            self.params[param_name]["range"] = range
        if fixed is not None:
            self.params[param_name]["fixed"] = fixed
        if units:
            self.params[param_name]["units"] = units

        self._set_attribute(param_name)
        
    def set_params_from_dict(self, param_dict):
            for param_name, param in param_dict.items():
                    self.params[param_name] = param
                    self._set_attribute(param_name)

            