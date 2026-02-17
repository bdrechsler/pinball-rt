
class Component():

    def __init__(self, name, params):
        
        self.name = name
        self.params = params

        for param_name in params:
             self._set_attribute(param_name)

    def _set_attribute(self, param_name):
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

            