

from opfunu.cec.cec2005.root import Root
from numpy import sum


class Model(Root):
    def __init__(self, f_name="Shifted Sphere Function", f_shift_data_file="data_sphere", f_ext='.txt', f_bias=-450):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        result = sum((solution - shift_data)**2) + self.f_bias
        return result











