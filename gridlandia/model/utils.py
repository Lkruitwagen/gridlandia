import numpy as np

def _sigmoid(x):
    return np.exp(x) / (1+np.exp(x))

def D_sigmoid(x):
    return _sigmoid(x)*(1-_sigmoid(x))

def growth_rate_bell(max_rate, min_rate, peak_year, half_decay):
    """ return a function which can calculate a rate in a given year with a curve with the input params"""
    
    def gr_fn(x):
        return min_rate + D_sigmoid((x-peak_year)*1.76274/(half_decay-peak_year))/0.25*(max_rate-min_rate)
    
    return gr_fn


def fill_product(d, list_a, list_b, list_c, default_val):
    for el_a in list_a:
        d[el_a] = {}
        for el_b in list_b:
            if list_c is None:
                d[el_a][el_b] = default_val
            else:
                d[el_a][el_b] = {}
                for el_c in list_c:
                    d[el_a][el_b][el_c] = default_val
    return d