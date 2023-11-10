import numpy
import pandas


def reward_function(VLt: numpy.array, CD4t: numpy.array) -> numpy.array:
    VLt_inclusion_element_mask = VLt > 0
    return VLt_inclusion_element_mask * numpy.nan_to_num(
        -0.7 * numpy.log(VLt) + 0.6 * numpy.log(CD4t), 0) + (~VLt_inclusion_element_mask) * (5 + 0.6 * numpy.log(CD4t))


def append_previous_month_info(subdf):
    subdf = subdf.sort_values(by=['Timepoints'])
    shifted_subdf = subdf.copy().rename({
        e: f'prev_month_{e}' for e in subdf if e not in ['Timepoints'] and 'Gender' not in e and 'Ethnic' not in e and 'Comp. NNTRI' not in e
    }, axis=1).shift(1).loc[:, [f'prev_month_{e}' for e in subdf if e not in ['Timepoints'] and 'Gender' not in e and 'Ethnic' not in e]].fillna(-1.0)
    subdf = pandas.concat([subdf, shifted_subdf], axis=1)
    return subdf