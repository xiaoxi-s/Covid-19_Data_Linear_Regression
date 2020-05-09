import pandas as pd
import numpy as np
import math

time_series = 'time_series_covid19_confirmed_global.csv'
hdi_path = 'hdi.csv'
gdp_per_capita_path = 'gdp_per_capita.csv'
average_of_years_in_school_path = 'mean_of_years_in_school.csv'
literacy_rate_path = 'literacy_rates.csv'
access_to_electricity_path = 'access_to_electricity.csv'
access_to_energy_for_cooking_path = 'energy_for_cooking.csv'

paths = [hdi_path, average_of_years_in_school_path, access_to_electricity_path,
         access_to_energy_for_cooking_path, literacy_rate_path]

countries_data_in_record = {}


def load_confirmed_cases(filename):
    confirmed_cases_data_frame = pd.read_csv(filename)
    return confirmed_cases_data_frame


def preprocess_cases(confirmed_cases_data_frame):
    # Remove the unneeded columns
    confirmed_cases_data_frame.pop('Lat')
    confirmed_cases_data_frame.pop('Long')
    confirmed_cases_data_frame.pop('Province/State')

    cases_across_countries = {}

    # Compute cases in each country by group by Country/Region
    confirmed_cases_accross_countries = confirmed_cases_data_frame.groupby(['Country/Region']).sum()
    for row in confirmed_cases_accross_countries.iterrows():
        # calculate x, y feature vector
        x, y, z, m = calculate_label(row[1].tolist())
        if x is math.nan or y is math.nan or z is math.nan:
            # ignore nan features
            continue
        if row[0] in countries_data_in_record.keys():
            countries_data_in_record[row[0]] += countries_data_in_record[row[0]] + 1
        else:
            countries_data_in_record[row[0]] = 1

        cases_across_countries[row[0]] = (x, y, z, m)

    return cases_across_countries


def load_hdi(filename):
    hdi_data_frame = pd.read_csv(filename)
    return hdi_data_frame


def preprocess_hdi(hdi_data_frame):
    """
    Get HDI in 2018 with country name as the key and HDI as the value
    :param hdi_data_frame: HDI data frame
    :return: HDI
    """
    hdi_by_country = {}

    for row in hdi_data_frame.to_numpy():

        try:  # Only store the value that is valid
            float(row[-2])
        except ValueError:
            continue

        temp_key = str(row[1]).strip()
        hdi_by_country[temp_key] = float(row[-2])
        # keep recording number of metrics each country has
        if temp_key in countries_data_in_record.keys():
            countries_data_in_record[temp_key] += 1
        else:
            countries_data_in_record[temp_key] = 1

    return hdi_by_country


def load_mean_of_years_in_school_path(filename):
    mean_of_years_in_school_data_frame = pd.read_csv(filename)
    return mean_of_years_in_school_data_frame


def preprocess_mean_of_years_in_school(mean_of_years_in_school_data_frame):
    latest_mean_years = mean_of_years_in_school_data_frame.groupby(['Entity']).max()
    average_years_in_school = {}
    for row in latest_mean_years.iterrows():
        temp_list = row[1].tolist()
        if temp_list[1] == 2017:
            # we only consider the records in 2017
            average_years_in_school[row[0]] = temp_list[2]

            # keep recording number of metrics each country has
            if row[0] in countries_data_in_record.keys():
                countries_data_in_record[row[0]] += 1
            else:
                countries_data_in_record[row[0]] = 1

    return average_years_in_school


def load_literacy_rate(filename):
    literacy_rate = pd.read_csv(filename)
    return literacy_rate


def preprocess_literacy_rate(literacy_rate):
    literacy_rate = literacy_rate.groupby(['Entity']).max()
    literacy_rate_in_2015 = {}

    for row in literacy_rate.iterrows():
        temp_list = row[1].tolist()
        if temp_list[1] == 2015:
            # we only consider the records in 2015
            literacy_rate_in_2015[row[0]] = temp_list[2]

            # keep recording number of metrics each country has
            if row[0] in countries_data_in_record.keys():
                countries_data_in_record[row[0]] += 1
            else:
                countries_data_in_record[row[0]] = 1

    return literacy_rate_in_2015


def load_process_access_to_electricity(filename):
    access_to_electricity = pd.read_csv(filename)
    access_to_electricity = access_to_electricity.groupby('Entity').max()
    share_of_people_access_to_electricity = {}

    for row in access_to_electricity.iterrows():
        temp_list = row[1].tolist()
        if temp_list[1] == 2016:
            # we only consider the records in 2017
            share_of_people_access_to_electricity[row[0]] = temp_list[2]

            # keep recording number of metrics each country has
            if row[0] in countries_data_in_record.keys():
                countries_data_in_record[row[0]] += 1
            else:
                countries_data_in_record[row[0]] = 1

    return share_of_people_access_to_electricity


def load_process_access_to_energy_for_cooking(filename):
    access_to_energy_for_cooking = pd.read_csv(filename)
    access_to_energy_for_cooking = access_to_energy_for_cooking.groupby(['Entity']).max()

    share_of_people_access_to_energy_for_cooking = {}

    for row in access_to_energy_for_cooking.iterrows():
        temp_list = row[1].tolist()
        if temp_list[1] == 2016:
            # we only consider the records in 2017
            share_of_people_access_to_energy_for_cooking[row[0]] = temp_list[2]

            # keep recording number of metrics each country has
            if row[0] in countries_data_in_record.keys():
                countries_data_in_record[row[0]] += 1
            else:
                countries_data_in_record[row[0]] = 1

    return share_of_people_access_to_energy_for_cooking


def load_and_preprocess_data():
    # cases
    confirmed_cases_over_time = load_confirmed_cases(time_series)

    # case vector
    case_feature_vector_by_countries = preprocess_cases(confirmed_cases_over_time)

    # hdi
    hdi_by_countries = preprocess_hdi(load_hdi(hdi_path))

    # average years in school
    average_of_years_in_school_in_2017 = \
        preprocess_mean_of_years_in_school(load_mean_of_years_in_school_path(average_of_years_in_school_path))

    # literacy rate
    literacy_rate = load_literacy_rate(literacy_rate_path)
    literacy_rate_in_2015 = preprocess_literacy_rate(literacy_rate)

    # energy access
    access_to_electricity = load_process_access_to_electricity(access_to_electricity_path)

    share_of_people_access_to_energy_for_cooking = \
        load_process_access_to_energy_for_cooking(access_to_energy_for_cooking_path)

    max_number_of_records = 0
    for key in countries_data_in_record.keys():
        if countries_data_in_record[key] > max_number_of_records:
            max_number_of_records = countries_data_in_record[key]

    for key in countries_data_in_record.keys():
        if countries_data_in_record[key] != max_number_of_records:
            # remove the countries with incomplete data
            if key in case_feature_vector_by_countries.keys():
                case_feature_vector_by_countries.pop(key)

            if key in hdi_by_countries.keys():
                hdi_by_countries.pop(key)
            if key in average_of_years_in_school_in_2017.keys():
                average_of_years_in_school_in_2017.pop(key)
            if key in access_to_electricity.keys():
                access_to_electricity.pop(key)
            if key in share_of_people_access_to_energy_for_cooking.keys():
                share_of_people_access_to_energy_for_cooking.pop(key)
            if key in literacy_rate_in_2015.keys():
                literacy_rate_in_2015.pop(key)

    keys = list(case_feature_vector_by_countries.keys())
    feature_matrix = np.full((len(keys), 5), 0, dtype=float)
    label = np.full((len(keys), 1), 0, dtype=float)

    i = 0
    for k in keys:
        feature_matrix[i][0] = hdi_by_countries[k]
        feature_matrix[i][1] = average_of_years_in_school_in_2017[k]
        feature_matrix[i][2] = access_to_electricity[k]
        feature_matrix[i][3] = share_of_people_access_to_energy_for_cooking[k]
        feature_matrix[i][4] = literacy_rate_in_2015[k]
        label[i][0] = case_feature_vector_by_countries[k][0] + case_feature_vector_by_countries[k][1] + \
                      case_feature_vector_by_countries[k][2] + case_feature_vector_by_countries[k][3]
        i += 1

    return feature_matrix, label, keys, ['hdi', 'average year in school', 'access to electricity',
                                         'access to energy for cooking', 'literacy rate']


def calculate_label(time_series):
    """
    Calculate the feature vector based on the time series
    :param time_series: the given input
    :return: (x, y) as specified in the document
    """

    # remove the country and region title and convert to int
    today_num = time_series[-1]
    number_of_days = len(time_series)

    if today_num == 0:
        return tuple((math.nan, math.nan))

    x = -1
    y = -1
    z = -1
    m = -1
    for i in range(number_of_days - 1, -1, -1):
        if time_series[i] <= today_num / 10 and x == -1:
            x = number_of_days - i - 1
        if time_series[i] <= today_num / 100 and y == -1:
            y = number_of_days - x - i - 1
        if time_series[i] <= today_num / 1000 and z == -1:
            z = number_of_days - x - y - i - 1
        if time_series[i] < today_num / 10000 and m == -1:
            m = number_of_days - x - y - z - i - 1
            return tuple((x, y, z, m))

    if x == -1:
        x = math.nan
    if y == -1:
        y = math.nan
    if z == -1:
        y = math.nan
    if m == -1:
        y = math.nan
    return tuple((x, y, z, m))
