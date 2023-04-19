import os, re, argparse, yaml
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def get_args():
    # Store all parameters for easy retrieval
    parser = argparse.ArgumentParser(description = 'fixed&flexible')
    parser.add_argument('--params_filename',
                        type=str,
                        default='params.yaml',
                        help = 'Loads model parameters')
    args = parser.parse_args()
    config = yaml.load(open(args.params_filename), Loader=yaml.FullLoader)
    for k,v in config.items():
        args.__dict__[k] = v
    return args

def get_nodal_inputs(args):
    nodal_load_input   = pd.read_excel(f'{args.data_dir}/nodes_pue_irri_inputs.xlsx')
    return nodal_load_input

def get_fixed_load(args):
    files_path = f'{args.data_dir}/{args.fixed_load_dir}'
    system_customers = os.listdir(files_path)
    system_customers = [f for f in system_customers if f.endswith('.csv')].sort()
    nodal_fixed_load = np.zeros((args.num_hour_fixed_load, len(system_customers)))
    for i in range(len(system_customers)):
        nodal_fixed_load[:, i] = pd.read_csv(os.path.join(files_path, system_customers[i]))
    return nodal_fixed_load


def annualization_rate(i, years):
    return (i*(1+i)**years)/((1+i)**years-1)

def get_cap_cost(args, years):
    # Annualize capacity costs for model
    annualization_solar   = annualization_rate(args.i_rate, args.annualize_years_solar)
    annualization_battery_la = annualization_rate(args.i_rate, args.annualize_years_battery_la)
    annualization_battery_li = annualization_rate(args.i_rate, args.annualize_years_battery_li)
    annualization_battery_inverter = annualization_rate(args.i_rate, args.annualize_years_battery_inverter)
    annualization_diesel  = annualization_rate(args.i_rate, args.annualize_years_diesel)
    # only solar will use piecewise capital cost
    solar_cap_cost = [years * annualization_solar * float(solar_cost) for solar_cost in args.solar_pw_cost_kw]
    solar_single_cap_cost = years * annualization_solar * float(args.solar_single_cost_kw)
    battery_la_cap_cost_kwh  = years * annualization_battery_la * float(args.battery_la_cost_kwh)
    battery_li_cap_cost_kwh  = years * annualization_battery_li * float(args.battery_li_cost_kwh)
    battery_inverter_cap_cost_kw  = years * annualization_battery_inverter * float(args.battery_inverter_cost_kw)
    diesel_cap_cost_kw    = years * annualization_diesel  * float(args.diesel_cap_cost_kw) * args.reserve_req
    return solar_cap_cost, solar_single_cap_cost, battery_la_cap_cost_kwh, battery_li_cap_cost_kwh, \
           battery_inverter_cap_cost_kw, diesel_cap_cost_kw

def load_timeseries(args, mod_level):
    # Load solar & load time series, all region use the same
    if mod_level == "cap":
        solar_po_hourly   = np.array(pd.read_csv(f'{args.data_dir}/solar_po_2019_33d.csv', index_col=0))[:,0]
        rain_rate_daily_mm_m2 = np.array(pd.read_csv(f'{args.data_dir}/rain_rate_mm_2014_2015_33d.csv', index_col=0))[:,0]
    if mod_level == "ope":
        solar_po_hourly   = np.array(pd.read_csv(f'{args.data_dir}/solar_po_2019_3m.csv', index_col=0))[:,0]
        rain_rate_daily_mm_m2 = np.array(pd.read_csv(f'{args.data_dir}/rain_rate_mm_2014_2015_3m.csv', index_col=0))[:,0]
    fix_load_hourly_kw = np.array(pd.read_csv(f'{args.data_dir}/domestic_load_kw.csv', index_col=0))[:,0]
    return fix_load_hourly_kw, solar_po_hourly, rain_rate_daily_mm_m2


def get_connection_info(args):
    # transmission connection is pre-solved with TLND model
    f = open(os.path.join(args.data_dir,'tlnd_modelOutput.txt'),'r')
    tx_f = f.read()
    lv_length = float(re.search(r'LVLength:(.*?)\n',tx_f).group(1))
    mv_length = float(re.search(r'MVLength:(.*?)\n',tx_f).group(1))
    tx_num    = float(re.search(r'Num Transformers:(.*?)\n',tx_f).group(1))
    meter_num = float(re.search(r'NumStructures:(.*?)\n',tx_f).group(1))
    print("connection info", lv_length, mv_length, tx_num)
    return lv_length, mv_length, tx_num, meter_num



'''
### There are some previously used functions, I put them below in case we will revisit it  ###
##------------------------------------------------------------------------------------------##

def read_irrigation_area(args):
    irrigation_area_m2 = pd.read_csv(os.path.join(args.data_dir, 'region_{}'.format(str(args.region_no)), 'pts_area.csv'))["AreaSqM"] * args.irrgation_area_ratio
    num_regions = len(irrigation_area_m2)
    return num_regions, irrigation_area_m2

def get_nodes_area(args, sce_sf_area_m2):
    # base on the config to get the nodes number and area
    if args.config == 0:
        num_nodes = 1
        irrigation_area_m2 = [0]
    elif args.config == 0.5:
        num_nodes, irrigation_area_m2 = read_irrigation_area(args)
        num_nodes = 1
        irrigation_area_m2 = [float(np.sum(irrigation_area_m2))]
    elif args.config == 1:
        num_nodes = 1
        irrigation_area_m2 = [sce_sf_area_m2]
    elif args.config == 2:
        num_nodes, irrigation_area_m2 = read_irrigation_area(args)
    elif args.config == 3 or args.config == 4:
        num_nodes, irrigation_area_m2 = read_irrigation_area(args)
        num_nodes = 1
        irrigation_area_m2 = [float(np.sum(irrigation_area_m2))]
    return num_nodes, irrigation_area_m2

def dry_season_solar_ts(args, solar_po_hourly):
    solar_daily = np.add.reduceat(solar_po_hourly, np.arange(0, len(solar_po_hourly), 24))
    solar_5_day = pd.DataFrame({'5_day_solar_sum':[np.sum(solar_daily[d:(d+5)]) for d in range(len(solar_daily)-4)]})
    solar_5_day_dry = solar_5_day[args.second_season_start:(args.second_season_end-3)]
    # get the 10 percentile solar 5-day sum in the dry season
    dry_season_solar_day = solar_5_day_dry.sort_values('5_day_solar_sum').index[int(np.round(len(solar_5_day_dry)*float(args.solar_ds_perc))-1)]
    print("dry season solar day", dry_season_solar_day, "sum",solar_5_day.iloc[dry_season_solar_day])
    return dry_season_solar_day
def find_extreme_solar_period(args):
    # find the least 5 days solar potential during the irrigation seasons
    T = args.num_hours
    solar_pot_hourly   = np.array(pd.read_excel(os.path.join(args.data_dir, 'solar_pot.xlsx'),
                                                index_col=None))[0:T,0]
    extreme_solar_start_day = 0
    for day in list(range(args.first_season_start, args.first_season_end - args.water_account_days + 2)) + \
               list(range(args.second_season_start, args.second_season_end - args.water_account_days + 2)):
        if np.sum(solar_pot_hourly[day*24:(day+5)*24]) < \
                np.sum(solar_pot_hourly[extreme_solar_start_day*24:(extreme_solar_start_day+5)*24]):
            extreme_solar_start_day = day
    return extreme_solar_start_day
def find_avg_solar(args):
    # find the average solar potnetial
    T = args.num_hours
    solar_pot_hourly   = np.array(pd.read_excel(os.path.join(args.data_dir, 'solar_pot.xlsx'),
                                                index_col=None))[0:T,0]
    solar_pot_hourly_daily = solar_pot_hourly.reshape(int(T/24), 24)
    avg_solar_po_day = np.average(solar_pot_hourly_daily, axis=0)
    avg_solar_po = np.tile(avg_solar_po_day, args.water_account_days)
    return avg_solar_po
def read_tx_distance(args):
    tx_matrix_dist_m = pd.read_csv(os.path.join(args.data_dir, 'tx_matrix_dist_m.csv'), header=0, index_col=0)
    return tx_matrix_dist_m
def get_tx_tuples(args):
    cap_ann = annualization_rate(args.i_rate, args.annualize_years_trans)
    tx_matrix_dist_m = pd.read_csv(os.path.join(args.data_dir, 'tx_matrix_dist_m.csv'),header=0, index_col=0)
    tx_tuple_list = []
    # tuple list in the order: (pt1, pt2), distance, cost kw, loss
    for i in range(len(tx_matrix_dist_m)):
        for j in range(len(tx_matrix_dist_m.columns)):
            if tx_matrix_dist_m.iloc[i, j] > 0:
                tx_tuple_list.append(((i + 1, j + 1),
                                      tx_matrix_dist_m.iloc[i, j],
                                      args.num_year * cap_ann * args.trans_cost_kw_m *
                                      tx_matrix_dist_m.iloc[i, j],
                                      tx_matrix_dist_m.iloc[i, j] * float(args.trans_loss)))
    return tx_tuple_list

'''