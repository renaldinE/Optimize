﻿from random import sample
from tkinter import messagebox
import numpy as np
import pandas as pd
import numpy_financial as npf
from scipy.optimize import minimize, Bounds
from math import log10, floor
import os 
import time
import ctypes
import warnings
import multiprocessing as mp
import traceback

warnings.filterwarnings('error',module=r'.*Optimize_V.*')

# Naming Rules:
# ending in _nrgs: vector in nrgs below
# ending in _hourly: vector/matrix including 4 years of hourly values

#Energy is in MWh
#Power is in MW.
#Cost is in M$
#CO2 is MTonne

#Globals

dirname, filename = os.path.split(os.path.abspath(__file__)) 
os.chdir(dirname + '\\..')

# had to add to deletechars, they got inserted at the beginning of the first genfromtext entry.
sample_years   = 4
sample_points  = int(sample_years * 365.25 * 24)
delete_chars   = " !#$%&'()*+, -./:;<=>?@[\\]^{|}~﻿ï»¿"

nrgs           = np.array(['Solar', 'Wind', 'Nuclear', 'Gas', 'Coal', 'Storage'])

others         = np.array(['Hydro', 'Oil', 'Other'])

# Output Matrix Columns
output_header = pd.Series(['Year', 'CO2_Price', 'Outage', 'Total_MW', 'Total_MWh', 'Energy_Cost', 'CO2_Cost', 'Demand'])
param_order   = pd.Series(['MW','MWh', 'Supply', 'Cost', 'CO2_Cost', 'CO2_MTon', 'MW_Cost', 'MWh_Cost', 'Knob', 'Max_Knob', 'Lifetime'])
tweaked_globals_order = pd.Series(['CO2_Price', 'Demand', 'Interest'])
tweaked_nrgs_order    = pd.Series(['Capital','Fixed', 'perMW', 'perMWh', 'Max_PCT', 'Lifetime', 'CO2_gen'])

#************ Debug Options ************
# True = normal run - go through minimize()
normal_run = True

# True = do not run parallel processes
kill_parallel = False

# run one set of knobs/Year - no minimize()
#  Also set normal_run to False
debug_one_case = False

# True = print minimize results
debug_minimizer = False

# Print out on each step of the minimizer 
debug_step_minimizer = False

# Print out numbers that should not change in each year
debug_unexpected_change = False

# Printout cost info
debug_cost = False

# Printout cost info less verbose
debug_cost_sum = False

# Printout update data once/year
debug_update = False

# Printout fig_gas data once/year
debug_fig_gas = False

# Printout need calculation once/year
debug_need = False

def msgbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def df_int_to_float(df):
    list = df.select_dtypes(include=['int']).columns.tolist()
    df[list] = df[list].astype(float) 
    return df

def round_significant(x,sig):
    x = round(x, -int(floor(log10(abs(x))))+sig-1)
    return x
# Save debug matrix
def save_debug(debug_matrix, file_name):
   file_path = './python/mailbox/outbox/' + file_name
   if os.path.exists(file_path):
       os.remove(file_path)
   debug_matrix.to_csv(file_path)
    
    
# Get price, CO2 generated, etc for each nrg
def get_specs_nrgs():
    specs_nrgs = pd.read_csv('./csv/specs.csv',
                         header=0, 
                         skiprows=1, 
                         index_col=0)
   
    specs_nrgs = df_int_to_float(specs_nrgs)
          
    return specs_nrgs

# Get parameters set in overlord by user
def get_inbox():
    inbox = pd.read_csv('./Python/mailbox/inbox.csv', 
                         header=0, 
                         index_col=0)
    inbox = df_int_to_float(inbox)
    
    return inbox

# Get list of regions
def get_all_regions():
    regions_temp = np.genfromtxt('./csv/Regions.csv', 
                              delimiter=',',
                              dtype=('U5, U20'), 
                              names=True,
                              deletechars=delete_chars)  
    return regions_temp['Abbr']
    
#Get hourly data
def get_eia_data(region):
    latest = open('./csv/Eia_Hourly/Latest.txt','r')
    csv_date = latest.read()
    latest.close()
    eia_filename = f'{region}_master_{csv_date}.csv'
    csv_path = f'./csv/Eia_Hourly/latest/{eia_filename}'

    hourly_nrgs = pd.read_csv(csv_path,
                         header=0, 
                         skiprows=0)
    storage = pd.DataFrame(0,index=hourly_nrgs.index, columns=['Storage'])
    others  = pd.DataFrame(0,index=hourly_nrgs.index, columns=['Others'])
    others['Others']  = hourly_nrgs['Hydro'] + hourly_nrgs['Oil'] + hourly_nrgs['Other']
    hourly_nrgs = pd.concat([hourly_nrgs,storage,others], axis=1)
    hourly_nrgs = df_int_to_float(hourly_nrgs)    
    return hourly_nrgs

def init_output_matrix():
    output_header_loc = output_header    
    for nrg in nrgs:
        for param in param_order:
            output_header_loc = pd.concat([output_header_loc,pd.Series([nrg + '_' + param])], axis=0, ignore_index=True)                 
    output_matrix = pd.DataFrame(columns=output_header_loc, dtype=float)
    return output_matrix

# Initial values for year zero - Spec Numbers
def init_tweaks(specs_nrgs,inbox):
    tweaked_nrgs = pd.DataFrame(1,columns=nrgs, index=tweaked_nrgs_order, dtype=float) 
    for nrg in nrgs: 
        tweaked_nrgs.at['Capital', nrg] = specs_nrgs.at['Capital', nrg]
        tweaked_nrgs.at['Fixed', nrg]   = specs_nrgs.at['Fixed', nrg]   
        tweaked_nrgs.at['perMWh', nrg]  = specs_nrgs.at['Variable', nrg]
        tweaked_nrgs.at['perMW', nrg]   = specs_nrgs.at['Fixed', nrg] + \
                     (-4 * npf.pmt(inbox['Interest'].loc['Initial']/4, specs_nrgs.at['Lifetime', nrg]*4,specs_nrgs.at['Capital', nrg]))
        tweaked_nrgs.at['Lifetime', nrg] = specs_nrgs.at['Lifetime', nrg]
        tweaked_nrgs.at['Max_PCT', nrg]  = specs_nrgs.at['Max_PCT', nrg]
        tweaked_nrgs.at['CO2_gen', nrg]  = specs_nrgs.at['CO2_gen', nrg]
        
    tweaked_globals = pd.Series(0,index=tweaked_globals_order, dtype=float)
    tweaked_globals['CO2_Price'] = 0
    tweaked_globals['Demand']    = 1
    tweaked_globals['Interest']  = 0
    return tweaked_globals, tweaked_nrgs

# Figure next year's info
def fig_tweaks(    
                tweaked_nrgs,
                tweaked_globals,
                inbox,
                year):

    if year == 1:
        loc_ = 'Initial'
        tweaked_globals['CO2_Price'] = inbox['CO2_Price'].loc[loc_]
        tweaked_globals['Demand']    = inbox['Demand'].loc[loc_] 
        tweaked_globals['Interest']  = inbox['Interest'].loc[loc_]
        
    else:
        loc_ = 'Yearly'
        tweaked_globals['CO2_Price'] += inbox['CO2_Price'].loc[loc_]
        tweaked_globals['Demand']    *= inbox['Demand'].loc[loc_] 
        tweaked_globals['Interest']  *= inbox['Interest'].loc[loc_] 
    
    for nrg in nrgs: 
        tweaked_nrgs.at['Capital', nrg]  *= inbox[nrg + '_Capital'].loc[loc_]
        tweaked_nrgs.at['Fixed', nrg]    *= inbox[nrg + '_Fixed'].loc[loc_]
        tweaked_nrgs.at['Lifetime', nrg] *= inbox[nrg + '_Lifetime'].loc[loc_]
        tweaked_nrgs.at['Max_PCT', nrg]  *= inbox[nrg + '_Max_PCT'].loc[loc_]
        tweaked_nrgs.at['perMWh', nrg]   *= inbox[nrg + '_Variable'].loc[loc_]
        
        tweaked_nrgs.at['perMW', nrg]     = tweaked_nrgs.at['Fixed', nrg] + \
                         (-4 * npf.pmt(tweaked_globals['Interest']/4, tweaked_nrgs.at['Lifetime', nrg]*4,tweaked_nrgs.at['Capital', nrg]))
                         # Note that this figures a quarterly payoff, 4 payments per year  
    return tweaked_globals, tweaked_nrgs

# Figure loss due to lifetime of plant
def fig_decadence(hourly_nrgs, MWh_nrgs, MW_nrgs, tweaked_nrgs):
    for nrg in nrgs:
        hourly_nrgs[nrg]   *= 1 - (1/tweaked_nrgs.at['Lifetime', nrg])
        MW_nrgs[nrg]       *= 1 - (1/tweaked_nrgs.at['Lifetime', nrg])
        MWh_nrgs[nrg]      *= 1 - (1/tweaked_nrgs.at['Lifetime', nrg])
    return hourly_nrgs, MWh_nrgs, MW_nrgs
    
# Gas fills any leftover need.  If not enough, storage.  If not enough, outage (VERY expensive)
def fig_gas_and_storage(needed_hourly, gas_max, storage_max, stored, MWh_nrgs, debug_temp):
    gas_used     = 0
    storage_used = 0
    outage_MWh   = 0
    excess   = 0
    if debug_temp:
        debug_matrix = pd.DataFrame(columns=['Needed', 'Gas', 'Excess', 'Path', 'nrg','Start_MWh_nrgs', 'End_MWh_nrgs','Excess_total'])
        count = 0
        for nrg in nrgs:
            debug_matrix.at[count,'nrg'] = nrg
            debug_matrix.at[count,'Start_MWh_nrgs'] = MWh_nrgs[nrg]
            count += 1
            
    count = 0
    old_gas = 0
    old_excess = 0
    for hour_of_need in needed_hourly:
        #Already have too much NRG
        if(hour_of_need < 0):
            path = "Excess"
            if (-hour_of_need > storage_max - stored):
                stored  = storage_max
                excess += -hour_of_need - (storage_max - stored)
            else:
                stored += -hour_of_need
                
        # if enough gas for everybody
        elif (hour_of_need + (storage_max - stored) < gas_max):
            path = 'Full_Charge'
            gas_used += hour_of_need + (storage_max - stored)
            stored    = storage_max
        # Enough gas to avoid outage + a little left over for stored
        elif (hour_of_need < gas_max):
            path = "Some_Charge"
            gas_used += gas_max
            stored   += gas_max - hour_of_need
        # Enough gas + storage to meet need
        elif (hour_of_need < gas_max + stored):
            path = 'Use_Store'
            gas_used     += gas_max
            stored       -= hour_of_need - gas_max
            storage_used += hour_of_need - gas_max
        # Not enough to meet need
        else:
            path = 'UhOh'
            outage_MWh   += hour_of_need - gas_max - stored                    
            gas_used     += gas_max
            storage_used += stored
            stored        = 0
        if(debug_temp):
            debug_matrix.at[count, 'Needed'] = hour_of_need
            debug_matrix.at[count, 'Gas']    = gas_used - old_gas
            debug_matrix.at[count, 'Excess'] = excess - old_excess
            debug_matrix.at[count, 'Path']   = path
            count += 1
            old_gas = gas_used
            old_excess = excess
            
    if(debug_temp):
        count = 0
        for nrg in nrgs:
            debug_matrix.at[count,'End_MWh_nrgs'] = MWh_nrgs[nrg]
            count += 1

        save_debug(debug_matrix, 'debug_fig_gas.csv' )   
    
    MWh_nrgs['Gas']     = gas_used / sample_years
    MWh_nrgs['Storage'] = storage_used / sample_years
    excess_MWh          = excess   / sample_years
    
    return MWh_nrgs, outage_MWh, stored, excess_MWh

def fig_excess(MWh_nrgs, excess_MWh, debug_temp):
    excess_nrgs = pd.Series(0,index=nrgs, dtype=float)    
    total_curtailable = 0
    for nrg in ['Solar', 'Wind', 'Nuclear', 'Coal']:
        total_curtailable += MWh_nrgs[nrg]
    
    if(debug_temp):
        debug_matrix = pd.DataFrame(columns=['nrg','Start_MWh_nrgs', 'End_MWh_nrgs','Excess', 'Curtailable'])
        debug_matrix.at[0,'Curtailable'] = total_curtailable
        count = 0
   
    for nrg in ['Solar', 'Wind', 'Nuclear', 'Coal']:
        excess_nrgs[nrg] = excess_MWh * MWh_nrgs[nrg] / total_curtailable
        if(debug_temp):
            debug_matrix.at[count,'nrg']            = nrg
            debug_matrix.at[count,'Start_MWh_nrgs'] = MWh_nrgs[nrg]
            debug_matrix.at[count,'Excess']         = excess_nrgs[nrg]
        MWh_nrgs[nrg]  -= excess_nrgs[nrg]
        if(debug_temp):
            debug_matrix.at[count,'End_MWh_nrgs'] = MWh_nrgs[nrg]
            count += 1
    
    if(debug_temp):
        save_debug(debug_matrix, 'debug_fig_excess.csv')

    return MWh_nrgs, excess_nrgs
    

# add another year to the output matrix
def add_output_year(
                  MWh_nrgs,            
                  MW_nrgs,
                  excess_nrgs,
                  tweaked_globals,
                  tweaked_nrgs,
                  expensive,
                  outage_MWh,
                  output_matrix,
                  year,
                  knobs_nrgs,
                  max_add_nrgs):
            
    output_matrix.at[year, 'Year']      = year
    output_matrix.at[year, 'CO2_Price'] = tweaked_globals['CO2_Price']
    output_matrix.at[year, 'Outage']    = outage_MWh
    output_matrix.at[year, 'Demand']    = tweaked_globals['Demand']
    
    energy_cost = 0
    total_CO2   = 0
    total_MW    = 0
    total_MWh   = 0
    
    for nrg in nrgs:
        output_matrix.at[year, nrg + '_MW']        = MW_nrgs[nrg]  
        output_matrix.at[year, nrg + '_MWh']       = MWh_nrgs[nrg]  
        output_matrix.at[year, nrg + '_Supply']    = MWh_nrgs[nrg] + excess_nrgs[nrg]  
        output_matrix.at[year, nrg + '_MW_Cost']   = MW_nrgs[nrg]  * tweaked_nrgs.at['perMW', nrg]
        output_matrix.at[year, nrg + '_MWh_Cost']  = MWh_nrgs[nrg] * tweaked_nrgs.at['perMWh', nrg]
        output_matrix.at[year, nrg + '_Cost']      = MW_nrgs[nrg]  * tweaked_nrgs.at['perMW', nrg]
        output_matrix.at[year, nrg + '_Cost']     += MWh_nrgs[nrg] * tweaked_nrgs.at['perMWh', nrg]
        output_matrix.at[year, nrg + '_CO2_MTon']  = MWh_nrgs[nrg] * tweaked_nrgs.at['CO2_gen', nrg]
        output_matrix.at[year, nrg + '_CO2_Cost']  = MWh_nrgs[nrg] * tweaked_nrgs.at['CO2_gen', nrg] * tweaked_globals['CO2_Price']
        output_matrix.at[year, nrg + '_Knob']      = knobs_nrgs[nrg]
        output_matrix.at[year, nrg + '_Max_Knob']  = max_add_nrgs[nrg]
        output_matrix.at[year, nrg + '_Lifetime']  = tweaked_nrgs.at['Lifetime', nrg]
        

        energy_cost += MW_nrgs[nrg]  * tweaked_nrgs.at['perMW', nrg]
        energy_cost += MWh_nrgs[nrg] * tweaked_nrgs.at['perMWh', nrg]
        total_CO2   += MWh_nrgs[nrg] * tweaked_nrgs.at['CO2_gen', nrg]
        # Storage is really not a producer, and its MW is really MWh of capacity
        if (nrg != 'Storage'):
            total_MW    += MW_nrgs[nrg]
            total_MWh   += MWh_nrgs[nrg]
    
    CO2_cost     = total_CO2 * tweaked_globals['CO2_Price']
    energy_cost += outage_MWh * expensive
    output_matrix.at[year, 'Energy_Cost'] = energy_cost
    output_matrix.at[year, 'CO2_Cost']    = CO2_cost 
    output_matrix.at[year, 'Total_MW']    = total_MW 
    output_matrix.at[year, 'Total_MWh']   = total_MWh
    return output_matrix
    
# Cost function used by minimizer
def cost_function(
                  MWh_nrgs,      
                  MW_nrgs, 
                  excess_nrgs,
                  tweaked_globals,
                  tweaked_nrgs,  
                  expensive,     
                  outage_MWh,    
                  adj_zeros):    
    cost = 0
    for nrg in nrgs:
        old_cost = cost
        supply_MWh = MWh_nrgs[nrg] + excess_nrgs[nrg]
        cost += MW_nrgs[nrg]  * tweaked_nrgs.at['perMW', nrg]
        cost += supply_MWh * tweaked_nrgs.at['perMWh', nrg]
        cost += supply_MWh * tweaked_nrgs.at['CO2_gen', nrg] * tweaked_globals['CO2_Price']
        if debug_cost:
            print(f'{nrg}: ({MW_nrgs[nrg]:,.0f} MW * {tweaked_nrgs.at[nrg, "perMW"]:,.6f} M$/MW)')
            print(f'      + {supply_MWh:,.0f} MWh * {tweaked_nrgs.at[nrg, "perMWh"]:,.6f} M$/MWh')
            print(f'      + {supply_MWh:,.0f} MWh  * {tweaked_nrgs.at[nrg, "CO2_gen"]:,.5f} MTon/MWh * {tweaked_globals["CO2_Price"]:,.6f} M$ per ton')
            print(f' sum = {cost - old_cost:,.0f}') 
        elif debug_cost_sum:
            print(f' {nrg},  {cost - old_cost:.0f}') 
    cost += outage_MWh * expensive
    cost += adj_zeros  * expensive
    if debug_cost:
        print(f'outage + zeros, {cost - old_cost:,.0f}')
        print(f'final: {cost:,.0f}')
    if debug_cost_sum:
        print(f'outage + zeros, {cost - old_cost:.0f}')
        print(f'final: {cost:.0f}')
         
    return cost
    
# This updates the data based on knob settings.
#   Solar, Wind, Nuclear and Coal have nrg total multiplied by knob
#   Gas and (if necessary) storage is used to fill up to target value
#   Any left is used to recharge storage

def update_data(
               knobs_nrgs,    
               hourly_nrgs,   
               MWh_nrgs,      
               MW_nrgs,
               tweaked_nrgs,
               stored,        
               target_hourly, 
               zero_nrgs,
               debug_fig_gas_temp,
               debug_need_temp):    
    
    needed_hourly = target_hourly.copy()
    MW_total      = MW_nrgs.sum()
    adj_zeros     = 0
    if(debug_need_temp):
        debug_need_count = 0
        debug_need_matrix = pd.DataFrame(columns=['When', 'Solar', 'Wind', 'Nuclear', 'Coal', 'Gas', 'Storage', 'Needed', 'Excess'])
        debug_need_matrix.at[debug_need_count, 'When'] = 'Start hourly sums/4'
        debug_need_matrix.at[debug_need_count, 'Needed'] = needed_hourly.sum()/sample_years
        excess = 0
        for hour_of_need in needed_hourly:
            if hour_of_need < 0:
                excess -= hour_of_need
        debug_need_matrix.at[debug_need_count, 'Excess'] = excess/4
        for nrg in ['Solar','Wind','Nuclear','Coal']:
            debug_need_matrix.at[debug_need_count, nrg] = hourly_nrgs[nrg].sum()/sample_years
            debug_need_count += 1
            debug_need_matrix.at[debug_need_count, 'When'] = 'Start MWh_nrgs'
            debug_need_matrix.at[debug_need_count, nrg] = MWh_nrgs[nrg]
            debug_need_count += 1
            debug_need_matrix.at[debug_need_count, 'When'] = 'Knobs'
            debug_need_matrix.at[debug_need_count, nrg] = knobs_nrgs[nrg]
            debug_need_count -= 2
            
        debug_need_count += 3
            
    for nrg in ['Solar','Wind','Nuclear','Coal']:
        if (zero_nrgs[nrg] == 0):
            # High Compute Time
            hourly_nrgs[nrg] *= knobs_nrgs[nrg]
            MWh_nrgs[nrg]    *= knobs_nrgs[nrg]
            if (knobs_nrgs[nrg] > 1):
                MW_nrgs[nrg] *= knobs_nrgs[nrg]
            # High Compute Time
            needed_hourly    -= hourly_nrgs[nrg]
        else:
            adj_zeros += knobs_nrgs[nrg]
    if debug_need_temp:    
        debug_need_matrix.at[debug_need_count, 'When'] = 'After Knobs/4'
        debug_need_matrix.at[debug_need_count, 'Needed'] = needed_hourly.sum()/sample_years
        excess = 0
        for hour_of_need in needed_hourly:
            if hour_of_need < 0:
                excess -= hour_of_need
        debug_need_matrix.at[debug_need_count, 'Excess'] = excess/sample_years
        for nrg in ['Solar','Wind','Nuclear','Coal']:
            debug_need_matrix.at[debug_need_count, nrg] = hourly_nrgs[nrg].sum()/sample_years
        debug_need_count += 1

# Storage MW max_add = 2, so max_build is specs Max_PCT  That is why we divide by 2        
    MW_nrgs['Storage'] += (tweaked_nrgs['Storage'].loc['Max_PCT'] * MW_total * knobs_nrgs['Storage']) / 2
        
    if (knobs_nrgs['Gas'] > 1):
        MW_nrgs['Gas']     = MW_nrgs['Gas'] * knobs_nrgs['Gas']
    
    MWh_nrgs, outage_MWh, stored, excess_MWh = \
        fig_gas_and_storage(
                needed_hourly = needed_hourly,
                gas_max       = MW_nrgs['Gas'],
                storage_max   = MW_nrgs['Storage'],
                stored        = stored,
                MWh_nrgs      = MWh_nrgs,
                debug_temp    = debug_fig_gas_temp)  
    
    if(debug_need_temp):
        debug_need_matrix.at[debug_need_count, 'When'] = 'After fig_gas'
        for nrg in ['Solar','Wind','Nuclear','Coal']:
            debug_need_matrix.at[debug_need_count, nrg] = MWh_nrgs[nrg]
        debug_need_matrix.at[debug_need_count, 'Excess'] = excess_MWh
        debug_need_count += 1
        
    MWh_nrgs, excess_nrgs = fig_excess(MWh_nrgs, excess_MWh, debug_fig_gas_temp)
    
    if(debug_need_temp):
        debug_need_matrix.at[debug_need_count, 'When'] = 'After fig_excess - MWh_nrgs'
        for nrg in nrgs:
            debug_need_matrix.at[debug_need_count, nrg] = MWh_nrgs[nrg]
        debug_need_count += 1
        
        debug_need_matrix.at[debug_need_count, 'When'] = 'After fig_excess - excess'
        for nrg in nrgs:
            debug_need_matrix.at[debug_need_count, nrg] = excess_nrgs[nrg]
        
        save_debug(debug_need_matrix, 'debug_need_matrix.csv') 
        
    return hourly_nrgs, \
           MWh_nrgs,    \
           MW_nrgs,     \
           stored,      \
           adj_zeros,   \
           outage_MWh,  \
           excess_nrgs

# Main function used by minimizer              
def solve_this(
               knobs,                   
               hourly_nrgs,     
               MWh_nrgs,     
               MW_nrgs,
               excess_nrgs,
               stored,       
               target_hourly,
               tweaked_globals,
               tweaked_nrgs,
               expensive,      
               zero_nrgs):   
               
    knobs_nrgs = pd.Series(knobs, index=nrgs, dtype=float)

# Must make a separate copy of these.  Otherwise, Python modifies the original.
# We need it to stay the same for the next minimize run
    new_hourly_nrgs = hourly_nrgs.copy()            
    new_MWh_nrgs    = MWh_nrgs.copy()             
    new_MW_nrgs     = MW_nrgs.copy() 
    new_excess_nrgs = excess_nrgs.copy()
    new_stored      = stored 
    
    if (debug_unexpected_change):
        # These should start the same for all calls to solve_this in one year
        print(new_hourly_nrgs.at[1234,'Solar'],  \
              new_MWh_nrgs['Wind'],   \
              new_MW_nrgs['Nuclear'], \
              new_stored              )
    
    new_hourly_nrgs,  \
    new_MWh_nrgs,     \
    new_MW_nrgs,      \
    new_stored,       \
    adj_zeros,        \
    outage_MWh,       \
    excess_nrgs       \
        = update_data(
                      knobs_nrgs    = knobs_nrgs,
                      hourly_nrgs   = new_hourly_nrgs,
                      MWh_nrgs      = new_MWh_nrgs,    
                      MW_nrgs       = new_MW_nrgs,
                      tweaked_nrgs  = tweaked_nrgs,
                      stored        = new_stored,
                      target_hourly = target_hourly,
                      zero_nrgs     = zero_nrgs,
                      debug_fig_gas_temp = False,
                      debug_need_temp    = False)
                                 
    cost = cost_function(
               MWh_nrgs        = new_MWh_nrgs,
               MW_nrgs         = new_MW_nrgs,
               excess_nrgs     = new_excess_nrgs,
               tweaked_globals = tweaked_globals,
               tweaked_nrgs    = tweaked_nrgs,
               expensive       = expensive,
               outage_MWh      = outage_MWh,
               adj_zeros       = adj_zeros)

    if (debug_step_minimizer):
        print(f'Knobs = {np.round(pd.Series(knobs_nrgs).values,4)} outage_MWh = {outage_MWh:,.0f} cost = {cost:,.0f}')

    return (cost)

# Initialize for year 1 starting place
def init_knobs(tweaked_globals, tweaked_nrgs):
    knobs_nrgs = pd.Series(1,index=nrgs, dtype=float)
    for nrg in nrgs:
        knobs_nrgs[nrg] = tweaked_globals['Demand'] + (1/tweaked_nrgs.at['Lifetime', nrg])
    return knobs_nrgs

def run_minimizer(    
                  hourly_nrgs,         
                  MWh_nrgs,          
                  MW_nrgs,
                  excess_nrgs,
                  stored,              
                  target_hourly,
                  tweaked_globals,
                  tweaked_nrgs,
                  expensive,               
                  zero_nrgs,
                  knobs_nrgs,
                  inbox,
                  region):
    
    #This is total energy produced - Storage is excluded to prevent double-counting
    # Also note that MW_nrgs['Storage'] units are actually MWh of capacity.  Not even compatable.
    MW_total     = MW_nrgs.sum() - MW_nrgs['Storage']
    
    max_add_nrgs = pd.Series(1,index=nrgs, dtype=float)
    for nrg in nrgs:
        if nrg == 'Storage':
            # Nominal for Storage is always half of max.
            max_add_nrgs['Storage'] = 2
        elif MW_nrgs[nrg] == 0.: 
            max_add_nrgs[nrg] = 10.
        else:    
            max_add_nrgs[nrg] = tweaked_globals['Demand'] + ((tweaked_nrgs.at['Max_PCT', nrg]*MW_total)/MW_nrgs[nrg])
        knobs_nrgs[nrg] = min(knobs_nrgs[nrg], max_add_nrgs[nrg] - .00001)
    # and retire some old plants
    hourly_nrgs, MWh_nrgs, MW_nrgs = \
                fig_decadence(hourly_nrgs, MWh_nrgs, MW_nrgs, tweaked_nrgs)  
    
    hi_bound = max_add_nrgs.copy()
    lo_bound = pd.Series(0.,index=nrgs, dtype=float)
    # Gas and Storage are as needed.  If knob < 1, is same as knob = 1 - no new capacity built
    lo_bound['Gas']     = 1.0
    lo_bound['Storage'] = 1.0
    bnds     = Bounds(lo_bound, hi_bound, True)
    method = 'Nelder-Mead'
    fatol = .0001
    xatol = .00001
    opt_done = False
    last_result = 0
    while(not(opt_done)):
        call_time = time.time()
        knobs = pd.Series(knobs_nrgs).values
        if(debug_minimizer):
            print(f'Start Knobs = {knobs}')

        results =   minimize(
                    solve_this, 
                    knobs, 
                    args=(               
                        hourly_nrgs,         
                        MWh_nrgs,          
                        MW_nrgs,
                        excess_nrgs,
                        stored,              
                        target_hourly,
                        tweaked_globals,
                        tweaked_nrgs,
                        expensive,               
                        zero_nrgs
                       ),
                    bounds=bnds,                  
                    method=method, 
                    options={'fatol'  : fatol,
                             'xatol'  : xatol,
                             'maxiter': 10000,
                             'maxfev' : 10000,
                             'disp'   : False
                            }
        )
        end_time = time.time() 

        if not(results.success):
            print('***************** Minimizer Failed ********************')
            msgbox('Minimizer Failure', f'{results.message}, {inbox["Title"].loc["Initial"]}, {region}' , 1)
            print(results)
            raise RuntimeError('Minimizer Failure' )
        elif(debug_minimizer):
            print (f'fatol {fatol} xatol {xatol}')
            print(f'Knobs  {results.x}')
            print(f'Results {results.fun:,.3f} Time {end_time - call_time:,.2f} with {results.nfev} runs')
            
        knobs      = results.x
        knobs_nrgs = pd.Series(knobs, index=nrgs, dtype=float)   
        if ((last_result > (results.fun * .99)) and \
            (last_result < (results.fun * 1.01))):
            opt_done = True
        else:
            if(last_result > 0):
                print('Extra try at minimizer')
            last_result = results.fun
            fatol       = round_significant(fatol/10.,4)
            xatol       = round_significant(xatol/10.,4)
                     
    return knobs_nrgs, max_add_nrgs

def one_case(year):
    # This array reverses decadence - so starting place is the same
    #  redemption = np.array([1.034483,	1.041666,	1.02564,	1.04167,	1.02564,	0.])
    #  Optimized Case
    # test_cases = np.array([[1.7906705,  1.15446494, 1.03271335, 1.12593594, 0.91245342, 0.23845428]])
    test_cases =   np.array([[1.034483,	1.041666,	1.02564,	1.04167,	1.02564,	1.999],
                             [1.034483,	1.041666,	1.02564,	1.04167,	1.02564,	1.999],
                             [1.034483,	1.041666,	1.02564,	1.04167,	1.02564,	1.999]])
    knobs = test_cases[year-1]
    knobs_nrgs = pd.Series(knobs, index=nrgs, dtype=float)
    print(knobs)
    return knobs_nrgs
    
# main is a once-through operation, so we try to do as much calc here as possible
def do_region(region):
    start_time    = time.time()
    inbox         = get_inbox()
    years         = inbox['Years'].loc['Initial']
    specs_nrgs    = get_specs_nrgs()
    hourly_nrgs   = get_eia_data(region) 
    # If there is old data there, remove it
    outbox_path = './python/mailbox/outbox'    
    file_path = f'{outbox_path}/{inbox["Title"].loc["Initial"]}-{region}.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
    
    MW_nrgs      = pd.Series(0,index=nrgs, dtype=float)
    MWh_nrgs     = pd.Series(0,index=nrgs, dtype=float)
    total_hourly = pd.Series(np.zeros(sample_points, dtype=float))
    zero_nrgs    = pd.Series(0,index=nrgs, dtype=float)
    for nrg in nrgs:
        MW_nrgs[nrg]    = hourly_nrgs[nrg].max()
        MWh_nrgs[nrg]   = hourly_nrgs[nrg].sum() / sample_years
        total_hourly   += hourly_nrgs[nrg]   
        if (MWh_nrgs[nrg] == 0) & (nrg != 'Storage'):
            zero_nrgs[nrg] = 1
            
    MW_total      = MW_nrgs.sum()                       
    output_matrix = init_output_matrix()
    hourly_cost   = (MWh_nrgs * specs_nrgs.at['Variable', nrg]).sum()/(365.25*24)
    hourly_cost  += (MW_nrgs  * specs_nrgs.at['Fixed', nrg]).sum()/(365.25*24)
    expensive     = hourly_cost * 100
    
    stored        = 0
    outage_MWh    = 0
    target_hourly = total_hourly.copy()
    CO2_Price     = 0
    if (debug_update):
        debug_temp = False
        debug_update_matrix = pd.DataFrame(columns=['When', 'Total_MWh', 'Sum_Of_Hourly/4', 'Target_Hourly/4'])
        debug_update_count = 0
        debug_update_matrix.at[debug_update_count, 'When'] = 'Year 0'
        debug_update_matrix.at[debug_update_count, 'Total_MWh'] = MWh_nrgs.sum()
        hourly_sum = 0
        for nrg in nrgs:
            hourly_sum += hourly_nrgs[nrg].sum()
        debug_update_matrix.at[debug_update_count, 'Sum_Of_Hourly/4'] = hourly_sum/sample_years
        debug_update_matrix.at[debug_update_count, 'Target_Hourly/4'] = target_hourly.sum()/sample_years
        debug_update_count += 1
        
    tweaked_globals, tweaked_nrgs = init_tweaks(specs_nrgs, inbox)
#Output Year Zero
    knobs_nrgs  = pd.Series(1., index=nrgs, dtype=float)
    excess_nrgs = pd.Series(0., index=nrgs, dtype=float)
    
    output_matrix = \
                add_output_year(
                    MWh_nrgs        = MWh_nrgs,                          
                    MW_nrgs         = MW_nrgs,
                    excess_nrgs     = excess_nrgs,
                    tweaked_globals = tweaked_globals,
                    tweaked_nrgs    = tweaked_nrgs,
                    expensive       = expensive,
                    outage_MWh      = outage_MWh,
                    output_matrix   = output_matrix,
                    year            = 0,
                    knobs_nrgs      = knobs_nrgs,
                    max_add_nrgs    = knobs_nrgs)
        
    knobs_nrgs = init_knobs(tweaked_globals=tweaked_globals, tweaked_nrgs=tweaked_nrgs)    
    
    for year in range(1, int(years)+1):
        print(f'Year {year} in {region}')
# Update prices                       
        tweaked_globals, tweaked_nrgs = \
            fig_tweaks (
                    tweaked_globals = tweaked_globals,
                    tweaked_nrgs    = tweaked_nrgs,
                    inbox           = inbox,
                    year            = year)
        
        target_hourly = (target_hourly * tweaked_globals['Demand']) + (hourly_nrgs['Others'] * (tweaked_globals['Demand']-1))
    
        if(debug_update):
            debug_update_matrix.at[debug_update_count, 'When'] = 'After_Demand_Tweak'
            debug_update_matrix.at[debug_update_count, 'Target_Hourly/4'] = target_hourly.sum()/sample_years            
            debug_update_count += 1
            
# Now optimize this year            
        # Normal run
        if normal_run:
            knobs_nrgs, max_add_nrgs = run_minimizer( \
                                hourly_nrgs    = hourly_nrgs,         
                                MWh_nrgs       = MWh_nrgs,         
                                MW_nrgs        = MW_nrgs, 
                                excess_nrgs    = excess_nrgs,
                                stored         = stored,              
                                target_hourly  = target_hourly,
                                tweaked_globals= tweaked_globals,
                                tweaked_nrgs    = tweaked_nrgs,
                                expensive      = expensive,               
                                zero_nrgs      = zero_nrgs,
                                knobs_nrgs     = knobs_nrgs,
                                inbox          = inbox,
                                region         = region)
                                
        elif (debug_one_case):
            knobs_nrgs   = one_case(year)
            max_add_nrgs = pd.Series(999.,index=nrgs, dtype=float)
             

# Update data based on optimized knobs 
        if (debug_update):
            debug_update_matrix.at[debug_update_count, 'When'] = 'Pre-Update decayed'
            debug_update_matrix.at[debug_update_count, 'Total_MWh'] = MWh_nrgs.sum()
            hourly_sum = 0
            for nrg in nrgs:
                hourly_sum += hourly_nrgs[nrg].sum()
            debug_update_matrix.at[debug_update_count, 'Sum_Of_Hourly/4'] = hourly_sum/sample_years
            debug_update_matrix.at[debug_update_count, 'Target_Hourly/4'] = target_hourly.sum()/sample_years
            debug_update_count += 1
            
        hourly_nrgs,    \
        MWh_nrgs,       \
        MW_nrgs,        \
        stored,         \
        adj_zeros,      \
        outage_MWh,     \
        excess_nrgs     \
            = update_data( 
                    knobs_nrgs    = knobs_nrgs,
                    hourly_nrgs   = hourly_nrgs,
                    MWh_nrgs      = MWh_nrgs,    
                    MW_nrgs       = MW_nrgs,
                    tweaked_nrgs  = tweaked_nrgs,
                    stored        = stored,
                    target_hourly = target_hourly,
                    zero_nrgs     = zero_nrgs,
                    debug_fig_gas_temp = debug_fig_gas,
                    debug_need_temp    = debug_need)
        
        if (debug_update): 
            debug_update_matrix.at[debug_update_count, 'When']      = 'Post-Update'
            debug_update_matrix.at[debug_update_count, 'Total_MWh'] = MWh_nrgs.sum()
            hourly_sum = 0
            for nrg in nrgs:
                hourly_sum += hourly_nrgs[nrg].sum()
            debug_update_matrix.at[debug_update_count, 'Sum_Of_Hourly/4'] = hourly_sum/sample_years
            debug_update_matrix.at[debug_update_count, 'Target_Hourly/4'] = target_hourly.sum()/sample_years
            debug_update_count += 1
            save_debug(debug_update_matrix, 'debug_update_matrix.csv')

# Output results of this year             
        output_matrix = \
            add_output_year(
              MWh_nrgs        = MWh_nrgs,         
              MW_nrgs         = MW_nrgs,
              excess_nrgs     = excess_nrgs,
              tweaked_globals = tweaked_globals,
              tweaked_nrgs     = tweaked_nrgs,
              expensive       = expensive,
              outage_MWh      = outage_MWh,
              output_matrix   = output_matrix,
              year            = year,
              knobs_nrgs      = knobs_nrgs,
              max_add_nrgs    = max_add_nrgs)
    # End of years for loop    
    outbox_path = './python/mailbox/outbox'
    file_path = f'{outbox_path}/{inbox["Title"].loc["Initial"]}-{region}.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
    # minimized returned a really really small number for outage.  Excel couldn't handle it.
    # So rounding it to make that number 0.  Careful if you use really small numbers here.
    output_matrix_t = output_matrix.round(8).transpose()
    output_matrix_t.to_csv(file_path)
    print(f'{region} Total Time = {(time.time() - start_time)/60:.2f} minutes') 
    
# Copied from Stack Overflow:

class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

def main():
    inbox         = get_inbox()
    region        = inbox['Region'].loc['Initial']
    print('Starting ', inbox['Title'].loc['Initial'])
    
        
    if (not kill_parallel) and (region == 'All'):
        regions = get_all_regions()
        region_process = pd.Series(0,index=regions,dtype=object)

        for region in regions:
    # Create new child process for each region
            region_process[region] = Process(target=do_region, args=(region,))
            region_process[region].start()
    
        for region in regions:
    # Now, wait for all of them to be done
            region_process[region].join()
            if(region_process[region].exception):
                error, traceback = region_process[region].exception
                print (region, inbox['Title'].loc['Initial'], 'Error')
                msgbox("Process Error", f"{region} {inbox['Title'].loc['Initial']} Error", 1)
                print(traceback)
            else:
                print(region, ' Done')
    # kill_parallel True or not 'All'
    elif region == 'All':
        regions = get_all_regions()
        for region in regions:
            do_region(region)
    else:
        do_region(region)
        
if __name__ == '__main__':
    main()



