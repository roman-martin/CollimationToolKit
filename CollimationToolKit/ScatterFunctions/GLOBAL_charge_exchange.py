'''
This module provides functions to run the GLOBAL charge exchange code as
a scatter function of the LimitFoil class.

The module DOES NOT do the calculations itself but merely provides an interface
to call the executable.

In its current state GLOBAL_charge_exchange.py only correctly runs on Linux
systems and requires the GLOBAL executable in PATH.

The GLOBAL executable can be obtained from
https://web-docs.gsi.de/~weick/charge_states/
'''

import os
import shutil
import subprocess
import csv
import numpy as np

from scipy.constants import physical_constants
nmass = physical_constants['atomic mass constant energy equivalent in MeV'][0] * 1e6


formatable_input_file_str = ''' GLOBAL Input file:
 Projectile:    A          Z        Q          Energy(MeV/u)
               {}        {}       {}             {}
 Target:        A          Z    Dt(mg/cm^2) 
              {}        {}      {}
 Options:     I_CHAR    I_LOOP   I_OUTP     I_WR
                0          0        1         3
 Q-states: 
    0  1  2  3  4  5  6  7  8  9
'''


def GLOBAL(self, particle, idx=[]):
    # Scatter function that can be used by the LimitFoil class
    target_A = self.A
    target_Z = self.Z
    density = self.density
    thickness = self.thickness  
    target_Dt = 1e3 * density * 1e2 *thickness # 1e3 mg/g and 1e2 cm/m to get mg/cm^2

    if not hasattr(particle.state, "__iter__"):
        if hasattr(particle, "A"):
                p_A = particle.A
        else:
            raise AttributeError("Atomic mass (particle.A) not defined.")
        p_Z = particle.Z
        p_Q = particle.q0 * particle.qratio
        p_energy = (particle.energy - particle.mass0) / p_A
                        # we will ignore delta_p here, assuming the spread is small
        
        chargestate_probability, E_out = __run_GLOBAL__(target_A, target_Z,
                                                        target_Dt, p_A, p_Z,
                                                        p_Q, p_energy)
                                                        
        new_charge_state = np.random.choice(len(chargestate_probability),
                                            p = chargestate_probability)
        particle.qratio = (particle.Z-new_charge_state) / particle.q0
        particle.add_to_energy( (E_out-p_energy)*p_A )
    else:
        tmp_add_to_energy = np.zeros(len(particle.state))
        for ii in idx:      # this is SUPER inefficient but straight forward
            if hasattr(particle, "A"):
                p_A = particle.A[ii]
            else:
                raise AttributeError("Atomic mass (particle.A) not defined.")
            p_Z = particle.Z[ii]
            if not p_Z:
                continue    # in case we ever track mixed particle sets including non-ions
            p_Q = particle.q0 * particle.qratio[ii]      
            p_energy = (particle.energy[ii] - particle.mass0) / p_A
            
            chargestate_probability, E_out = __run_GLOBAL__(target_A, target_Z,
                                                            target_Dt, p_A, p_Z,
                                                            p_Q, p_energy)
                                                        
            new_charge_state = np.random.choice(len(chargestate_probability),
                                                p = chargestate_probability)
            particle.qratio[ii] = (p_Z-new_charge_state) / particle.q0
            tmp_add_to_energy[ii] = (E_out-p_energy)*p_A
        tmp_qratio = particle.qratio    # this is needed to...
        particle.qratio = tmp_qratio    # ... trigger the qratio setter
        particle.add_to_energy(tmp_add_to_energy)
                                                    

def __run_GLOBAL__(target_A, target_Z, target_Dt, p_A, p_Z, p_Q, p_energy):
    if p_Z - p_Q > 28:
        chargestate = 28
    else:
        chargestate = p_Z - p_Q
    
    #---------------------------------------------------------------------------
    #----- Run the GLOBAL charge exchange code ------------------------------
    #---------------------------------------------------------------------------
    # the following is neccesarily complicated because GLOBAL needs to be run in
    # its own directory to find its data files and only accepts input file names
    # up to five characters (INCLUDING PATH but excluding .ginput file extension)
    globalcodedir_path = shutil.which('global')[:-len('global')] #this only works on Linux, but do we need to support inferior systems?
    if not globalcodedir_path:
        raise RuntimeError("Executable of GLOBAL charge exchange code not found")
    
    #----- Create input file for GLOBAL code---------------------------------
    input_str = formatable_input_file_str.format(p_A, p_Z, chargestate, p_energy/1e6, target_A, target_Z, target_Dt)
    with open(globalcodedir_path + 'tmp.ginput', 'w') as global_input_file:
        global_input_file.write(input_str)
    
    #----- run GLOBAL code---------------------------------------------------
    with subprocess.Popen('global', cwd = globalcodedir_path,
                                    stdin = subprocess.PIPE,
                                    stdout = subprocess.DEVNULL) as p:
        p.stdin.write('./tmp'.encode() + os.linesep.encode()) # load input file
        p.stdin.write('0'.encode() + os.linesep.encode()) # no changing parameters
        p.stdin.write('./tmp'.encode() + os.linesep.encode())  # output file name
        p.stdin.write('n'.encode() + os.linesep.encode())  # no new input file
        p.communicate(input=('n'.encode() + os.linesep.encode())) # do not repeat


    #---------------------------------------------------------------------------
    #----- Check if output file is correct-----------------------------------
    #---------------------------------------------------------------------------
    with open(globalcodedir_path + 'tmp.globout', 'r') as outputfile:
        nextline_are_results = False
        for line in outputfile:
            if '(Z=' in line and 'A=' in line and 'Qe=' in line:
                verify_input_str = line
            elif 'Q(0 )' in line and 'Q(1 )' in line and 'Q(2 )' in line:
                nextline_are_results = True
            elif nextline_are_results:
                results_str = line

    projectile_str= verify_input_str[:verify_input_str.find('on')]
    target_str = verify_input_str[verify_input_str.find('on'):]
    # check projectile data
    for in_var, start_str, end_str in zip(  [p_Z,p_A,p_Q,p_energy/1e6],
                                            ['(Z=','A=','Qe=','E = '],
                                            [', ',', ',') at E',' MeV/u']):
        idx_s = projectile_str.find(start_str) + len(start_str)
        idx_e = projectile_str.find(end_str, idx_s)
        if start_str == 'E = ':
            comparison_precision = 0.05*in_var  # this parameter might need some tweaking
        else:
            comparison_precision = 0.1
        if not abs(in_var - float(projectile_str[idx_s:idx_e])) < comparison_precision: #GLOBAL does some rounding
            print(start_str, float(projectile_str[idx_s:idx_e]), ', should be', in_var)
            raise RuntimeError("GLOBAL charge exchange code did not use correct particle input")
    # check target data
    for in_var, start_str, end_str in zip(  [target_Z,target_A,target_Dt],
                                            ['(Z=','A=','D='],
                                            [', ',', ','mg/cm^2']):
        idx_s = target_str.find(start_str) + len(start_str)
        idx_e = target_str.find(end_str, idx_s)
        if start_str == 'D=':
            comparison_precision = 0.05*in_var  # this parameter might need some tweaking
        else:
            comparison_precision = 0.1
        if not abs(in_var - float(target_str[idx_s:idx_e])) < comparison_precision: #GLOBAL does some rounding
            print(start_str, float(target_str[idx_s:idx_e]), ', should be', in_var)
            raise RuntimeError("GLOBAL charge exchange code did not use correct target input")

    #----- Clean up after us -------------------------------
    if os.path.isfile(globalcodedir_path + 'tmp.ginput'):
        os.remove(globalcodedir_path + 'tmp.ginput')
    if os.path.isfile(globalcodedir_path + 'tmp.globout'):
        os.remove(globalcodedir_path + 'tmp.globout')

    #---------------------------------------------------------------------------
    #----- Parse GLOBAL output string----------------------------------------
    #---------------------------------------------------------------------------
    results = list(csv.reader([results_str], delimiter=' ',
                                            skipinitialspace=True,
                                            quoting=csv.QUOTE_NONNUMERIC    
                  ))[0]
    E_out = results[2]*1e6   # 1e6: MeV to eV
    chargestate_probability = results[3:13]
    # normalization because GLOBAL does some rounding or uses 32bit floats
    chargestate_probability = [ p/np.sum(chargestate_probability) \
                                for p in chargestate_probability  ]
    
    return chargestate_probability, E_out



    
