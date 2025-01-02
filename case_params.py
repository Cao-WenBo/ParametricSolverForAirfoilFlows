case_params_list = []

# NACA0012
case_params = {
    #flow parameters
    'Re':100, 'alpha':15,
    #shape parameters
    'cst_up':[0.171787,0.155338,0.161996,0.137638,0.145718,0.143815],
    'cst_down':[-0.171787,-0.155338,-0.161996,-0.137638,-0.145718,-0.143815],
    
    #result
    'ref_Cp_path':'result_Cp/NACA0012_400_150_Re100_AoA15_CpCf.dat',
    }
case_params_list.append(case_params)

# NACA4424
case_params = {
    #flow parameters
    'Re':200, 'alpha':11,
    #shape parameters
    'cst_up':[ 0.46684444,  0.2651003 ,  0.63876362,  0.19284059,  0.52135861, 0.43724843],
    'cst_down':[ -0.25590958, -0.28210027, -0.10081671, -0.28192463,-0.0643716 , -0.21992341],
    
    #result
    'ref_Cp_path':'result_Cp/NACA4424_400_150_Re200_AoA11_CpCf.dat',
    }
case_params_list.append(case_params)

# RAE2822
case_params = {
    #flow parameters
    'Re':500, 'alpha':7,
    #shape parameters
    'cst_up':[ 0.1256687 ,  0.14574057,  0.15190872,  0.21353791,  0.17871332, 0.20866746],
    'cst_down':[ -0.1331873 , -0.11892053, -0.22043243, -0.12812116, -0.08110105,  0.05140584],
    
    #result
    'ref_Cp_path':'result_Cp/RAE2822_400_150_Re500_AoA7_CpCf.dat',
    }
case_params_list.append(case_params)

# RAE5214
case_params = {
    #flow parameters
    'Re':1000, 'alpha':3,
    #shape parameters
    'cst_up':[ 0.2172471 ,  0.05024209,  0.25149746,  0.07927185,  0.2133891 , 0.12742785],
    'cst_down':[ -0.13526327, -0.02885926, -0.19752833, -0.06115319, -0.01507064,  0.03944415],
    
    #result
    'ref_Cp_path':'result_Cp/RAE5214_400_150_Re1000_AoA3_CpCf.dat',
    }
case_params_list.append(case_params)

# S2050
case_params = {
    #flow parameters
    'Re':2500, 'alpha':-1,
    #shape parameters
    'cst_up':[ 0.1269453 ,  0.19145795,  0.08596173,  0.21290957,  0.07890382, 0.2033505],
    'cst_down':[ -0.1117779 , -0.08793519, -0.0596662 , -0.11482333, -0.03899674,  0.06464392],
    
    #result
    'ref_Cp_path':'result_Cp/S2050_400_150_Re2500_AoA-1_CpCf.dat',
    }
case_params_list.append(case_params)

# S9000
case_params = {
    #flow parameters
    'Re':5000, 'alpha':-5,
    #shape parameters
    'cst_up':[ 0.15575312,  0.21740341,  0.12037415,  0.23169055,  0.12565486, 0.21000754],
    'cst_down':[ -0.08496331, -0.06819646, -0.04163422, -0.04041423, -0.04856975,  0.07381856],
    
    #result
    'ref_Cp_path':'result_Cp/S9000_400_150_Re5000_AoA-5_CpCf.dat',
    }
case_params_list.append(case_params)

# %%