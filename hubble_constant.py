#Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import math
from uncertainties import ufloat

#Load the spectral dataset
spectral_data = pd.read_csv('Data/SpectralData_Hbeta.csv', skiprows=4)

#Read the distance data and do some data cleanup
#The following line is adapted with input from ChatGPT
dist_data = np.genfromtxt('Data/dist_data.txt', comments='#', dtype=None, encoding='utf-8')

dist_data_remove_headers = dist_data[1:]
dist_df = pd.DataFrame(dist_data_remove_headers, columns=['User_id', 'TImestamp', 'Observation_number', 'Distance', 'Instrument_response'])
dist_df_cleaned = dist_df[dist_df['Instrument_response'] != 'E'].iloc[:, 2:4]

#Creating a function to fit the plot
def model(x, m, c, A, mu, sigma):
    return (A * np.exp(-(x - mu)**2 / (2 * sigma**2))) + (m * x + c)

#Check if function is working properly
print(model(2, 9, 2, 3, 4, 4))
#Function works properply (verified using calculator)

#Write a function to compute the veloctiy

def velocity(lambda_0, lambda_e=486.1e-9, c=2.997e8):
    numerator = (lambda_0/lambda_e)**2 - 1
    denominator = (lambda_0/lambda_e)**2 + 1
    v = (numerator/denominator)*c
    return v

#Filter the spectral data to get only valid observations (input from Claude Code)
valid_obs_numbers = dist_df_cleaned['Observation_number'].tolist()

# Build list of columns to keep
columns_to_keep = []

for i in range(0, len(spectral_data.columns), 2):
    obs_number = spectral_data.columns[i]  # First column of the pair
    
    if obs_number in valid_obs_numbers:
        # Keep both columns (the frequency and intensity columns)
        columns_to_keep.append(spectral_data.columns[i])     
        columns_to_keep.append(spectral_data.columns[i+1])    

# Create filtered dataframe
spectral_data_filtered = spectral_data[columns_to_keep]

#Estimating guess parameters and fitting the curve for each plot 
no_columns = spectral_data_filtered.shape[1]

observed_frequency = []
velocities = []

observation_number = []

uncertainties = []

for i in range(0, int(no_columns), 2):
   j = i+1
   obs = spectral_data_filtered.iloc[:, i:j+1].copy()
   obs_no = obs.columns[0]
   observation_number.append(obs_no)
   obs.rename(columns={obs.columns[0]: 'Frequency', obs.columns[1]: 'Intensity'}, inplace=True)
   
   #To estimate slope use final and initial points
   guess_m = ((obs['Intensity'].tolist()[-1]) - (obs['Intensity'].tolist()[0]))/((obs['Frequency'].tolist()[-1]) - (obs['Frequency'].tolist()[0]))
   #To estimate y-intercept utilise estimate slope and any point
   guess_c = (obs['Intensity'].tolist()[0]) - (guess_m * obs['Frequency'].tolist()[0])
   #To estimate mean, use position of max value (input from Claude Code)
   max_intensity_index = obs['Intensity'].idxmax()
   guess_mu = obs.loc[max_intensity_index, 'Frequency']
   #To estimate A, subtract line fit value at mean from max value 
   line_estimate = np.polyval([guess_m, guess_c], guess_mu)
   guess_A = (obs['Intensity'].max() - line_estimate)
   #Estimate sigma ~ assuming standard for all curves
   fwhm = 1e13
   guess_sigma = fwhm/(2*np.sqrt(2*math.log(2)))

   my_guesses = [guess_m, guess_c, guess_A, guess_mu, guess_sigma]

   obs_frequency = obs['Frequency']
   obs_intensity = obs['Intensity']

   #Convert from Hz to THz
   obs_frequency_thz = obs['Frequency']/1e12

   obs_fit = curve_fit(model, obs_frequency, obs_intensity, p0=my_guesses)
   obs_data_fit = model(obs_frequency, *obs_fit[0])

   optimised_mean = obs_fit[0][3]
   #optimised_mean = observed frequency
   observed_frequency.append(optimised_mean)

   perr = np.sqrt(np.diag(obs_fit[1]))

   #Appending the uncertainty associated with the observed frequency used in following calculations 
   uncertainty = perr[3]
   uncertainties.append(uncertainty)

   #Plotted data to verify if curve fit functioned properly, not necessary for code to work. 

#    plt.figure()
#    plt.plot(obs_frequency_thz, obs_intensity, label='data')
#    plt.plot(obs_frequency_thz, obs_data_fit, label='fit')
#    plt.xlabel('Frequency (THz)')
#    plt.ylabel('Intensity (A.U)')
#    plt.title('Frequency v. Intensity')
#    plt.legend()
#    plt.show()

#Calculate the velocity and its associated error from the observed frequency
velocities = []
index = 0
for f in observed_frequency:
    f = ufloat(f, uncertainties[index])
    wavelength = 2.9979e8/f
    obs_velocity = velocity(wavelength)
    velocities.append(obs_velocity)
    index += 1


#Extract only the nominal values 
velocities_nominal_values = []
for v in velocities:
    velocities_nominal_values.append(v.n)

#Extract only the uncertainties 
velocities_uncertainties = []
for v in velocities:
    velocities_uncertainties.append(v.s)

#Let's sort the distance dataframe to match the velocities
observation_number = [str(num) for num in observation_number]
#The following line is adapted from Claude Code
dist_df_cleaned_sorted = dist_df_cleaned.set_index('Observation_number').loc[observation_number].reset_index()
dist_df_cleaned_sorted['Distance'] = dist_df_cleaned_sorted['Distance'].astype(float)

#Let us fit our line for the velocity against distance graph and plot it
polyfit, polycov = np.polyfit(dist_df_cleaned_sorted['Distance'], velocities_nominal_values, 1, cov=True, w=1/np.array(velocities_uncertainties))
polyfit_sigma = np.sqrt(polycov[0,0])

polyval = np.polyval(polyfit, dist_df_cleaned_sorted['Distance'])

velocities_nominal_values_converted = [x/1e7 for x in velocities_nominal_values]

plt.plot(dist_df_cleaned_sorted['Distance'], velocities_nominal_values_converted, 'x')
plt.plot(dist_df_cleaned_sorted['Distance'], polyval/1e7)
plt.xticks(np.arange(80, 250, 40))
plt.ylabel(r'Velocities (10$^{7}$ m/s)')
plt.xlabel(r'Distance (Mpc)')
plt.title('Redshift Velocity v. Distance')
plt.show()

print(f'Estimate of Hubble constant {polyfit[0]} +/- {polyfit_sigma} ms-1 Mpc-1')

#Estimate of Hubble's constant: 71.23271235285971 +/- 4.974261884860732 km/s/Mpc 