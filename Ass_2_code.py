#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def file_reader(filename):
    '''
    file_reader creates new dataframe with given file path

    Parameters
    ----------
    filename : STR
        String of filepath.

    Returns
    -------
    df : Dataframe
        Dataframe of given filepath.

    '''
    df = pd.read_csv(filename, skiprows=(4))
    
    return df


def transpose(df):
    '''
    transpose function will create transpose of given dataframe

    Parameters
    ----------
    df : Dataframe
        Dataframe for which transpose to be found.

    Returns
    -------
    df : Dataframe
        Given dataframe.
    df_transpose : Dataframe
        Transpose of given dataframe.

    '''
    df_transpose = df.transpose()
    df_transpose.columns = df.index
    df_transpose = df_transpose.iloc[1:, :]
 
    return df, df_transpose


def grouped_barplot(data, xlabel, ylabel, title):
    '''
    grouped_barplot produces grouped bar graph

    Parameters
    ----------
    data : dataframe
        dataframe for which bar graph to be plot.
    xlabel : STR
        xlabel used in bar graph.
    ylabel : STR
        ylabel used in bar graph.
    title : STR
        title used in bar graph.

    '''
    num_columns = len(data.columns)
    wid = 0.8/num_columns
    offst = wid/2
    x_tks = data.index
    col=['b','g','r','c','m','0.7']

    plt.figure()
    sb_plot = plt.subplot()
    for i, year in enumerate(data.columns):
        sb_plot.bar([tick+offst+wid*i for tick in range(len(x_tks))],
                    data[year], width=wid, label=year, color=col[i])

    sb_plot.set_xlabel(xlabel)
    sb_plot.set_ylabel(ylabel)
    sb_plot.set_title(title)

    sb_plot.set_xticks([tick+0.4 for tick in range(len(x_tks))])
    sb_plot.set_xticklabels(x_tks, rotation=45)
    sb_plot.legend(title='Year', bbox_to_anchor=(1, 1))
    plt.savefig(title+'.png', bbox_inches='tight', dpi=450)
    plt.show()


def country_wise_data(my_data_list, column_names, country):
    '''
    country_wise_data will create a new dataframe with only one country data

    Parameters
    ----------
    my_data_list : List
        List of dataframes.
    column_names : List
        List of strings used as new dataframe column names.
    country : STR
        String of my country.

    Returns
    -------
    country_wise_data : Dataframe
        Dataframe with only my country data.

    '''
    country_wise_data = pd.DataFrame()
    for i in range(len(my_data_list)):
        country_wise_data[column_names[i]] = my_data_list[i
                                                          ].loc['1990':'2019',
                                                                 country
                                                                 ].astype(int)

    return country_wise_data


def correlation_coefficient_heatmap(my_data, title, cmap):
    '''
    Creates correlation heatmap of given data columns

    Parameters
    ----------
    my_data : Datfarame
        Dataframe of my country.
    title : STR
        String used as title in plot.
    cmap : STR
        cmap attribute for colour.

    '''
    corr_matrix = np.array(my_data.corr())
    print('Correlation Coefficient matrix of ', title, ' on indicators is')
    print(corr_matrix, '\n')
    plt.figure()
    plt.imshow(corr_matrix, cmap=cmap,
               interpolation='nearest', aspect='auto')
    plt.xticks(range(len(my_data.columns)), my_data.columns, rotation=90)
    plt.yticks(range(len(my_data.columns)), my_data.columns)

    for i in range(len(my_data.columns)):
        for j in range(len(my_data.columns)):
            plt.text(j, i, corr_matrix[i, j].round(2),
                     ha="center", va="center", color="black")
    plt.colorbar()
    plt.title(title)
    plt.savefig(title+'.png', bbox_inches='tight', dpi=500)
    plt.show()


#Reading Forest Area file
forest_area = file_reader("forest_area.csv")
forest_area = forest_area.set_index('Country Name', drop=True)
print('Forest Area data head')
print(forest_area.head(), '\n')

#Creating transpose of my dataframe
forest_area, forest_area_tr = transpose(forest_area)

print('Forest Area data Transpose head')
print(forest_area_tr.head(), '\n')

my_countries = ['Australia', 'China', 'Germany', 'India',
                'United Kingdom', 'United States']


#Grouped barplot of forest area of my countries for every 6 years
forest = forest_area.loc[my_countries, '1990':'2020':6].copy()
print('Forest Area data description:')
print(forest.describe(),'\n')
grouped_barplot(forest, 'Country Name', 'Forest Area in sq Km',
                'Forest Area in sq Km of my countries')

#Reading Agricultural Land Area file
agricultural_land = file_reader("agricultural_land.csv")
agricultural_land = agricultural_land.set_index('Country Name', drop=True)

#Creating transpose of my dataframe
agricultural_land, agricultural_land_tr = transpose(agricultural_land)


#Grouped barplot of Agricultural Land area of my countries for every 6 years
#from 1990 to 2020
agri_land = agricultural_land.loc[my_countries, '1990':'2014':6].copy()
grouped_barplot(agri_land, 'Country Name', 'Agricultural Land Area in sq Km',
                'Agricultural Land Area in sq Km of my countries')


agri_land_tr = agricultural_land_tr.loc['1960':'2020', my_countries].copy()
plt.figure()
for i in agri_land_tr.columns:
    plt.plot(agri_land_tr.index, agri_land_tr[i], label=i)
plt.xticks(agri_land_tr.index[::10])
plt.xlabel('Year')
plt.ylabel('Agricultural Land Area in sq Km')
plt.title('Agricultural Land Area in sq Km of my countries')
plt.legend(title='Country', bbox_to_anchor=(1, 1))
plt.show()


#Reading Population Total file
population_total = file_reader("population_total.csv")
population_total = population_total.set_index('Country Name', drop=True)

#Creating transpose of my dataframe
population_total, population_total_tr = transpose(population_total)


#Grouped barplot of Population Total of my countries for every 6 years from
#1990 to 2020
population = population_total.loc[my_countries, '1990':'2020':6].copy()
grouped_barplot(population, 'Country Name', 'Population Total',
                'Population Total of my countries')


population_tr = population_total_tr.loc['1960':'2021', my_countries].copy()
plt.figure()
for i in population_tr.columns:
    plt.plot(population_tr.index, population_tr[i], label=i)
plt.xticks(population_tr.index[::10])
plt.xlabel('Year')
plt.ylabel('Population Total')
plt.title('Population Total of my countries')
plt.legend(title='Country', bbox_to_anchor=(1, 1))
plt.show()

#Reading Urban Population Total file
urban_population = file_reader("urban_population.csv")
urban_population = urban_population.set_index('Country Name', drop=True)

#Creating transpose of my dataframe
urban_population, urban_population_tr = transpose(urban_population)


#Grouped barplot of Urban Population of my countries for every 6 years from
#1990 to 2020
urban_pop = urban_population.loc[my_countries, '1990':'2020':6].copy()
grouped_barplot(urban_pop, 'Country Name', 'Urban Population',
                'Urban Population of my countries')


urban_pop_tr = urban_population_tr.loc['1960':'2021', my_countries].copy()
plt.figure()
for i in urban_pop_tr.columns:
    plt.plot(urban_pop_tr.index, urban_pop_tr[i], label=i)
plt.xticks(urban_pop_tr.index[::10])
plt.xlabel('Year')
plt.ylabel('Urban Population')
plt.title('Urban Population of my countries')
plt.legend(title='Country', bbox_to_anchor=(1, 1))
plt.show()

#Calculating skewness and kurtosis of Urban Population

urb_sk=[]
urb_kurt=[]
urb_sk_kurt=pd.DataFrame()
urb_sk_kurt.index=urban_pop_tr.columns
for i in urban_pop_tr.columns:
    urb_sk.append(urban_pop_tr[i].skew())
    urb_kurt.append(urban_pop_tr[i].kurtosis())
urb_sk_kurt['Skewness']=urb_sk
urb_sk_kurt['Kurtosis']=urb_kurt
print('Skewness and Kurtosis of Urban Population is:')
print(urb_sk_kurt, '\n')

#Reading CO2 Emissions Total file
co2_emissions = file_reader("co2_emissions.csv")
co2_emissions = co2_emissions.set_index('Country Name', drop=True)

#Creating transpose of my dataframe
co2_emissions, co2_emissions_tr = transpose(co2_emissions)


#Grouped barplot of CO2 Emissions of my countries for every 6 years from
#1990 to 2020
co2 = co2_emissions.loc[my_countries, '1990':'2020':6].copy()
grouped_barplot(co2, 'Country Name', 'CO2 Emissions in kt',
                'CO2 Emissions of my countries')

#Variation of CO2 emissions of my countries over years
co2_tr = co2_emissions_tr.loc['1990':'2020', my_countries].copy()
plt.figure()
for i in co2_tr.columns:
    plt.plot(co2_tr.index, co2_tr[i], label=i)
plt.xticks(co2_tr.index[::5])
plt.xlabel('Year')
plt.ylabel('CO2 Emissions in kt')
plt.title('CO2 Emissions of my countries')
plt.legend(title='Country', bbox_to_anchor=(1, 1))
plt.savefig('CO2_Emissions', bbox_inches='tight',dpi=450)
plt.show()

my_data_list = [forest_area, agricultural_land,
                population_total, urban_population,co2]
my_data_list_tr = [forest_area_tr, agricultural_land_tr,
                   population_total_tr, urban_population_tr,co2_tr]
indicator_list = ['Forest Area', 'Agricultural Land Area',
                  'Total Population', 'Urban Population','CO2 Emissions']

#Creating data of australia and plotting correlation heatmap
australia = country_wise_data(my_data_list_tr, indicator_list, 'Australia')
correlation_coefficient_heatmap(australia, 'Australia', 'Spectral_r')

#Creating data of china and plotting correlation heatmap
china = country_wise_data(my_data_list_tr, indicator_list, 'China')
correlation_coefficient_heatmap(china, 'China', 'ocean')

#Creating data of germany and plotting correlation heatmap
germany = country_wise_data(my_data_list_tr, indicator_list, 'Germany')
correlation_coefficient_heatmap(germany, 'Germany', 'jet')

#Creating data of india and plotting correlation heatmap
india = country_wise_data(my_data_list_tr, indicator_list, 'India')
correlation_coefficient_heatmap(india, 'India', 'gist_rainbow')
