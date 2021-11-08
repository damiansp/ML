import requests
import urllib.request
import pandas as pd
import os
import json
import re
import numpy as np
from datetime import timedelta
import datetime
from bs4 import BeautifulSoup, SoupStrainer

HOW_MANY = 10
START = '2020-03-30 12:00'
END = '2020-04-01 12:00'

"""
Request with the header information
"""
def pull_request(value):
    url = f'https://labondemand.com/api/v3/{value}'
    data = requests.get( url,
                   params={'q': 'requests+language:python'},
                   headers={'api_key':'ca17f725-b345-4019-b80b-ab375db471ea'} )
    json_response = data.json()
    #f'Lab Profile ID: {json_response["Name"]}'
    return json_response
"""
Filter out Lab Profiles from the response pull_request
"""
def filter_labProfiles():
    for name in pull_request("catalog")['LabProfiles']:
            if ( name['DevelopmentStatusId'] == 10 ):
                #print  ( f'{name["DeliveryRegions"]}' ) 
                print ( f'Profile ID: {name["Id"]}  : Title  {name["Name"]} : RAM {name["Ram"]} ' )
        ## Pull raw data
        #print ( pull_request()['LabProfiles'] ) 
"""
Filter out Lab Series from the response pull_request
"""
def filter_labSeries():
    for series in pull_request("catalog")["LabSeries"]:
        print ( f'Lab Series : {series["Name"]}' )
        print (len(series))
"""
FIND which Profiles have less than 100 launches
Get Profile ID's and put into an array
Then pull LabResults within time frame
Match Profile ID == LabResults LabProfileID
If lab CompletionStatus is 4 = Complete, then print list
"""
### Find the completed profile ID's and return the value
def profile_Id():
    new_list = []
    for labid in pull_request("catalog")['LabProfiles']:
        new_list.append( labid["Id"] )      ## Return list of completed profile ID's
    return new_list
### Ask user date range for finding the launch ID's
def date_Range():
    #input_Start_Date = input( "Enter start date format YYYY-DD-MM HH:MM " )
    #input_End_Date = input( "Enter end date format YYYY-DD-MM HH:MM " )
    input_Start_Date = START
    input_End_Date = END
    ### Strip out the dilimiters - : and white space
    strip_out_delimiter_start = re.split(r'[-:\s]\s*', input_Start_Date)
    strip_out_delimiter_end = re.split(r'[-:\s]\s*', input_End_Date)
    ### Convert the date and time from string to int
    conv_date_start = [int(i) for i in strip_out_delimiter_start ]
    conv_date_end = [int(i) for i in strip_out_delimiter_end ]
    ### Apply the convertion to the datetime for the Epoch
    timestamp_start = datetime.datetime( *conv_date_start, 0).timestamp()
    timestamp_end = datetime.datetime( *conv_date_end, 0).timestamp()
    ### Strip off after the decimal
    #print ( timestamp_start )
    return int(timestamp_start),  int(timestamp_end)
### Find the launches within date range
def launch_ID():
    start , stop = date_Range()
    #print (start,stop)
    ## Resquest date range for value
    return pull_request(f'{"Results"}/?start={start}&end={stop}')     ## Pass in the Results start and end date range
### Parse Labs launched and extract LabProfile ID's
### Then put LabProfile ID's in a dataframe
### Use Pandas to groupby and value_count of ID's 
### Compare LabProfiles that are in a Complete status
def labProfiles(num):
    ### Parse the launch ID's from the date range 
    def parse_launch_id():
        ### Empty List
        new_arr = []
         ### Pull the launches from Results
        for instance in launch_ID()["Results"]:
            pd_id = instance["LabProfileId"]
            ### Append Empty list with launched profile id's
            new_arr.append(pd_id)
            ### Move profile ID's into a panda series.
            pd_series = pd.Series(new_arr)
            # return  f'{new_arr}'
        return pd_series    
    ### Pull the Profile ID's 
    def completed_prfiles():
        ### Empty List
        new_arr = []
        ### Iterate through the profile_id list with completed profile id's
        for new_id in profile_Id():
            ### Append Empty list with completed profile id's
            new_arr.append(new_id)
            ### Move profile ID's into a panda series.
            pd_new_series = pd.Series(new_arr)
        return pd_new_series  
    if num == 1:
        return parse_launch_id
    else:
        return completed_prfiles
labProfiles_launchID = labProfiles(1)
labProfiles_completed = labProfiles(2)
### Only the Launch IDs
#labProfiles_launchID()
### Read through pandas series labProfiles_launchID
def panda(num):
    ### Launch ID's are pulled from labProfiles_launchID()
    launch = labProfiles_launchID()
    ### Completed ID's are pulled from labProfiles_launchID()
    completed = labProfiles_completed()
    ### Make Launch and Completed one DF
    df = pd.concat([launch, completed] ,axis=1  )
    ### Rename the columns in the df 'launch', 'completed' 
    df.columns = ['launch', 'completed']
    #gby_launch = df.groupby('launch')
    ### Put Profile ID's in list
    new_arr = []
    ### Loop through launched profile ID's that are greater than certain value and print those ID's
    for pro_id, pro_value in zip(df['launch'] , df['launch'].value_counts()):
        ### If profile value is greater than .... append to new_arr
        if pro_value < int(num):
            new_arr.append(pro_id)
            #new_val = df.groupby('launch')
    return list(dict.fromkeys(new_arr))
    #print( len(list(dict.fromkeys(new_arr))) )  
### Remove duplicates
def remove_dups(elem_one, elem_two):
    #print(f'In remove_dups: elem_one:\n{elem_one}\nelem_two:\n{elem_two}')
    # Create an empty list to store unique elements
    uniqueList = []
    # Iterate over the original list and for each element
    # add it to uniqueList, if its not already there.
    for elem_two in elem_one:
        if elem_two not in uniqueList:
            uniqueList.append(elem_two)
    # Return the list of unique elements        
    return uniqueList
### Compare the Completed and Launched lists
### Sort and then drop the non matching ID's in a new array


def comp_list():
    ### How many launches?
    #how_many = input( "How many launches: ")
    how_many = HOW_MANY
    ### Count how many completed labs
    launch  = list ( set( panda(how_many) ))
    #print(f'launch: {launch}')
    complete = list ( set( profile_Id() ))
    #print(f'complete: {complete}')
    """
    Remove the duplicates with remove_dups()
    """
    #new_list = remove_dups(launch, complete)
    new_list = remove_dups(complete, launch)
    print ( new_list ) 
    print ( len( new_list )  )
    #print ( len( complete ), len( launch ) )       
    # print ( complete - launch  )
    # print ( len(complete), len(launch) 


if __name__ == '__main__':
    comp_list()
