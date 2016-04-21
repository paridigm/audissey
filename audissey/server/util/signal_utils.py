#####
#
# Contains functions that assist with 
# extracting info from signals
#
#####

import numpy as np
import scipy.signal as signal

""" returns the closest index to a value """
def search_for_nearest(val, search_list):
    min_dist = None
    ans = -1
    for i in range(0, len(search_list)):
        if(min_dist != None):
            if( abs(val - search_list[i]) < min_dist ):
                min_dist = abs(val - search_list[i])
                ans = i
            else:
                break;
        else:
            min_dist = abs(val - search_list[i])
            ans = i
        #endif
    #endif
    return ans



""" searches for peaks above a certain threshold """
def search_peaks(array, thresh, cutoff_for_delay=0):
    
    peaks = []
    thresh_offset = 0       # compensate for delay
    if(cutoff_for_delay > 0):
        thresh_offset = int((1/cutoff_for_delay)*6000)
    found = False
    max_a = 0
    max_i = 0

    for i in range(1, array.size-1):
        
        if(array[i] < thresh):   # below thresh?
            if(found):              # no peak and continue
                if(max_i >= thresh_offset):
                    peaks.append(max_i - thresh_offset) 
                max_a = 0
                max_i = 0
                found = False
            continue

        else:                       #above thresh
            
            #if peak
            if(array[i-1] < array[i] and array[i] > array[i+1]):

                # no peak found yet since crossing?
                if(not found):      
                    max_a = array[i]
                    max_i = i
                    found = True
                else:
                    # otherwise compare with max
                    if(array[i] > max_a):
                        max_a = array[i]
                        max_i = i
                #endif
                        
            #endif                
                     
    #endfor
                     
    return peaks
    
    
""" performs a nearest matching between two lists """
def compare(list1, list2, MAX_DIST):

    res = [] #result

    # if a list is empty, return the other list
    if( len(list1) == 0 ):
        return list2

    if( len(list2) == 0 ):
        return list1

    #let a be the smaller array
    a = None 
    b = None
    if( len(list1) < len(list2) ):
        a = list1
        b = list2
    else:
        a = list2
        b = list1
       
    l = [-1]*len(a) #list of locked in values
    
    # get nearest for each of the smaller array
    l[0] = search_for_nearest(a[0], b)
    for i in range(1, len(a)):
        j = search_for_nearest(a[i], b)
        
        if( abs(a[i] - b[j]) < MAX_DIST ): # must be within dist        
            if(l[i-1] == j):           # if in conflict with prev
                if(abs(a[i]-b[j]) < abs(a[i]-b[l[i-1]])):
                    l[i] = j
                    l[i-1] = -1
            else:
                l[i] = j        # no conflict.. just set                
    #endfor
    
    #populate result
    for i in range(0, len(a)):
        if(l[i] != -1):
            # r.append( max(a[i], b[l[i]]) )
            res.append( int( (a[i] + b[l[i]])/2 ) ) # temp
    
    return res
    

""" searches downward from peaks to get to a minimum threshold value """
def traverse_starts_ends(peaks, array, thresh):
    
    starts = []
    ends = []
    
    for x in peaks:
        
        # traverse left for starts
        i = x
        while( i > 0 and array[i] > thresh ):
            i = i - 1
        starts.append(i)
        
        #traverse right for ends
        i = x
        while( i < array.size-1 and array[i] > thresh ):
            i = i + 1
        ends.append(i)
        
    #endfor
        
    return (starts, ends)



#binary search for the closest value in b
'''
val = a[0]
low = 0
high = len(b)
while(low < high):
    mid = int( (low+high)/2 )
    if(b[mid] >= val):
        high = mid
    else:
        low = mid+1
high = high
'''

