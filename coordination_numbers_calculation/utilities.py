import numpy as np
import pandas as  pd 
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def hanging_line(point1, point2):
    '''This function draws a hanging line that completes the curved section of the stereographic triangle'''
    a = (point2[1] - point1[1])/(np.cosh(point2[0]) - np.cosh(point1[0]))
    b = point1[1] - a*np.cosh(point1[0])
    x = np.linspace(point1[0], point2[0], 100)
    y = a*np.cosh(x) + b

    return (x,y)


def chirality(miller_idx):
    '''This function determines chirality of the miller index input'''

    chirality= []
    R = []
    S = []
    for j in 1000*miller_idx: 
        i = np.absolute(j)
        # checks if h≠k≠l≠0
        if  i[0]==i[1] or i[0]==i[2] or i[1]==i[2] or i[0]==0 or i[1]==0 or i[2]==0: 
            chirality.append(0)
            R.append(0)
            S.append(0)
        ## checks if   h
        ##            / \
        ##           k - l   is clockwise or counter clockwise based on magnitude
        elif (i[0]>i[1] and i[1] >i[2]) or (i[1]>i[2] and i[2] >i[0]) or (i[2]>i[0] and i[0] >i[1]) :
            ## count the number of negative indices inside each miller index, if odd,
            ## the chirality switchs from R to S and vice versa. If even, it stays the same. 
            if np.sum(j < 0) %2 ==0:
                R.append(-1)
                S.append(1)
                chirality.append(1)
            else: 
                R.append(1)
                S.append(-1)
                chirality.append(1)
                
        else: 
            if np.sum(j < 0) %2 ==0:
                R.append(1)
                S.append(-1) 
                chirality.append(1)
            else:
                R.append(-1)
                S.append(1) 
                chirality.append(1)
                

    chirality = np.array(chirality).reshape(-1,1)
    R = np.array(R).reshape(-1,1)
    S = np.array(S).reshape(-1,1)

    return chirality, R, S   



def half_time(dta_path, lta_path, coordinates_df):
    """a function to get extract the half-time data for D-TA and L-TA on the (100) and the (110) samples"""
    
    # L-TA half_time data
    df_1= pd.read_csv(dta_path)
    df_1 = df_1.dropna()
    df_1=df_1.drop(columns= "time(sec)")
    df_1 = df_1.T
    df_1 = df_1.reset_index(drop=True)
    df_1 = pd.concat([coordinates_df, df_1], axis=1)
    if 0 in df_1.columns:
        df_1 = df_1.rename(columns={0:"half_time"})
    else: 
        df_1 = df_1.rename(columns={2:"half_time"})
    
    # L-TA half_time data
    df_2= pd.read_csv(lta_path)
    df_2 = df_2.dropna()
    df_2=df_2.drop(columns= "time(sec)")
    df_2 = df_2.T
    df_2 = df_2.reset_index(drop=True)
    df_2 = pd.concat([coordinates_df, df_2], axis=1)
    if 0 in df_2.columns:
        df_2 = df_2.rename(columns={0:"half_time"})
    else: 
        df_2 = df_2.rename(columns={2:"half_time"})
    
    # half_time difference calculation
    df_3= pd.DataFrame(df_1["half_time"]-df_2["half_time"])
    df_3 = pd.concat([coordinates_df, df_3], axis=1)
    
    return df_1, df_2, df_3



def predictions_df(sample_df, model, idx_train, y_train, idx_test,X_test, num_samples):
    ''' This function returns the combines the test data predictions and the training labels into a    dataframe that has the same order as the original data'''
    
    # create a dataframe of the training data, keeping the training split indices
    df_train = pd.DataFrame({"x":np.array([sample_df["x"].values] * num_samples).reshape(-1,)[idx_train],
    "y": np.array([sample_df["y"].values] * num_samples).reshape(-1,)[idx_train],
    "half_time" : y_train.reshape(-1,)})
    df_train["idx"]= idx_train
    df_train = df_train.set_index("idx")

     # create a dataframe of the test data, keeping the test split indices
    df_test = pd.DataFrame({"x":np.array([sample_df["x"].values] * num_samples).reshape(-1,)[idx_test],
    "y": np.array([sample_df["y"].values] * num_samples).reshape(-1,)[idx_test], 
    "half_time" : model.predict(X_test).reshape(-1,)})
    df_test["idx"] = idx_test
    df_test = df_test.set_index("idx")

    # combine the two dataframes sorting buy the order of the indices to get the original order of data points
    full_df = pd.concat([df_train, df_test])
    full_df = full_df.sort_index( axis = 0 )
    return full_df



def sample_contour_plot(label, title):
    xyz_df= pd.read_csv("Coordinates_Cu.csv")
    xyz_df = xyz_df.drop(columns ="Unnamed: 0")

    X,Y = np.meshgrid(xyz_df["x"].values,xyz_df["y"].values)
    Z = griddata((xyz_df["x"].values,xyz_df["y"].values),label.reshape(-1,), (X, Y),method='linear')
    fig, ax = plt.subplots(figsize=(6, 4))
    norm = plt.Normalize(vmin=-600, vmax=600)
    ax.contourf(X,Y,Z,levels = 100,cmap='bwr')
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="bwr"), ax = ax,orientation="vertical")
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("$t_{1/2}$ (D-TA) - $t_{1/2}$ (L-TA) (s)", fontsize= 14,rotation=270)
    ax.scatter(xyz_df["x"].values,xyz_df["y"].values, c = "w", s = 1)
    rad= np.linspace(4.5,-4.5,50)
    ax.set_xlabel("X from center (mm)",fontsize= 14)
    ax.set_ylabel("Y from center (mm)",fontsize= 14)
    ax.set_title(title,fontsize= 14)
    plt.show()

def hanging_line(point1, point2):
    a = (point2[1] - point1[1])/(np.cosh(point2[0]) - np.cosh(point1[0]))
    b = point1[1] - a*np.cosh(point1[0])
    x = np.linspace(point1[0], point2[0], 100)
    y = a*np.cosh(x) + b

    return (x,y)


# Define the Harmonic Functions
def cubic_harmonic(x, y, z, degree):
    """Returns the cubic harmonic function of the given degree."""

    if degree == 9:
        # 9th Degree Cubic Harmonic
        cube_harm = 110*x*y*z*(x**2-y**2)*(y**2-z**2)*(x**2-z**2)
        
    elif degree == 13:
        # 13th Degree Cubic Harmonic
        cube_harm = 4.2*x*y*(x**2 - y**2)*z*(x**2 - z**2)*(y**2 - z**2)*(378 + 575*x**4 + 
       575*y**4 - 828*z**2 + 575*z**4 + 23*y**2*(-36 + 25*z**2) + 
       23*x**2*(-36 + 25*y**2 + 25*z**2))

    elif degree ==15:
        # 15th Degree Cubic Harmonic
        cube_harm = 63.8*x*y*z*(x**2 - y**2)*(x**2 - z**2)*(y**2 - z**2)*(-154 + 435*x**6 + 435*y**6         +550*z**2 - 825*z**4 + 435*z**6 + 15*y**4*(-55 + 29*z**2) + 15*x**4*(-55 + 29*y**2 +                 29*z**2) + 5*y**2*(110 - 165*z**2 + 87*z**4)+ 5*x**2*(110 + 87*y**4 - 165*z**2 + 87*z**4 +           3*y**2*(-55 + 29*z**2)))

    elif degree == 17:
        # 17th Degree Cubic Harmonic
        cube_harm = 1.6*x*y*z*(5005*x**4*(y**2 - z**2) - 25740*x**6*(y**2 - z**2) + 62205*x**8*             (y**2 - z**2) - 70122*x**10*(y**2 - z**2) + 29667*x**12*(y**2 - z**2) + y**2*z**2*                   (5005*y**2- 25740*y**4 + 62205*y**6 - 70122*y**8 + 29667*y**10 + z**2*(-5005 + 25740*z**2 -         62205*z**4+ 70122*z**6 - 29667*z**8)) + x**2*(-5005*y**4 + 25740*y**6 - 62205*y**8 +                 70122*y**10 -29667*y**12 + z**4*(5005 - 25740*z**2 + 62205*z**4 - 70122*z**6 + 29667*z**8)))

    elif degree == 19: 
        # 19th Degree Cubic Harmonic
        cube_harm = 11.4*x*y*z*(-2457*x**4*(y**2 - z**2) + 16965*x**6*(y**2 - z**2) - 58435*x**8*
        (y**2 - z**2) + 105183*x**10*(y**2 - z**2) - 94395*x**12*(y**2 - z**2) + 33263*x**14*(y**2 -
        z**2) + y**2*z**2*(-2457*y**2 + 16965*y**4 - 58435*y**6 + 105183*y**8 - 94395*y**10 + 
        33263*y**12 + z**2*(2457 - 16965*z**2 + 58435*z**4 - 105183*z**6 + 94395*z**8 - 
        33263*z**10)) +x**2*(2457*y**4 - 16965*y**6 + 58435*y**8 - 105183*y**10 + 94395*y**12 -
        33263*y**14 + z**4*(-2457 + 16965*z**2 - 58435*z**4 + 105183*z**6 - 94395*z**8 +
        33263*z**10)))

    else:
        print("Error: No Cubic Harmonic is defined for this Degree")
    
    return cube_harm.reshape(-1,1)