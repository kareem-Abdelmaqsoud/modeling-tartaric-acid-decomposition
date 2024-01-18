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

    if degree == 0:
        # 0th Degree Cubic Harmonic
        cube_harm = np.ones(len(x))

    elif degree == 9:
        # 9th Degree Cubic Harmonic
        cube_harm = x*y*z*(x**2-y**2)*(y**2-z**2)*(x**2-z**2)

    elif degree == 13:
        # 13th Degree Cubic Harmonic
        cube_harm = x*y*z*(378*x**4*(y**2-z**2)-828*x**6*(y**2-z**2)+575*x**8*(y**2-z**2)
        + y**2*z**2*(378*y**2 -828*y**4 +575*y**6 -378*z**2 +828*z**4 -575*z**6)
        + x**2*(-378*y**4 +828*y**6 -575*y**8 +378*z**4 -828*z**6+575*z**8))

    elif degree ==15:
        # 15th Degree Cubic Harmonic
        cube_harm = x*y*z*(x**2 - y**2)*(x**2 - z**2)*(y**2 - z**2)*(-154 + 435*x**6 + 435*y**6 +               550*z**2 - 825*z**4 + 435*z**6 + 15*y**4*(-55 + 29*z**2) + 15*x**4*(-55 + 29*y**2 + 29*z**2) +         5*y**2*(110 - 165*z**2 + 87*z**4)+ 5*x**2*(110 + 87*y**4 - 165*z**2 + 87*z**4 + 3*y**2*(-55 +           29*z**2)))

    elif degree == 17:
        # 17th Degree Cubic Harmonic

        cube_harm = x*y*z*(5005*x**4*(y**2 - z**2) - 25740*x**6*(y**2 - z**2) + 62205*x**8*(y**2 -             z**2) - 70122*x**10*(y**2 - z**2) + 29667*x**12*(y**2 - z**2) + y**2*z**2*(5005*y**2 -                 25740*y**4 + 62205*y**6 - 70122*y**8 + 29667*y**10 + 
        z**2*(-5005 + 25740*z**2 - 62205*z**4 + 70122*z**6 - 29667*z**8)) + 
        x**2*(-5005*y**4 + 25740*y**6 - 62205*y**8 + 70122*y**10 - 29667*y**12 + z**4*(5005 -                   25740*z**2 + 62205*z**4 - 70122*z**6 + 29667*z**8)))

    elif degree ==19: 
        # 19th Degree Cubic Harmonic

        cube_harm = x*y*z*(-2457*x**4*(y**2 - z**2) + 16965*x**6*(y**2 - z**2) - 58435*x**8*(y**2 -             z**2) + 105183*x**10*(y**2 - z**2) - 94395*x**12*(y**2 - z**2) + 33263*x**14*(y**2 - z**2) +           y**2*z**2*(-2457*y**2 + 16965*y**4 - 58435*y**6 + 105183*y**8 - 94395*y**10 + 
        33263*y**12 + z**2*(2457 - 16965*z**2 + 58435*z**4 - 105183*z**6 + 94395*z**8 - 33263*z**10)) +
        x**2*(2457*y**4 - 16965*y**6 + 58435*y**8 - 105183*y**10 + 94395*y**12 - 33263*y**14 + 
        z**4*(-2457 + 16965*z**2 - 58435*z**4 + 105183*z**6 - 94395*z**8 + 33263*z**10)))
    
    elif degree == 21:
        # 21st Degree Cubic Harmonic
 
        cube_harm = x*y*z*(18564*x**4*(y**2 - z**2) - 164424*x**6*(y**2 - z**2) + 753610*x**8*(y**2 -           z**2) - 1918280*x**10*(y**2 - z**2) + 2729860*x**12*(y**2 - z**2) - 2027896*x**14*(y**2 - z**2)         + 611351*x**16*(y**2 - z**2) + y**2*z**2*(18564*y**2 - 164424*y**4 + 753610*y**6 - 1918280*y**8         + 2729860*y**10 - 2027896*y**12 + 611351*y**14 + z**2*(-18564 + 164424*z**2 - 753610*z**4 +             1918280*z**6 - 2729860*z**8 + 2027896*z**10 - 611351*z**12)) + 
        x**2*(-18564*y**4 + 164424*y**6 - 753610*y**8 + 1918280*y**10 - 2729860*y**12 + 2027896*y**14 -         611351*y**16 + z**4*(18564 - 164424*z**2 + 753610*z**4 - 1918280*z**6 + 2729860*z**8 -                 2027896*z**10 + 611351*z**12)))

    elif degree == 23:
        # 23th Degree Cubic Harmonic
        cube_harm = x*y*z*(-3876*x**4*(y**2 - z**2) + 42636*x**6*(y**2 - z**2) - 248710*x**8*(y**2 -           z**2) + 836570*x**10*(y**2 - z**2) - 1673140*x**12*(y**2 - z**2) + 1959964*x**14*(y**2 - z**2)         - 1239389*x**16*(y**2 - z**2) + 326155*x**18*(y**2 - z**2) + 
        y**2*z**2*(-3876*y**2 + 42636*y**4 - 248710*y**6 + 836570*y**8 - 1673140*y**10 + 1959964*y**12         - 1239389*y**14 + 326155*y**16 + z**2*(3876 - 42636*z**2 + 248710*z**4 - 836570*z**6 +                 1673140*z**8 - 1959964*z**10 + 1239389*z**12 - 326155*z**14)) + x**2*(3876*y**4 - 42636*y**6 +         248710*y**8 - 836570*y**10 + 1673140*y**12 - 1959964*y**14 + 1239389*y**16 - 326155*y**18 + 
        z**4*(-3876 + 42636*z**2 - 248710*z**4 + 836570*z**6 - 1673140*z**8 + 1959964*z**10 -                   1239389*z**12 + 326155*z**14)))

    elif degree == 25:
        # 25th Degree Cubic Harmonic
        cube_harm = x*y*z*(5720330*x**4*(y**2 - z**2) - 106234700*x**6*(y**2 - z**2) + 1088905675*x**8*         (y**2 - z**2) - 6810610040*x**10*(y**2 - z**2) + 27504386700*x**12*(y**2 - z**2) -                     73868924280*x**14*(y**2 - z**2) + 133072694475*x**16*(y**2 - z**2) - 158753389900*x**18*(y**2 -         z**2) + 120198995210*x**20*(y**2 - z**2) - 52260432700*x**22*(y**2 - z**2) + 9929482213*x**24*         (y**2 - z**2) + y**2*z**2*(5720330*y**2 - 106234700*y**4 + 1088905675*y**6 - 6810610040*y**8 +         27504386700*y**10 - 73868924280*y**12 + 133072694475*y**14 - 158753389900*y**16 +                       120198995210*y**18 - 52260432700*y**20 + 9929482213*y**22 + z**2*(-5720330 + 106234700*z**2 -           1088905675*z**4 + 6810610040*z**6 - 27504386700*z**8 + 73868924280*z**10 - 
        133072694475*z**12 + 158753389900*z**14 - 120198995210*z**16 + 52260432700*z**18 -                     9929482213*z**20)) + x**2*(-5720330*y**4 + 106234700*y**6 - 1088905675*y**8 + 6810610040*y**10         - 27504386700*y**12 + 73868924280*y**14 - 133072694475*y**16 + 158753389900*y**18 -                     120198995210*y**20 + 52260432700*y**22 - 9929482213*y**24 + z**4*(5720330 - 106234700*z**2 +           1088905675*z**4 - 6810610040*z**6 + 27504386700*z**8 - 73868924280*z**10 + 
        133072694475*z**12 - 158753389900*z**14 + 120198995210*z**16 - 52260432700*z**18 +                     9929482213*z**20)))

    elif degree == 27:
        # 27th Degree Cubic Harmonic
        cube_harm = x*y*z*(-31461815*x**4*(y**2 - z**2) + 498894495*x**6*(y**2 - z**2) -                       4323752290*x**8*(y**2 - z**2) + 22562125586*x**10*(y**2 - z**2) - 
        74628569246*x**12*(y**2 - z**2) + 159918362670*x**14*(y**2 - z**2) - 221063618985*x**16*(y**2 -         z**2) + 190037146145*x**18*(y**2 - z**2) - 92303756699*x**20*(y**2 - z**2) + 19336360099*x**22*         (y**2 - z**2) + y**2*z**2*(-31461815*y**2 + 498894495*y**4 - 4323752290*y**6 + 22562125586*y**8         - 74628569246*y**10 + 159918362670*y**12 - 221063618985*y**14 + 190037146145*y**16 -                   92303756699*y**18 + 19336360099*y**20 + z**2*(31461815 - 498894495*z**2 + 4323752290*z**4 -             22562125586*z**6 + 74628569246*z**8 - 159918362670*z**10 + 221063618985*z**12 -                         190037146145*z**14 + 92303756699*z**16 - 19336360099*z**18)) + x**2*(31461815*y**4 -                   498894495*y**6 + 4323752290*y**8 - 22562125586*y**10 + 74628569246*y**12 - 159918362670*y**14 
        + 221063618985*y**16 - 190037146145*y**18 + 92303756699*y**20 - 19336360099*y**22 + z**4*               (-31461815 + 498894495*z**2 - 4323752290*z**4 + 22562125586*z**6 - 74628569246*z**8 +                   159918362670*z**10 - 221063618985*z**12 + 190037146145*z**14 - 92303756699*z**16 +                     19336360099*z**18)))
    
    elif degree == 29:
        cube_harm = x * y * z * (143702130 * x**8 * (y**6 - 7 * y**4 * z**2 + 7 * y**2 * z**4 - z**6) -
           736800012 * x**10 * (y**6 - 7 * y**4 * z**2 + 7 * y**2 * z**4 - z**6) +
           2314307730 * x**12 * (y**6 - 7 * y**4 * z**2 + 7 * y**2 * z**4 - z**6) -
           4496369304 * x**14 * (y**6 - 7 * y**4 * z**2 + 7 * y**2 * z**4 - z**6) +
           5256784701 * x**16 * (y**6 - 7 * y**4 * z**2 + 7 * y**2 * z**4 - z**6) -
           3381557410 * x**18 * (y**6 - 7 * y**4 * z**2 + 7 * y**2 * z**4 - z**6) +
           917851297 * x**20 * (y**6 - 7 * y**4 * z**2 + 7 * y**2 * z**4 - z**6) +
           y**2 * z**2 * (143702130 * y**6 * z**4 - 736800012 * y**8 * z**4 +
           2314307730 * y**10 * z**4 - 4496369304 * y**12 * z**4 +
           5256784701 * y**14 * z**4 - 3381557410 * y**16 * z**4 +
           917851297 * y**18 * z**4 + 209 * z**2 * (7 - 130 * z**2) +
           209 * y**2 * (-7 + 4797 * z**4) +
           y**4 * (27170 - 1002573 * z**2 - 143702130 * z**6 + 736800012 * z**8 -
           2314307730 * z**10 + 4496369304 * z**12 - 5256784701 * z**14 +
           3381557410 * z**16 - 917851297 * z**18)) +
           x**6 * (-143702130 * y**8 + 736800012 * y**10 - 2314307730 * y**12 +
           4496369304 * y**14 - 5256784701 * y**16 + 3381557410 * y**18 -
           917851297 * y**20 + 334191 * y**4 * (-3 + 344 * z**2) -
           5434 * y**2 * (-5 + 21156 * z**4) +
           z**2 * (-27170 + 1002573 * z**2 + 143702130 * z**6 - 736800012 * z**8 +
           2314307730 * z**10 - 4496369304 * z**12 + 5256784701 * z**14 -
           3381557410 * z**16 + 917851297 * z**18)) +
           x**4 * (1005914910 * y**8 * z**2 - 5157600084 * y**10 * z**2 +
           16200154110 * y**12 * z**2 - 31474585128 * y**14 * z**2 +
           36797492907 * y**16 * z**2 - 23670901870 * y**18 * z**2 +
           6424959079 * y**20 * z**2 - 334191 * y**6 * (-3 + 344 * z**2) -
           209 * z**2 * (-7 + 4797 * z**4) +
           y**2 * (-1463 + 114961704 * z**6 - 1005914910 * z**8 + 5157600084 * z**10 -
           16200154110 * z**12 + 31474585128 * z**14 - 36797492907 * z**16 +
           23670901870 * z**18 - 6424959079 * z**20)) +
           x**2 * (-1005914910 * y**8 * z**4 + 5157600084 * y**10 * z**4 -
           16200154110 * y**12 * z**4 + 31474585128 * y**14 * z**4 -
           36797492907 * y**16 * z**4 + 23670901870 * y**18 * z**4 -
           6424959079 * y**20 * z**4 + 209 * z**4 * (-7 + 130 * z**2) +
           5434 * y**6 * (-5 + 21156 * z**4) +
           y**4 * (1463 - 114961704 * z**6 + 1005914910 * z**8 - 5157600084 * z**10 +
           16200154110 * z**12 - 31474585128 * z**14 + 36797492907 * z**16 -
           23670901870 * z**18 + 6424959079 * z**20))
)

    else:
        print("Error: No Cubic Harmonic is defined for this Degree")
    
    return cube_harm.reshape(-1,1)