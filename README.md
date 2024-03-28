### This repository includes Jupyter notebooks to showcase the code to fit two models to experimental data about the enantiospecific dehydrogenation of tartaric acid over chiral copper surfaces. 

### Abstract:
##### Enantiospecific heterogeneous catalysis utilizes chiral catalytic surfaces to impart different reaction kinetics to two adsorbed enantiomers, thereby enabling the reactive purification of one enantiomer with respect to the other. Intrinsically chiral surfaces with different crystallographic orientations exhibit surface structures which result in enantiospecific reaction kinetics. This work models the enantiospecific decomposition kinetics of tartaric acid (TA) on Cu〖(hkl)〗^(R/S) surfaces as a function of surface orientation. The complementary experimental data was collected using curved single crystal samples that allow concurrent measurement of TA decomposition kinetics on hundreds of surfaces spanning continuous regions of surface orientation space. 
##### The goal of this work is to build models to understand what makes one surface orientation more enantiospecific than another and to predict the surface orientation that has the highest enantiospecificity. Two models were built for this purpose: the first utilizes Generalized Coordination Number (GCN) features to correlate the surface structure with observed enantiospecificity, while the second employs an expansion using chiral cubic harmonic functions as a basis set to approximate enantiospecific behavior.

### Notebooks: 
1) generalized-coordination-model
Shows the code that takes the vectors that represent the fraction of atoms on each surface orientation with each of the 58 unqiue GCN values and corresponding halftime difference labels which are stored in the .csv file named experimental_data_with_gcn_features and fits a linear regression model with Ridge regularization to the experimental data. The notebook includes cells for evaluating the fitted model performance on the training data and the testing data. The notebook also includes sections for making predictions about surface orientations that span the orientation space and uses these predictions to determine the location of the next experiment. 
2) cubic-harmonics-model
Takes in the Miller indices of the different surface orientations and the corresponding halftime difference labels from the experimental_data_with_gcn_features.csv file. The code then computes the value of the 5 cubic harmonics functions (L=9,13,15,17,19) at the orientations that tested experimentally. The values of these 5 functions are used as a feature vector representation of each surface orientation. The model is fitted to the to the cubic harmonics features and experimental halftime difference labels and its performance is evaluated in the code. This model is also used to determine the location of the next experiment. This model's prediction of the location of the next experiment agrees with the prediction of the GCN model, giving us confidence in the model predictions. However, we can only verify the predictions by running the experiment.
3) harmonic-functions-visualization
shows the method for obtaining the 3D spherical visualization of the cubic harmonics functions used in this work. These visualizations are used in the paper figures.
4) rotational-averaging
This notebook shows the rotational averaging method that is used to quantify the experimental error in the data.
5) gcn-features-calculator
This notebook computes the GCN feature vectors used by the GCN model based on method that computes the coordination numbers and generalized coordination numbers based on the connectivity matrix of the surface structures.
6) slab generation
This notebook used Pymatgen slab generator class to generate the slabs that are used to represent the surface structures computationally to calculate the coordination numbers. 

