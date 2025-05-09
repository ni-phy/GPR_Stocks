# GPR_Stocks
Experimental project on using GPR as a long term equity statistical model.

GPRs incorporate uncertainties in the model developement. This can prove a useful tool when dealing with noisy data (i.e. astrophysics, stock market). Essentially each point (or each day's close here) is modeled as a gaussian distribuition which is correlated the other points using the kernel. This may remind sound similar random walks where the next step is only depended on the currect step, but here the next step is correlated to all previous steps. The project aims at exploring GPR as an analysis tool. As essential fundamental flaw is that the distribution of the next day's close is fat-tailed, here it is modelled as Gaussian.

Summary of the project plan:

-Use method of moving averages to get clean data

-Undersample the data (still debating whether to do this)

-Stop undersampled data early (i.e. test predictive accuracy)

-Develop a model using the undersampled data and a white kernel to create the model 

-Get a standard deviation from the averaged data (project is currently here)

-Develop a new model using standard deviation in the training data (might need to switch to tensorflow for this)

-Incorporate MCMC for hyperparameters (TBC if necessary)
