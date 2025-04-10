# GPR_Stocks
Experimental project on using GPR as a long term equity statistical model.

GPRs incorporate uncertainties in the model developement. This can prove a useful tool when dealing with noisy data (i.e. astrophysics, stock market). The project aims at exploring GPR as an analysis tool. Summary of the project plan:

-Use method of moving averages to get clean data

-Undersample the data (still debating whether to do this)

-Stop undersampled data early (i.e. test predictive accuracy)

-Develop a model using the undersampled data and a white kernel to create the model (project is currently here)

-Get a standard deviation from the averaged data

-Develop a new model using standard deviation in the training data (might need to switch to tensorflow for this)

-Incorporate MCMC for hyperparameters (TBC if necessary)
